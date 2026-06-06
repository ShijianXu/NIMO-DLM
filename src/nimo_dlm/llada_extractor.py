"""
LLaDA hidden-state extractor with controlled masking for NIMO-DLM.

For each (sequence, target_position, timestep) triple the extractor:
  1. Randomly masks each non-target position with probability t  (independent Bernoulli)
  2. Runs a single forward pass of LLaDA-8B
  3. Collects residual-stream activations at the specified SAE layers
  4. Applies the SAE to produce sparse feature vectors  f^(l,t) ∈ ℝ^{d_SAE}
  5. Records y^(t) = log P_{dLLM}(x_j | x^(t)_{-j})  — log-prob of the correct token

The MASK token ID for LLaDA-8B is 126336.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .sae_wrapper import TopKSAE, AVAILABLE_LAYERS


MASK_ID = 126_336


class LLaDAExtractor:
    """
    Extracts (SAE feature, log-prob target) pairs from LLaDA-8B across
    controlled masking timesteps.

    Args:
        model_id:  HuggingFace model id (default: GSAI-ML/LLaDA-8B-Base)
        sae_dict:  {layer_index: TopKSAE}  — SAEs for each desired layer
        device:    Primary device for the LLM
        dtype:     Model dtype (torch.bfloat16 recommended for L40S)
    """

    def __init__(
        self,
        sae_dict:  Dict[int, TopKSAE],
        model_id:  str   = "GSAI-ML/LLaDA-8B-Base",
        device:    str   = "cuda",
        dtype:     torch.dtype = torch.bfloat16,
    ):
        self.sae_dict = sae_dict
        self.device   = device
        self.dtype    = dtype
        self.layers   = sorted(sae_dict.keys())

        print(f"Loading LLaDA from {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        ).eval()

        # Move SAEs to device
        for l, sae in self.sae_dict.items():
            self.sae_dict[l] = sae.to(device=device, dtype=torch.float32)

        # Locate transformer blocks (LLaDA: model.transformer.blocks)
        base = self.model
        if hasattr(base, "model"):
            base = base.model
        self._blocks = base.transformer.blocks  # nn.ModuleList of 32 blocks
        self._n_layers = len(self._blocks)

    # ──────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_batch(
        self,
        input_ids:   torch.Tensor,          # [B, L]  full (unmasked) token ids
        target_pos:  torch.Tensor,          # [B]     index of the target position per seq
        timesteps:   List[float],           # e.g. [0.1, 0.25, 0.5, 0.75, 1.0]
        n_mask_samples: int = 1,            # independent mask draws per (seq, t)
    ) -> Dict:
        """
        Returns a dict:
            "features"   : {layer: Tensor [B, T, d_sae]}   float32 SAE features
            "log_probs"  : Tensor [B, T]                    log-prob of correct token
            "timesteps"  : List[float]
            "target_ids" : Tensor [B]                       correct token id per seq
        where T = len(timesteps) * n_mask_samples.
        """
        B, L  = input_ids.shape
        T_eff = len(timesteps) * n_mask_samples

        target_ids = input_ids[
            torch.arange(B, device=input_ids.device), target_pos
        ]                                                     # [B]

        all_feats   = {l: [] for l in self.layers}
        all_logprob = []

        for t in timesteps:
            for _ in range(n_mask_samples):
                # ── Build masked input ────────────────────────────
                masked_ids = self._mask_input(input_ids, target_pos, t)  # [B, L]

                # ── Hook hidden states at SAE layers ──────────────
                hidden_states = {}
                hooks = self._register_hooks(hidden_states)

                outputs = self.model(
                    input_ids=masked_ids,
                    output_hidden_states=False,  # we use hooks instead
                )
                for h in hooks:
                    h.remove()

                # ── Logit of correct token at target position ─────
                logits_target = outputs.logits[
                    torch.arange(B), target_pos, :
                ]                                             # [B, vocab]
                log_probs = F.log_softmax(logits_target.float(), dim=-1)
                lp = log_probs[torch.arange(B), target_ids]  # [B]
                all_logprob.append(lp.cpu())

                # ── Apply SAEs ────────────────────────────────────
                for layer_idx in self.layers:
                    h_l = hidden_states[layer_idx]            # [B, L, d_model]
                    h_target = h_l[
                        torch.arange(B), target_pos, :
                    ].float()                                 # [B, d_model]
                    sae  = self.sae_dict[layer_idx]
                    f_l  = sae.encode(h_target)               # [B, d_sae]
                    all_feats[layer_idx].append(f_l.cpu())

        # Stack along the timestep dimension
        features   = {l: torch.stack(all_feats[l], dim=1)   # [B, T_eff, d_sae]
                      for l in self.layers}
        log_probs  = torch.stack(all_logprob, dim=1)         # [B, T_eff]

        return {
            "features":   features,
            "log_probs":  log_probs,
            "timesteps":  [t for t in timesteps for _ in range(n_mask_samples)],
            "target_ids": target_ids.cpu(),
        }

    # ──────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────

    def _mask_input(
        self,
        input_ids:  torch.Tensor,   # [B, L]
        target_pos: torch.Tensor,   # [B]
        t: float,
    ) -> torch.Tensor:
        """Independently mask each non-target position with probability t."""
        B, L = input_ids.shape
        masked = input_ids.clone()

        # Always mask the target position (it's what we're predicting)
        masked[torch.arange(B), target_pos] = MASK_ID

        # Randomly mask non-target positions
        if t > 0.0:
            noise = torch.rand(B, L, device=input_ids.device)
            # Mask positions where noise < t, but skip target_pos
            should_mask = noise < t
            target_one_hot = torch.zeros_like(should_mask)
            target_one_hot[torch.arange(B), target_pos] = True
            should_mask = should_mask & ~target_one_hot
            masked[should_mask] = MASK_ID

        return masked

    def _register_hooks(self, storage: dict) -> list:
        """Register forward hooks to capture residual-stream output of each SAE layer."""
        hooks = []
        for layer_idx in self.layers:
            block = self._blocks[layer_idx]

            def make_hook(l_idx):
                def hook_fn(module, inp, out):
                    # LLaDA blocks return a tuple; index 0 is the residual output
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    storage[l_idx] = h.detach()
                return hook_fn

            hooks.append(block.register_forward_hook(make_hook(layer_idx)))
        return hooks

    # ──────────────────────────────────────────────────────────────
    #  Dataset-level extraction
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_dataset(
        self,
        texts:             List[str],
        timesteps:         List[float] = None,
        n_targets_per_seq: int         = 5,
        max_length:        int         = 128,
        batch_size:        int         = 32,
        n_mask_samples:    int         = 1,
    ) -> Dict:
        """
        Tokenise texts, sample n_targets_per_seq positions per sequence, then
        run batched forward passes grouping all (seq, target) pairs together.
        This is ~40× faster than the naive one-target-at-a-time loop.

        Returns:
            {
              "features":  {layer: Tensor [N_total, T_eff, d_sae]},
              "log_probs": Tensor [N_total, T_eff],
              "timesteps": List[float],
              "meta":      {"seq_idx": ..., "pos_idx": ..., "target_id": ...}
            }
        """
        if timesteps is None:
            timesteps = [0.1, 0.25, 0.5, 0.75, 1.0]

        # Tokenise all texts at once
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = enc.input_ids.to(self.device)   # [N, L]
        N, L      = input_ids.shape

        # Find valid (non-pad, non-boundary) positions
        pad_id  = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        not_pad = (input_ids != pad_id)
        not_pad[:, 0]  = False
        not_pad[:, -1] = False

        # Pre-collect all (seq_index, target_position) pairs
        all_seq_indices: List[int] = []
        all_tgt_positions: List[int] = []
        for seq_i in range(N):
            valid_pos = not_pad[seq_i].nonzero(as_tuple=True)[0]
            if len(valid_pos) == 0:
                continue
            n_tgt  = min(n_targets_per_seq, len(valid_pos))
            chosen = valid_pos[torch.randperm(len(valid_pos), device=self.device)[:n_tgt]]
            for tp in chosen:
                all_seq_indices.append(seq_i)
                all_tgt_positions.append(tp.item())

        M = len(all_seq_indices)     # total (seq, target) pairs
        print(f"  Total (seq, target) pairs: {M}")

        feat_accum  = {l: [] for l in self.layers}
        lp_accum    = []
        tgt_id_acc  = []

        for start in range(0, M, batch_size):
            end       = min(start + batch_size, M)
            chunk_seq = all_seq_indices[start:end]
            chunk_pos = all_tgt_positions[start:end]

            ids_b  = input_ids[chunk_seq]                          # [B, L]
            tpos_b = torch.tensor(chunk_pos, device=self.device)   # [B]

            result = self.extract_batch(ids_b, tpos_b, timesteps, n_mask_samples)

            for l in self.layers:
                feat_accum[l].append(result["features"][l].cpu())   # [B, T_eff, d_sae]
            lp_accum.append(result["log_probs"].cpu())              # [B, T_eff]
            tgt_id_acc.append(result["target_ids"].cpu())           # [B]

            if (start // batch_size) % 20 == 0:
                print(f"  [{start+end}/ {2*M}] pairs processed …")

            # Periodic GPU memory cleanup
            if (start // batch_size) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        features  = {l: torch.cat(feat_accum[l], dim=0)  for l in self.layers}
        log_probs = torch.cat(lp_accum,  dim=0)           # [M, T_eff]
        target_ids = torch.cat(tgt_id_acc, dim=0)         # [M]

        return {
            "features":  features,
            "log_probs": log_probs,
            "timesteps": [t for t in timesteps for _ in range(n_mask_samples)],
            "meta": {
                "seq_idx":   torch.tensor(all_seq_indices),
                "pos_idx":   torch.tensor(all_tgt_positions),
                "target_id": target_ids,
            },
        }
