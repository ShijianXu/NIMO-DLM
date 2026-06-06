"""
AR-LLM baseline feature extraction for RQ5.

Instead of diffusion timesteps, uses PREFIX LENGTH VARIATION as the
analogue of "context richness":
  - Short prefix (≈ t=1 in dLLM): sparse context, model relies on priors
  - Full prefix  (≈ t=0 in dLLM): rich context, full bidirectional information

For GPT-2-XL with SAEs from Joseph Bloom (jbloom/GPT2-Small-SAEs-Reformatted
or saprmarks/gpt2-xl-sae), we extract:
  f^(l, prefix_len) = SAE(h^(l)(x[:prefix_len]))
  y^(prefix_len) = log P_{GPT2}(x_j | x[:j])   (causal, left-context only)

The prediction is that GPT-2-XL will NOT show a clean phase transition
(LR vs prefix_length should be flat), because causal attention breaks
the bidirectional x_{-j} symmetry.

SAEs used: jbloom/GPT2-Small-SAEs-Reformatted (layers 0,2,4,6,8,10)
  or saprmarks/gpt2-xl-sae if available.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gc
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files, model_info

# Prefix lengths as analogue of timesteps
# Short prefix → sparse context (like t≈1)
# Full prefix  → rich context   (like t≈0)
PREFIX_LENGTHS_FRAC = [0.1, 0.25, 0.5, 0.75, 1.0]  # fraction of sequence length


# ──────────────────────────────────────────────────────────────────
#  GPT-2 SAE wrapper
# ──────────────────────────────────────────────────────────────────

class GPT2SAE(torch.nn.Module):
    """Minimal TopK/ReLU SAE for GPT-2 hidden states."""

    def __init__(self, d_model: int, d_sae: int, k: int = 50):
        super().__init__()
        self.d_model = d_model
        self.d_sae   = d_sae
        self.k       = k
        self.encoder = torch.nn.Linear(d_model, d_sae, bias=True)
        self.decoder = torch.nn.Linear(d_sae, d_model, bias=True)

    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """TopK encode. Returns [N, d_sae] sparse activations."""
        f_pre = F.relu(self.encoder(h))
        # TopK sparsification
        k     = min(self.k, f_pre.shape[-1])
        topv, topi = torch.topk(f_pre, k, dim=-1)
        out   = torch.zeros_like(f_pre)
        out.scatter_(-1, topi, topv)
        return out


GPT2_SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
GPT2_SAE_LAYERS = list(range(12))    # GPT-2 Small: 12 layers


def load_gpt2_sae(layer: int, device: str, k: int = 50) -> GPT2SAE:
    """Load Joseph Bloom's GPT-2 Small SAE from HuggingFace.

    Format: W_enc[d_model, d_sae], W_dec[d_sae, d_model], b_enc[d_sae], b_dec[d_model]
    Activation: ReLU (not TopK) — we apply TopK post-hoc.
    """
    from safetensors.torch import load_file
    filename = f"blocks.{layer}.hook_resid_pre/sae_weights.safetensors"
    local = hf_hub_download(repo_id=GPT2_SAE_REPO, filename=filename)
    state = load_file(local)

    d_model = state["b_dec"].shape[0]   # 768 for GPT-2 Small
    d_sae   = state["b_enc"].shape[0]   # 24576

    sae = GPT2SAE(d_model, d_sae, k=k)
    # Map keys: W_enc → encoder.weight (transposed to [d_sae, d_model])
    sae.encoder.weight.data.copy_(state["W_enc"].t())
    sae.encoder.bias.data.copy_(state["b_enc"])
    sae.decoder.weight.data.copy_(state["W_dec"].t())  # [d_model, d_sae] → [d_sae, d_model] for Linear
    sae.decoder.bias.data.copy_(state["b_dec"])
    return sae.to(device=device, dtype=torch.float32).eval()


# ──────────────────────────────────────────────────────────────────
#  AR extractor
# ──────────────────────────────────────────────────────────────────

class GPT2Extractor:
    """Extract (SAE features, log-prob) pairs from GPT-2 with varying prefix lengths."""

    def __init__(self, sae_dict: Dict[int, GPT2SAE],
                 model_id: str = "gpt2-large",
                 device: str = "cuda:0"):
        self.sae_dict = sae_dict
        self.device   = device
        self.layers   = sorted(sae_dict.keys())

        print(f"Loading {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(
            model_id, device_map=device
        ).eval()
        for l, sae in self.sae_dict.items():
            self.sae_dict[l] = sae.to(device=device, dtype=torch.float32)

        # GPT-2's transformer blocks
        self._blocks = self.model.transformer.h

    @torch.no_grad()
    def extract_batch(
        self,
        input_ids:    torch.Tensor,   # [B, L]
        target_pos:   torch.Tensor,   # [B] — position j to predict
        prefix_fracs: List[float],    # analogue of timesteps
    ) -> Dict:
        B, L = input_ids.shape
        target_ids = input_ids[torch.arange(B), target_pos]

        all_feats   = {l: [] for l in self.layers}
        all_logprob = []

        for frac in prefix_fracs:
            # Build prefix: keep tokens 0..prefix_len, pad/replace the rest
            # For causal LM: we can only use tokens UP TO target_pos
            # "prefix_len" = fraction of tokens before target_pos

            # For GPT-2 comparison: use varying context window ending at target_pos
            # Short context (frac≈0.1): only the last few tokens before j
            # Full context  (frac≈1.0): all tokens from 0..j-1

            prefix_ids = input_ids.clone()

            for b in range(B):
                tpos   = target_pos[b].item()
                n_ctx  = max(1, int(tpos * frac))  # context tokens from position 0..tpos-1
                ctx_start = tpos - n_ctx
                # Zero out tokens before ctx_start (replace with pad)
                if ctx_start > 0:
                    prefix_ids[b, :ctx_start] = self.tokenizer.eos_token_id
                # Zero out tokens at and after target_pos
                prefix_ids[b, tpos:] = self.tokenizer.eos_token_id

            # Hook hidden states (resid_pre = input to each block)
            hidden_states = {}
            hooks = []
            for l_idx in self.layers:
                block = self._blocks[l_idx]
                def make_pre_hook(li):
                    def hook_fn(m, inp):
                        # inp is a tuple; inp[0] is the residual stream [B, L, d_model]
                        h = inp[0] if isinstance(inp, (tuple, list)) else inp
                        hidden_states[li] = h.detach()
                    return hook_fn
                hooks.append(block.register_forward_pre_hook(make_pre_hook(l_idx)))

            outputs = self.model(input_ids=prefix_ids)
            for h in hooks:
                h.remove()

            # Log-prob at target position
            logits = outputs.logits[torch.arange(B), target_pos - 1, :]
            log_probs_all = F.log_softmax(logits.float(), dim=-1)
            lp = log_probs_all[torch.arange(B), target_ids]
            all_logprob.append(lp.cpu())

            for l_idx in self.layers:
                h_t = hidden_states[l_idx][torch.arange(B), target_pos - 1, :].float()
                f_l = self.sae_dict[l_idx].encode(h_t)
                all_feats[l_idx].append(f_l.cpu())

        features  = {l: torch.stack(all_feats[l], dim=1) for l in self.layers}
        log_probs = torch.stack(all_logprob, dim=1)

        return {
            "features":    features,
            "log_probs":   log_probs,
            "prefix_fracs": prefix_fracs,
            "target_ids":  target_ids.cpu(),
        }


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",      type=str, default="data/features_ar")
    p.add_argument("--n-seqs",       type=int, default=2_000)
    p.add_argument("--n-targets",    type=int, default=10)
    p.add_argument("--max-length",   type=int, default=128)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--device",       type=str, default="cuda:0")
    p.add_argument("--model-id",     type=str, default="gpt2")
    p.add_argument("--layers",       type=int, nargs="+", default=[0, 2, 4, 6, 8, 10])
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load GPT-2 Small SAEs from Joseph Bloom's repo
    sae_dict = {}
    print("Loading GPT-2 Small SAEs (jbloom/GPT2-Small-SAEs-Reformatted) …")
    for l in args.layers:
        sae_dict[l] = load_gpt2_sae(l, device="cpu", k=50)
        print(f"  Loaded SAE for layer {l}: d_model={sae_dict[l].d_model}  d_sae={sae_dict[l].d_sae}")

    extractor = GPT2Extractor(
        sae_dict=sae_dict, model_id=args.model_id, device=args.device
    )

    # Load texts
    print(f"Loading {args.n_seqs} Wikipedia texts …")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [row["text"].strip() for row in ds if len(row["text"].strip()) > 80][:args.n_seqs]

    # Tokenise
    enc = extractor.tokenizer(
        texts, return_tensors="pt", max_length=args.max_length,
        truncation=True, padding="max_length"
    )
    input_ids = enc.input_ids.to(args.device)
    N, L = input_ids.shape

    # Collect (seq, target) pairs
    all_seq, all_tpos = [], []
    pad_id = extractor.tokenizer.eos_token_id
    for i in range(N):
        valid = ((input_ids[i] != pad_id) &
                 (torch.arange(L, device=args.device) > 0) &
                 (torch.arange(L, device=args.device) < L - 1)).nonzero(as_tuple=True)[0]
        # Need at least some left context
        valid = valid[valid > 5]
        if len(valid) == 0:
            continue
        n_t   = min(args.n_targets, len(valid))
        chosen = valid[torch.randperm(len(valid))[:n_t]]
        for tp in chosen:
            all_seq.append(i)
            all_tpos.append(tp.item())

    M = len(all_seq)
    print(f"Total (seq, target) pairs: {M}")

    feat_accum  = {l: [] for l in args.layers}
    lp_accum    = []
    tgt_id_acc  = []

    for start in range(0, M, args.batch_size):
        end  = min(start + args.batch_size, M)
        ids_b = input_ids[all_seq[start:end]]
        tpos_b = torch.tensor(all_tpos[start:end], device=args.device)

        result = extractor.extract_batch(ids_b, tpos_b, PREFIX_LENGTHS_FRAC)

        for l in args.layers:
            feat_accum[l].append(result["features"][l].cpu())
        lp_accum.append(result["log_probs"].cpu())
        tgt_id_acc.append(result["target_ids"].cpu())

        if (start // args.batch_size) % 20 == 0:
            print(f"  {end}/{M} pairs processed …")

    # Save
    features  = {l: torch.cat(feat_accum[l], dim=0) for l in args.layers}
    log_probs = torch.cat(lp_accum, dim=0)

    for l in args.layers:
        f_all  = features[l]
        lp_all = log_probs
        for t_idx, frac in enumerate(PREFIX_LENGTHS_FRAC):
            f_dense = f_all[:, t_idx, :]
            topk_vals, topk_idx = torch.topk(f_dense.abs(), 50, dim=-1)
            topk_vals_signed = f_dense.gather(1, topk_idx)

            out_path = os.path.join(args.out_dir, f"layer{l}_t{frac:.2f}.pt")
            torch.save({
                "feat_idx":   topk_idx.to(torch.int16),
                "feat_vals":  topk_vals_signed.to(torch.float16),
                "log_probs":  lp_all[:, t_idx].contiguous(),
                "d_sae":      f_dense.shape[1],
                "layer":      l,
                "timestep":   frac,
                "model":      "ar",
            }, out_path)
            print(f"  Saved {out_path}")

    torch.save({
        "seq_idx":   torch.tensor(all_seq),
        "pos_idx":   torch.tensor(all_tpos),
        "target_id": torch.cat(tgt_id_acc),
        "prefix_fracs": PREFIX_LENGTHS_FRAC,
        "layers":    args.layers,
    }, os.path.join(args.out_dir, "meta.pt"))
    print("Done.")


if __name__ == "__main__":
    main()
