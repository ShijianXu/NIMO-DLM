"""
TopK Sparse Autoencoder wrapper compatible with DLM-Scope checkpoints.

Checkpoint format (AwesomeInterpretability/llada-mask-topk-sae):
  encoder.weight  [d_sae, d_model]
  encoder.bias    [d_sae]
  decoder.weight  [d_model, d_sae]
  b_dec           [d_model]   pre-encoder / post-decoder bias
  k               scalar int
  threshold       scalar float (unused at inference; we use true TopK)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


DLMSCOPE_MASK_REPO   = "AwesomeInterpretability/llada-mask-topk-sae"
DLMSCOPE_UNMASK_REPO = "AwesomeInterpretability/llada-unmask-topk-sae"
# Layers available in both repos
AVAILABLE_LAYERS = [1, 6, 11, 16, 26, 30]


class TopKSAE(nn.Module):
    """TopK sparse autoencoder (DLM-Scope format).

    Forward pass:
        h_pre  = h - b_dec
        f_pre  = W_enc @ h_pre + b_enc
        f      = TopK(ReLU(f_pre), k=self.k)   # [N, d_sae], exactly k nonzero per row
        h_hat  = W_dec @ f + b_dec
    """

    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae   = d_sae
        self.k       = k

        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=False)
        self.register_buffer("b_dec", torch.zeros(d_model))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_state_dict(cls, state: dict) -> "TopKSAE":
        d_model = state["b_dec"].shape[0]
        d_sae   = state["encoder.weight"].shape[0]
        k       = int(state["k"].item())
        sae = cls(d_model, d_sae, k)
        sae.load_state_dict(state, strict=True)
        return sae

    def load_state_dict(self, state: dict, strict: bool = True):
        # Map checkpoint keys to module attributes
        mapped = {}
        for key, val in state.items():
            if key == "encoder.weight":
                mapped["encoder.weight"] = val
            elif key == "encoder.bias":
                mapped["encoder.bias"] = val
            elif key == "decoder.weight":
                # Checkpoint: [d_model, d_sae]; nn.Linear weight: [d_sae, d_model] transposed
                # We store it directly as the Linear weight (Linear.weight is [out, in])
                # decoder: Linear(d_sae -> d_model), weight shape [d_model, d_sae]
                mapped["decoder.weight"] = val
            elif key == "b_dec":
                mapped["b_dec"] = val
            elif key in ("k", "threshold"):
                pass  # handled separately
        super().load_state_dict(mapped, strict=False)
        if "k" in state:
            self.k = int(state["k"].item())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, h: torch.Tensor) -> torch.Tensor:
        """Return sparse feature activations.

        Args:
            h: [..., d_model] hidden states (bf16/fp32/fp16 ok)
        Returns:
            f: [..., d_sae] TopK-sparse features (same dtype as h)
        """
        h_pre = h - self.b_dec
        f_pre = F.linear(h_pre, self.encoder.weight, self.encoder.bias)
        f_pre = F.relu(f_pre)
        f = self._topk_sparse(f_pre)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Reconstruct hidden state from sparse features."""
        return F.linear(f, self.decoder.weight) + self.b_dec

    def forward(self, h: torch.Tensor):
        """Returns (features, reconstruction)."""
        f = self.encode(h)
        return f, self.decode(f)

    def _topk_sparse(self, f_pre: torch.Tensor) -> torch.Tensor:
        """Keep only the top-k values, zero the rest."""
        k = min(self.k, f_pre.shape[-1])
        topk_vals, topk_idx = torch.topk(f_pre, k, dim=-1)
        out = torch.zeros_like(f_pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    @torch.no_grad()
    def active_indices(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (indices, values) of the K active SAE features per token.

        Returns:
            indices: [..., k]   int64 — indices of top-k active features
            values:  [..., k]   float — their activation values
        """
        h_pre = h - self.b_dec
        f_pre = F.relu(F.linear(h_pre, self.encoder.weight, self.encoder.bias))
        k = min(self.k, f_pre.shape[-1])
        values, indices = torch.topk(f_pre, k, dim=-1)
        return indices, values


# ------------------------------------------------------------------
# Loader helpers
# ------------------------------------------------------------------

def _sae_path_in_repo(layer: int, trainer: int = 0) -> str:
    return f"resid_post_layer_{layer}/trainer_{trainer}/ae.pt"


def load_dlmscope_sae(
    layer: int,
    trainer: int = 0,
    masked: bool = True,
    device: str = "cpu",
) -> TopKSAE:
    """Download (or use cached) a DLM-Scope TopK SAE for a given layer.

    Args:
        layer:   Transformer layer index (one of AVAILABLE_LAYERS).
        trainer: Trainer index (0–5 available; 0 is the default).
        masked:  If True, use SAEs trained on masked inputs (recommended).
        device:  Target device.

    Returns:
        Loaded TopKSAE in eval mode on `device`.
    """
    if layer not in AVAILABLE_LAYERS:
        raise ValueError(f"Layer {layer} not available. Choose from {AVAILABLE_LAYERS}.")

    repo  = DLMSCOPE_MASK_REPO if masked else DLMSCOPE_UNMASK_REPO
    fpath = _sae_path_in_repo(layer, trainer)

    local = hf_hub_download(repo_id=repo, filename=fpath)
    state = torch.load(local, map_location="cpu", weights_only=True)

    sae = TopKSAE.from_state_dict(state)
    sae = sae.to(device=device, dtype=torch.float32).eval()
    return sae


def load_all_saes(
    layers: list[int] | None = None,
    trainer: int = 0,
    masked: bool = True,
    device: str = "cpu",
) -> dict[int, TopKSAE]:
    """Load SAEs for all (or specified) layers into a dict keyed by layer index."""
    layers = layers or AVAILABLE_LAYERS
    return {l: load_dlmscope_sae(l, trainer=trainer, masked=masked, device=device)
            for l in layers}
