"""Utilities for loading sparse feature files produced by extract_features.py."""

import torch


def load_features_dense(path: str) -> dict:
    """Load a sparse feature file and reconstruct dense [N, d_sae] tensor.

    Returns dict with:
        features   [N, d_sae]  float32
        log_probs  [N]         float32
        layer      int
        timestep   float
    """
    data = torch.load(path, map_location="cpu")

    if "features" in data:
        # Already-dense format (legacy)
        return {
            "features":  data["features"].float(),
            "log_probs": data["log_probs"].float(),
            "layer":     data["layer"],
            "timestep":  data["timestep"],
        }

    # Sparse format: reconstruct from (indices, values)
    idx  = data["feat_idx"].long()           # [N, K]
    vals = data["feat_vals"].float()         # [N, K]
    N, K = idx.shape
    d    = int(data["d_sae"])

    f_dense = torch.zeros(N, d)
    f_dense.scatter_(1, idx, vals)

    return {
        "features":  f_dense,
        "log_probs": data["log_probs"].float(),
        "layer":     data["layer"],
        "timestep":  data["timestep"],
    }


def iter_sparse_features(path: str, batch_size: int = 1024):
    """Yield (f_dense [B, d_sae], log_prob [B]) batches from a sparse feature file."""
    data = load_features_dense(path)
    f    = data["features"]
    lp   = data["log_probs"]
    N    = f.shape[0]
    for i in range(0, N, batch_size):
        yield f[i:i+batch_size], lp[i:i+batch_size]
