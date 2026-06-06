"""
Stage 1: Feature extraction with masking-induced variation.

Produces per-(layer, timestep) datasets of (SAE features, log-prob targets)
saved as  data/features/<layer>_<timestep>.pt  with keys:
    features   [N, d_sae]  float32  (sparse SAE activations)
    log_probs  [N]         float32  (log P(x_j | x^(t)_{-j}))

and  data/meta.pt  with keys:
    seq_idx    [N]  int64
    pos_idx    [N]  int64
    target_id  [N]  int64
    timesteps  list[float]
    layers     list[int]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import load_dataset

from nimo_dlm import LLaDAExtractor
from nimo_dlm.sae_wrapper import load_all_saes, AVAILABLE_LAYERS


TIMESTEPS = [0.1, 0.25, 0.5, 0.75, 1.0]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir",         type=str,   default="data/features")
    p.add_argument("--n-seqs",          type=int,   default=5_000,
                   help="Number of Wikipedia paragraphs to use")
    p.add_argument("--n-targets",       type=int,   default=10,
                   help="Target positions sampled per sequence")
    p.add_argument("--max-length",      type=int,   default=128)
    p.add_argument("--batch-size",      type=int,   default=4)
    p.add_argument("--layers",          type=int,   nargs="+", default=None,
                   help=f"SAE layers to use (default: all {AVAILABLE_LAYERS})")
    p.add_argument("--trainer",         type=int,   default=0)
    p.add_argument("--device",          type=str,   default="cuda:0")
    p.add_argument("--model-id",        type=str,   default="GSAI-ML/LLaDA-8B-Base")
    return p.parse_args()


def load_texts(n: int) -> list[str]:
    """Load Wikipedia paragraphs (uses cached HF dataset)."""
    print(f"Loading {n} Wikipedia texts …")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    # Filter out empty / very short lines
    texts = [row["text"].strip() for row in ds
             if len(row["text"].strip()) > 80][:n]
    print(f"  Got {len(texts)} usable paragraphs.")
    return texts


def main():
    args = get_args()
    layers = args.layers or AVAILABLE_LAYERS
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load SAEs ────────────────────────────────────────────────
    print("Loading DLM-Scope SAEs …")
    sae_dict = load_all_saes(layers=layers, trainer=args.trainer,
                             masked=True, device="cpu")
    print(f"  Loaded SAEs for layers: {list(sae_dict.keys())}")

    # ── Load texts ───────────────────────────────────────────────
    texts = load_texts(args.n_seqs)

    # ── Run extraction ───────────────────────────────────────────
    extractor = LLaDAExtractor(
        sae_dict  = sae_dict,
        model_id  = args.model_id,
        device    = args.device,
    )

    print("\nStarting feature extraction …")
    result = extractor.extract_dataset(
        texts            = texts,
        timesteps        = TIMESTEPS,
        n_targets_per_seq= args.n_targets,
        max_length       = args.max_length,
        batch_size       = args.batch_size,
    )

    # ── Save per (layer, timestep) in sparse format ──────────────
    # Sparse format: indices [N, K] int16 + values [N, K] float16
    # Saves ~200× less disk space vs dense float32 (K=50 vs d_sae=16384)
    T_eff     = len(TIMESTEPS)
    ts_labels = TIMESTEPS

    for l in layers:
        f_all  = result["features"][l]   # [N_total, T_eff, d_sae]
        lp_all = result["log_probs"]     # [N_total, T_eff]

        for t_idx, t in enumerate(ts_labels):
            f_dense = f_all[:, t_idx, :]           # [N, d_sae]
            lp      = lp_all[:, t_idx]             # [N]

            # Convert to sparse (indices + values for top-k nonzero)
            topk_vals, topk_idx = torch.topk(f_dense.abs(), 50, dim=-1)
            # Keep sign
            topk_vals_signed = f_dense.gather(1, topk_idx)

            out_path = os.path.join(args.out_dir, f"layer{l}_t{t:.2f}.pt")
            torch.save({
                "feat_idx":  topk_idx.to(torch.int16),        # [N, 50]
                "feat_vals": topk_vals_signed.to(torch.float16),  # [N, 50]
                "log_probs": lp.contiguous(),                  # [N]
                "d_sae":     f_dense.shape[1],
                "layer":     l,
                "timestep":  t,
            }, out_path)
            N = f_dense.shape[0]
            size_mb = (topk_idx.numel() * 2 + topk_vals_signed.numel() * 2) / 1e6
            print(f"  Saved {out_path}  N={N}  sparse_size≈{size_mb:.1f}MB")

    # ── Save metadata ─────────────────────────────────────────────
    meta_path = os.path.join(args.out_dir, "meta.pt")
    torch.save({
        **result["meta"],
        "timesteps": TIMESTEPS,
        "layers":    layers,
    }, meta_path)
    print(f"\nMeta saved to {meta_path}")
    print(f"Total samples extracted: {result['log_probs'].shape[0]}")


if __name__ == "__main__":
    main()
