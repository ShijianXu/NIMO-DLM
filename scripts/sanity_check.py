"""
Quick sanity checks after extraction and NIMO fitting.
Run this to verify the pipeline is healthy before the full analysis.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from nimo_dlm.sae_wrapper import AVAILABLE_LAYERS


TIMESTEPS = [0.1, 0.25, 0.5, 0.75, 1.0]


def check_features(feat_dir="data/features"):
    print("── Feature files ──────────────────────────────────────")
    all_ok = True
    for l in AVAILABLE_LAYERS:
        for t in TIMESTEPS:
            path = os.path.join(feat_dir, f"layer{l}_t{t:.2f}.pt")
            if not os.path.exists(path):
                print(f"  [MISSING] {path}")
                all_ok = False
                continue
            d = torch.load(path, map_location="cpu")
            N = d["feat_idx"].shape[0]
            K = d["feat_idx"].shape[1]
            # Check sparsity
            feat_idx = d["feat_idx"].long()
            feat_vals = d["feat_vals"].float()
            dense = torch.zeros(min(10, N), d["d_sae"])
            dense.scatter_(1, feat_idx[:10], feat_vals[:10])
            nnz = (dense != 0).sum(1).float().mean().item()
            lp_range = (d["log_probs"].min().item(), d["log_probs"].max().item())
            print(f"  Layer {l:2d} t={t:.2f}: N={N:>6}  K={K}  "
                  f"mean_nnz={nnz:.1f}  "
                  f"log_prob=[{lp_range[0]:.2f},{lp_range[1]:.2f}]")
    return all_ok


def check_nimo(nimo_dir="data/nimo"):
    print("\n── NIMO results ───────────────────────────────────────")
    rows = []
    for l in AVAILABLE_LAYERS:
        for t in TIMESTEPS:
            path = os.path.join(nimo_dir, f"layer{l}_t{t:.2f}.pt")
            if not os.path.exists(path):
                print(f"  [MISSING] {path}")
                continue
            d = torch.load(path, map_location="cpu")
            best_r2 = max(h["val_r2"] for h in d["history"])
            rows.append({
                "l": l, "t": t, "LR": d["LR"],
                "linear_r2": d["linear_r2"],
                "nimo_r2": best_r2,
                "n_active": d["n_active_beta"],
            })
            print(f"  Layer {l:2d} t={t:.2f}: LR={d['LR']:.3f}  "
                  f"linear_R²={d['linear_r2']:.3f}  nimo_R²={best_r2:.3f}  "
                  f"active_β={d['n_active_beta']}")

    if rows:
        print("\n  Phase transition check (LR should decrease as t decreases):")
        for l in AVAILABLE_LAYERS:
            lr_by_t = {row["t"]: row["LR"] for row in rows if row["l"] == l}
            if not lr_by_t:
                continue
            ts_sorted = sorted(lr_by_t.keys(), reverse=True)  # t=1.0 first
            lrs = [lr_by_t[t] for t in ts_sorted]
            monotone = all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1))
            trend = "✓ monotone decreasing" if monotone else "✗ NOT monotone"
            print(f"  Layer {l:2d}: LR(t=1.0)={lrs[0]:.3f} → LR(t=0.1)={lrs[-1]:.3f}  {trend}")


def main():
    feat_dir  = "data/features"
    nimo_dir  = "data/nimo"
    results_dir = "results"

    ok = check_features(feat_dir)
    if os.path.exists(nimo_dir) and os.listdir(nimo_dir):
        check_nimo(nimo_dir)
    else:
        print("\nNo NIMO results yet — run fit_nimo.py first.")

    if os.path.exists(results_dir):
        figs = [f for f in os.listdir(results_dir) if f.endswith(".png")]
        print(f"\n── Results ───────────────────────────────────────────")
        for f in sorted(figs):
            print(f"  {os.path.join(results_dir, f)}")


if __name__ == "__main__":
    main()
