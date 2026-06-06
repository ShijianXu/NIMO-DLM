"""
RQ5: AR-LLM vs dLLM linearity ratio comparison.

Loads NIMO results from both models and plots LR(t) side-by-side.
The key hypothesis:
  - dLLM (LLaDA-8B):  LR decreases monotonically as t decreases
                       (more context → richer interactions → more nonlinear)
  - AR-LLM (GPT-2):   LR is flat across prefix_frac
                       (causal attention breaks bidirectional x_{-j} symmetry)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


TIMESTEPS_DLM = [0.1, 0.25, 0.5, 0.75, 1.0]
TIMESTEPS_AR  = [0.1, 0.25, 0.5, 0.75, 1.0]   # prefix fracs (same grid)
LAYERS_DLM    = [1, 6, 11, 16, 26, 30]
LAYERS_AR     = [0, 2, 4,  6,  8, 10]
DEPTH_DLM     = 32   # LLaDA-8B total layers
DEPTH_AR      = 12   # GPT-2 Small total layers


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nimo-dir",    type=str, default="data/nimo")
    p.add_argument("--nimo-ar-dir", type=str, default="data/nimo_ar")
    p.add_argument("--out-dir",     type=str, default="results")
    return p.parse_args()


def load_lr_table(nimo_dir, layers, ts_list):
    """Return dict (layer, t) → LR."""
    table = {}
    for l in layers:
        for t in ts_list:
            path = os.path.join(nimo_dir, f"layer{l}_t{t:.2f}.pt")
            if os.path.exists(path):
                d = torch.load(path, map_location="cpu")
                table[(l, t)] = d["LR"]
    return table


def plot_comparison(dlm_table, ar_table, out_dir):
    """Side-by-side LR(t) comparison plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Left: dLLM per layer ────────────────────────────────────────
    ax = axes[0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(LAYERS_DLM)))
    for i, l in enumerate(LAYERS_DLM):
        ts  = [t for t in TIMESTEPS_DLM if (l, t) in dlm_table]
        lrs = [dlm_table[(l, t)] for t in ts]
        ax.plot(ts, lrs, "o-", color=colors[i],
                label=f"L{l} ({l}/{DEPTH_DLM})", linewidth=2)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Diffusion timestep t", fontsize=12)
    ax.set_ylabel("Linearity Ratio LR", fontsize=12)
    ax.set_title("dLLM (LLaDA-8B) — expected: LR↓ as t↓", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)

    # ── Middle: AR per layer ─────────────────────────────────────────
    ax = axes[1]
    colors_ar = plt.cm.viridis(np.linspace(0.1, 0.9, len(LAYERS_AR)))
    for i, l in enumerate(LAYERS_AR):
        ts  = [t for t in TIMESTEPS_AR if (l, t) in ar_table]
        lrs = [ar_table[(l, t)] for t in ts]
        ax.plot(ts, lrs, "s--", color=colors_ar[i],
                label=f"L{l} ({l}/{DEPTH_AR})", linewidth=2)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Prefix fraction", fontsize=12)
    ax.set_ylabel("Linearity Ratio LR", fontsize=12)
    ax.set_title("AR-LLM (GPT-2) — expected: LR flat", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # ── Right: mean LR overlay ───────────────────────────────────────
    ax = axes[2]
    # Average over layers for each model
    dlm_mean = {}
    for t in TIMESTEPS_DLM:
        vals = [dlm_table[(l, t)] for l in LAYERS_DLM if (l, t) in dlm_table]
        if vals:
            dlm_mean[t] = np.mean(vals)

    ar_mean = {}
    for t in TIMESTEPS_AR:
        vals = [ar_table[(l, t)] for l in LAYERS_AR if (l, t) in ar_table]
        if vals:
            ar_mean[t] = np.mean(vals)

    if dlm_mean:
        ts_d  = sorted(dlm_mean)
        lrs_d = [dlm_mean[t] for t in ts_d]
        ax.plot(ts_d, lrs_d, "o-", color="crimson", linewidth=2.5,
                markersize=8, label="dLLM (LLaDA-8B) mean")

    if ar_mean:
        ts_a  = sorted(ar_mean)
        lrs_a = [ar_mean[t] for t in ts_a]
        ax.plot(ts_a, lrs_a, "s--", color="steelblue", linewidth=2.5,
                markersize=8, label="AR-LLM (GPT-2) mean")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Timestep / prefix fraction", fontsize=12)
    ax.set_ylabel("Mean LR across layers", fontsize=12)
    ax.set_title("Comparison: Phase Transition vs Flat (RQ5)", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ar_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def print_monotonicity_test(dlm_table, ar_table):
    """Report whether LR(t) is monotone decreasing for each layer."""
    print("\n── dLLM Monotonicity test (LR should increase as t increases) ──")
    for l in LAYERS_DLM:
        ts_sorted = sorted([t for t in TIMESTEPS_DLM if (l, t) in dlm_table])
        lrs = [dlm_table[(l, t)] for t in ts_sorted]
        if len(lrs) < 2:
            continue
        monotone = all(lrs[i] <= lrs[i+1] for i in range(len(lrs)-1))
        trend = "PASS (monotone ↑)" if monotone else "FAIL (not monotone)"
        print(f"  Layer {l:2d}: LR={[f'{r:.3f}' for r in lrs]}  {trend}")

    print("\n── AR-LLM Flatness test (LR should be approximately constant) ──")
    for l in LAYERS_AR:
        ts_sorted = sorted([t for t in TIMESTEPS_AR if (l, t) in ar_table])
        lrs = [ar_table[(l, t)] for t in ts_sorted]
        if len(lrs) < 2:
            continue
        lr_std = np.std(lrs)
        trend = "FLAT (std<0.05)" if lr_std < 0.05 else f"NOT FLAT (std={lr_std:.3f})"
        print(f"  Layer {l:2d}: LR={[f'{r:.3f}' for r in lrs]}  {trend}")


def save_comparison_csv(dlm_table, ar_table, out_dir):
    rows = []
    for l in LAYERS_DLM:
        for t in TIMESTEPS_DLM:
            if (l, t) in dlm_table:
                rows.append({
                    "model": "dLLM", "layer": l, "layer_frac": l/DEPTH_DLM,
                    "t": t, "LR": dlm_table[(l, t)]
                })
    for l in LAYERS_AR:
        for t in TIMESTEPS_AR:
            if (l, t) in ar_table:
                rows.append({
                    "model": "AR", "layer": l, "layer_frac": l/DEPTH_AR,
                    "t": t, "LR": ar_table[(l, t)]
                })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "ar_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    print(df.to_string(index=False))
    return df


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading dLLM NIMO results …")
    dlm_table = load_lr_table(args.nimo_dir, LAYERS_DLM, TIMESTEPS_DLM)
    print(f"  Found {len(dlm_table)} (layer, t) pairs")

    print("Loading AR NIMO results …")
    ar_table = load_lr_table(args.nimo_ar_dir, LAYERS_AR, TIMESTEPS_AR)
    print(f"  Found {len(ar_table)} (layer, t) pairs")

    if not dlm_table and not ar_table:
        print("No results found. Run fit_nimo.py on both datasets first.")
        return

    print("\n── Plotting comparison ──")
    plot_comparison(dlm_table, ar_table, args.out_dir)

    print_monotonicity_test(dlm_table, ar_table)

    print("\n── Saving CSV ──")
    save_comparison_csv(dlm_table, ar_table, args.out_dir)

    print(f"\nDone. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
