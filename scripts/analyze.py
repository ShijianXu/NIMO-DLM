"""
Stage 3: Analysis suite.

Produces:
  results/lr_curves.png         LR(t) per layer (RQ2)
  results/lr_heatmap.png        LR as layer × timestep heatmap
  results/beta_stability.pt     beta^(l,t) trajectories (RQ1)
  results/beta_stability.png    β trajectory plots for top features
  results/interaction_graph.pt  Sparse interaction graph (RQ4)
  results/causal_ranking.pt     Feature rankings by |β|, freq, correlation (RQ3)
  results/summary_table.csv     Numeric summary table
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
import matplotlib.colors as mcolors
import pandas as pd

from nimo_dlm.sae_wrapper import AVAILABLE_LAYERS


TIMESTEPS = [0.1, 0.25, 0.5, 0.75, 1.0]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nimo-dir",   type=str, default="data/nimo")
    p.add_argument("--feat-dir",   type=str, default="data/features")
    p.add_argument("--out-dir",    type=str, default="results")
    p.add_argument("--layers",     type=int, nargs="+", default=None)
    p.add_argument("--timesteps",  type=float, nargs="+", default=None)
    return p.parse_args()


def load_results(nimo_dir, layers, ts_list):
    """Load all saved NIMO results into a nested dict."""
    data = {}
    for l in layers:
        for t in ts_list:
            path = os.path.join(nimo_dir, f"layer{l}_t{t:.2f}.pt")
            if os.path.exists(path):
                data[(l, t)] = torch.load(path, map_location="cpu")
    return data


# ──────────────────────────────────────────────────────────────────
#  RQ2: LR(t) curves and heatmap
# ──────────────────────────────────────────────────────────────────

def plot_lr_curves(results, layers, ts_list, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Line plot: LR vs timestep per layer ─────────────────────
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for i, l in enumerate(layers):
        lrs = [results[(l, t)]["LR"] for t in ts_list if (l, t) in results]
        ts  = [t for t in ts_list if (l, t) in results]
        ax.plot(ts, lrs, "o-", color=colors[i], label=f"Layer {l}", linewidth=2)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="LR=0.5")
    ax.set_xlabel("Diffusion timestep t", fontsize=12)
    ax.set_ylabel("Linearity Ratio LR(t)", fontsize=12)
    ax.set_title("Phase Transition: LR(t) per Layer", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.invert_xaxis()   # t=1 (fully masked) on left, t=0 (unmasked) on right
    ax.grid(alpha=0.3)

    # ── Heatmap: layer × timestep ────────────────────────────────
    ax2 = axes[1]
    mat = np.full((len(layers), len(ts_list)), np.nan)
    for i, l in enumerate(layers):
        for j, t in enumerate(ts_list):
            if (l, t) in results:
                mat[i, j] = results[(l, t)]["LR"]

    im = ax2.imshow(mat, aspect="auto", cmap="RdYlGn",
                    vmin=0, vmax=1, origin="lower")
    ax2.set_xticks(range(len(ts_list)))
    ax2.set_xticklabels([f"{t:.2f}" for t in ts_list])
    ax2.set_yticks(range(len(layers)))
    ax2.set_yticklabels([f"L{l}" for l in layers])
    ax2.set_xlabel("Diffusion timestep t", fontsize=12)
    ax2.set_ylabel("Layer", fontsize=12)
    ax2.set_title("LR(layer, t) Heatmap", fontsize=13)
    plt.colorbar(im, ax=ax2, label="Linearity Ratio")

    # Annotate cells
    for i in range(len(layers)):
        for j in range(len(ts_list)):
            if not np.isnan(mat[i, j]):
                ax2.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                         fontsize=8, color="black")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "lr_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved lr_curves.png")


# ──────────────────────────────────────────────────────────────────
#  RQ1: Beta stability analysis
# ──────────────────────────────────────────────────────────────────

def analyze_beta_stability(results, layers, ts_list, out_dir):
    """
    Track |β_j(t)| for each feature j across timesteps within a layer.
    Cluster into: always-on, early-dominant, late-dominant.
    """
    stability = {}

    for l in layers:
        layer_data = {}
        for t in ts_list:
            if (l, t) not in results:
                continue
            d = results[(l, t)]
            beta      = d["beta"].squeeze(0)   # [V]  (n_classes=1 → squeeze)
            active_idx = d["active_indices"]   # [V]
            layer_data[t] = {"beta": beta, "active_idx": active_idx}

        if not layer_data:
            continue

        # Find union of active feature indices across all timesteps
        all_idx = set()
        for td in layer_data.values():
            all_idx.update(td["active_idx"].tolist())
        all_idx = sorted(all_idx)

        # Build beta matrix [V_union, T]
        idx2col = {v: i for i, v in enumerate(all_idx)}
        T = len([t for t in ts_list if t in layer_data])
        beta_mat = np.zeros((len(all_idx), T))
        t_avail  = [t for t in ts_list if t in layer_data]

        for col, t in enumerate(t_avail):
            td = layer_data[t]
            for feat_local, feat_global in enumerate(td["active_idx"].tolist()):
                if feat_global in idx2col:
                    row = idx2col[feat_global]
                    beta_mat[row, col] = td["beta"][feat_local].item()

        # Cluster by β trajectory shape
        # always-on: std / mean < 0.5 and mean > threshold
        # early-dominant: first-half mean >> second-half mean
        # late-dominant: second-half mean >> first-half mean
        threshold = 1e-3
        mid = T // 2
        categories = {}
        for row_idx, feat_global in enumerate(all_idx):
            betas = np.abs(beta_mat[row_idx])
            mean_b = betas.mean()
            if mean_b < threshold:
                continue
            if mid > 0 and T - mid > 0:
                early_mean = betas[:mid].mean()
                late_mean  = betas[mid:].mean()
                ratio = (early_mean - late_mean) / (mean_b + 1e-8)
                if abs(ratio) < 0.3:
                    cat = "always-on"
                elif ratio > 0.3:
                    cat = "early-dominant"
                else:
                    cat = "late-dominant"
            else:
                cat = "always-on"
            categories[feat_global] = cat

        stability[l] = {
            "beta_mat":   beta_mat,
            "all_idx":    all_idx,
            "t_avail":    t_avail,
            "categories": categories,
        }

        counts = {}
        for c in ["always-on", "early-dominant", "late-dominant"]:
            counts[c] = sum(1 for v in categories.values() if v == c)
        print(f"  Layer {l}: always-on={counts.get('always-on',0)}  "
              f"early-dom={counts.get('early-dominant',0)}  "
              f"late-dom={counts.get('late-dominant',0)}")

    # Save
    torch.save(stability, os.path.join(out_dir, "beta_stability.pt"))
    print(f"  Saved beta_stability.pt")

    # Plot β trajectories for top-10 always-on features per layer
    _plot_beta_trajectories(stability, layers, ts_list, out_dir)
    return stability


def _plot_beta_trajectories(stability, layers, ts_list, out_dir):
    n_cols = min(3, len(layers))
    n_rows = (len(layers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, l in enumerate(layers):
        ax  = axes[idx // n_cols][idx % n_cols]
        if l not in stability:
            ax.set_visible(False)
            continue
        sd    = stability[l]
        bmat  = sd["beta_mat"]                  # [V, T]
        tidxs = sd["t_avail"]
        cats  = sd["categories"]
        all_i = sd["all_idx"]

        # Pick top-10 by max |β| among always-on
        ao_feats = [(fi, gi) for fi, gi in enumerate(all_i)
                    if gi in cats and cats[gi] == "always-on"]
        ao_feats.sort(key=lambda x: -np.abs(bmat[x[0]]).max())
        for fi, gi in ao_feats[:10]:
            ax.plot(tidxs, np.abs(bmat[fi]), "o-", alpha=0.8, linewidth=1.5,
                    label=f"feat {gi}")

        ax.set_title(f"Layer {l} — top always-on β", fontsize=10)
        ax.set_xlabel("timestep t")
        ax.set_ylabel("|β_j(t)|")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        if ao_feats[:5]:
            ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "beta_stability.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved beta_stability.png")


# ──────────────────────────────────────────────────────────────────
#  RQ4: Feature interaction graph
# ──────────────────────────────────────────────────────────────────

def build_interaction_graph(results, layers, ts_list, out_dir, feat_dir="data/features"):
    """
    For each (layer, t), fit sparse linear models:
        G_{u,ij} ≈ Σ_{j'} α_{j'j} f_{j'}
    to identify which features modulate each other.

    Uses the G_mat saved during NIMO fitting.
    """
    graphs = {}

    for l in layers:
        # Use the late-denoising timestep (t small → rich context, large G)
        t_use = min([t for t in ts_list if (l, t) in results])
        if (l, t_use) not in results:
            continue

        d = results[(l, t_use)]
        G_mat = d["G_mat"]      # [N, V, 1]  → [N, V]
        G_mat = G_mat.squeeze(-1)

        # Load the corresponding feature data to get activations
        feat_path_rel = os.path.join(feat_dir, f"layer{l}_t{t_use:.2f}.pt")
        if not os.path.exists(feat_path_rel):
            print(f"  [SKIP] interaction graph for layer {l}: features not found")
            continue

        sys.path.insert(0, os.path.dirname(__file__))
        from utils_features import load_features_dense
        feat_data  = load_features_dense(feat_path_rel)
        feat_full  = feat_data["features"]    # [N, d_sae]
        active_idx = d["active_indices"]
        feat_compact = feat_full[:, active_idx]       # [N, V]
        N, V = feat_compact.shape

        # Fit: for each target feature j, regress G_{ij} on feat_compact
        # Use ridge regression for speed: top alpha_j'j edges
        lam = 0.1
        A   = feat_compact.t() @ feat_compact + lam * torch.eye(V)
        B   = feat_compact.t() @ G_mat          # [V, V]
        try:
            Alpha = torch.linalg.solve(A, B)     # [V, V]  Alpha[j', j]
        except Exception:
            print(f"  [WARN] Interaction solve failed for layer {l}")
            continue

        # Threshold: keep top-k edges by absolute weight
        k_edges = min(200, V * V // 10)
        flat    = Alpha.abs().flatten()
        thresh  = flat.topk(k_edges).values[-1].item()
        sparse  = (Alpha.abs() >= thresh)

        graphs[l] = {
            "alpha_matrix": Alpha,     # [V, V]
            "sparse_mask":  sparse,    # [V, V] bool
            "active_idx":   active_idx,
            "timestep_used": t_use,
        }

        # Count edges
        n_edges = sparse.sum().item()
        print(f"  Layer {l}: {n_edges} significant interactions "
              f"({n_edges}/{V*V} = {100*n_edges/(V*V):.1f}%)")

    torch.save(graphs, os.path.join(out_dir, "interaction_graph.pt"))
    print(f"  Saved interaction_graph.pt")
    return graphs


# ──────────────────────────────────────────────────────────────────
#  RQ3: Feature ranking comparison
# ──────────────────────────────────────────────────────────────────

def build_causal_rankings(results, layers, ts_list, out_dir, feat_dir="data/features"):
    """
    For each (layer, t), produce three feature rankings:
      (a) |β_j| — NIMO global importance
      (b) activation frequency — DLM-Scope default
      (c) absolute Pearson correlation with log-prob target
    Save for downstream causal steering experiments.
    """
    from utils_features import load_features_dense
    rankings = {}

    for l in layers:
        for t in ts_list:
            if (l, t) not in results:
                continue
            d          = results[(l, t)]
            beta       = d["beta"].squeeze(0).abs()  # [V]
            active_idx = d["active_indices"]         # [V]

            feat_path = os.path.join(feat_dir, f"layer{l}_t{t:.2f}.pt")
            if not os.path.exists(feat_path):
                continue
            fd   = load_features_dense(feat_path)
            f_c  = fd["features"][:, active_idx]   # [N, V]
            y    = fd["log_probs"]                  # [N]

            # (b) Activation frequency
            freq = (f_c != 0).float().mean(0)   # [V]

            # (c) Pearson correlation
            f_c_z = f_c - f_c.mean(0, keepdim=True)
            y_z   = y - y.mean()
            denom = f_c_z.std(0) * y_z.std() + 1e-8
            corr  = (f_c_z * y_z.unsqueeze(1)).mean(0) / denom  # [V]

            # Rank indices (descending)
            rank_beta  = beta.argsort(descending=True).tolist()
            rank_freq  = freq.argsort(descending=True).tolist()
            rank_corr  = corr.abs().argsort(descending=True).tolist()

            rankings[(l, t)] = {
                "active_idx":   active_idx,
                "beta_scores":  beta,
                "freq_scores":  freq,
                "corr_scores":  corr.abs(),
                "rank_beta":    rank_beta,
                "rank_freq":    rank_freq,
                "rank_corr":    rank_corr,
            }

    torch.save(rankings, os.path.join(out_dir, "causal_ranking.pt"))
    print(f"  Saved causal_ranking.pt")

    # Top-10 beta features for each (layer, t) — print for inspection
    for (l, t), rk in rankings.items():
        top_global = rk["active_idx"][rk["rank_beta"][:10]].tolist()
        betas      = rk["beta_scores"][rk["rank_beta"][:10]].tolist()
        print(f"  Layer {l}  t={t:.2f}  top-10 |β|: "
              + "  ".join(f"{g}({b:.3f})" for g, b in zip(top_global, betas)))

    return rankings


# ──────────────────────────────────────────────────────────────────
#  Summary CSV
# ──────────────────────────────────────────────────────────────────

def save_summary_csv(results, layers, ts_list, out_dir):
    rows = []
    for l in layers:
        for t in ts_list:
            if (l, t) not in results:
                continue
            d = results[(l, t)]
            rows.append({
                "layer":          l,
                "timestep":       t,
                "LR":             d["LR"],
                "linear_r2":      d["linear_r2"],
                "nimo_best_r2":   max(h["val_r2"] for h in d["history"]),
                "n_active_beta":  d["n_active_beta"],
                "vocab_size":     d["vocab_size"],
            })
    df = pd.DataFrame(rows).sort_values(["layer", "timestep"])
    csv_path = os.path.join(out_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved summary_table.csv")
    print(df.to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args    = get_args()
    layers  = args.layers    or AVAILABLE_LAYERS
    ts_list = args.timesteps or TIMESTEPS
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading NIMO results …")
    results = load_results(args.nimo_dir, layers, ts_list)
    if not results:
        print("No results found. Run fit_nimo.py first.")
        return

    found_layers = sorted(set(l for (l, _) in results))
    found_ts     = sorted(set(t for (_, t) in results))
    print(f"  Found {len(results)} (layer, t) pairs: "
          f"layers={found_layers}  timesteps={found_ts}")

    print("\n── RQ2: LR(t) curves ──")
    plot_lr_curves(results, found_layers, found_ts, args.out_dir)

    print("\n── RQ1: Beta stability ──")
    analyze_beta_stability(results, found_layers, found_ts, args.out_dir)

    print("\n── RQ4: Interaction graph ──")
    build_interaction_graph(results, found_layers, found_ts, args.out_dir,
                            feat_dir=args.feat_dir)

    print("\n── RQ3: Causal rankings ──")
    build_causal_rankings(results, found_layers, found_ts, args.out_dir,
                          feat_dir=args.feat_dir)

    print("\n── Summary table ──")
    save_summary_csv(results, found_layers, found_ts, args.out_dir)

    print(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
