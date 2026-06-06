"""
Stage 2: Fit SparseSAENIMO for each (layer, timestep) pair.

For each pair:
  1. Load the (features, log_probs) dataset from data/features/
  2. Build feature vocabulary (max_vocab most frequent SAE features)
  3. Train SparseSAENIMO via MSE loss on log-prob targets
  4. Re-solve gamma on full dataset after each epoch
  5. Save model + interpretable beta to  data/nimo/<layer>_<timestep>/

Also saves:
  data/nimo/summary.pt — dict mapping (layer, t) → {LR, beta_l1, R2, ...}
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from nimo_dlm.sparse_nimo import SparseSAENIMO
from nimo_dlm.sae_wrapper import AVAILABLE_LAYERS
from utils_features import load_features_dense


TIMESTEPS = [0.1, 0.25, 0.5, 0.75, 1.0]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feat-dir",    type=str,   default="data/features")
    p.add_argument("--out-dir",     type=str,   default="data/nimo")
    p.add_argument("--layers",      type=int,   nargs="+", default=None)
    p.add_argument("--timesteps",   type=float, nargs="+", default=None)
    p.add_argument("--max-vocab",   type=int,   default=2_048)
    p.add_argument("--hidden-dim",  type=int,   default=64)
    p.add_argument("--lambda-reg",  type=float, default=0.5)
    p.add_argument("--mu-reg",      type=float, default=0.5)
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--val-frac",    type=float, default=0.1)
    p.add_argument("--device",      type=str,   default="cuda:1")
    p.add_argument("--d-sae",       type=int,   default=16_384)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
#  Linear baseline (for comparison)
# ──────────────────────────────────────────────────────────────────

def fit_linear_probe(f_train, y_train, f_val, y_val, lambda_reg=0.1):
    """Fit ridge regression as baseline. Returns val R²."""
    N, d = f_train.shape
    # Use float64 to avoid numerical singularity when features are near-constant
    ft = f_train.double()
    yt = y_train.double()
    A = ft.t() @ ft + lambda_reg * torch.eye(d, device=ft.device, dtype=torch.float64)
    b = ft.t() @ yt
    try:
        w = torch.linalg.solve(A, b).float()
    except Exception:
        # Fallback: lstsq on CPU if GPU solve fails (e.g. rank-deficient at t=1.0)
        w = torch.linalg.lstsq(A.cpu(), b.cpu().unsqueeze(1)).solution.squeeze(1).float().to(f_train.device)
    y_pred = f_val @ w                          # [N_val]
    ss_res = ((y_val - y_pred) ** 2).sum()
    ss_tot = ((y_val - y_val.mean()) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot).item()
    return r2, w


# ──────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────

def train_nimo(model, train_loader, val_loader, optimizer, scheduler,
               epochs, device, full_feat, full_lp):
    """Return list of dicts: epoch metrics."""
    history = []
    full_feat_dev = full_feat.to(device)
    full_lp_dev   = full_lp.to(device)

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        tot_loss = 0.0
        for f_b, y_b in train_loader:
            f_b, y_b = f_b.to(device), y_b.to(device)
            y_b = y_b.unsqueeze(-1)   # [B, 1]
            optimizer.zero_grad()
            total, mse, g_pen, c_pen = model.compute_loss(f_b, y_b)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_loss += total.item() * f_b.shape[0]
        scheduler.step()

        # ── Full-dataset gamma re-solve ────────────────────
        model.eval()
        model.recompute_gamma(
            _iter_full(full_feat_dev, full_lp_dev, batch_size=1024)
        )

        # ── Validation R² ─────────────────────────────────
        with torch.no_grad():
            y_hat_val_list, y_val_list = [], []
            for f_b, y_b in val_loader:
                f_b = f_b.to(device)
                y_hat = model(f_b).squeeze(-1)
                y_hat_val_list.append(y_hat.cpu())
                y_val_list.append(y_b)
            y_hat_val = torch.cat(y_hat_val_list)
            y_val     = torch.cat(y_val_list)
            ss_res = ((y_val - y_hat_val) ** 2).sum()
            ss_tot = ((y_val - y_val.mean()) ** 2).sum()
            r2 = 1.0 - (ss_res / ss_tot).item()

        train_loss = tot_loss / len(train_loader.dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_r2": r2})

        if epoch % 10 == 0:
            beta_0, beta, _ = model.extract_beta()
            n_active = (beta.abs() > 1e-4).sum().item()
            print(f"  Epoch {epoch:>3}  loss={train_loss:.4f}  val_R²={r2:.4f}  "
                  f"active_beta={n_active}")

    return history


def _iter_full(feat, lp, batch_size):
    for i in range(0, len(feat), batch_size):
        yield feat[i:i+batch_size], lp[i:i+batch_size].unsqueeze(-1)


# ──────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────

def main():
    args    = get_args()
    torch.manual_seed(args.seed)
    layers  = args.layers    or AVAILABLE_LAYERS
    ts_list = args.timesteps or TIMESTEPS
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}

    for layer in layers:
        for t in ts_list:
            tag      = f"layer{layer}_t{t:.2f}"
            feat_path = os.path.join(args.feat_dir, f"{tag}.pt")
            if not os.path.exists(feat_path):
                print(f"[SKIP] {feat_path} not found.")
                continue

            print(f"\n{'='*60}")
            print(f"  Fitting NIMO: layer={layer}  t={t}")
            print(f"{'='*60}")

            # ── Load data ─────────────────────────────────────────
            data     = load_features_dense(feat_path)
            feat_all = data["features"]    # [N, d_sae]
            lp_all   = data["log_probs"]   # [N]

            N = feat_all.shape[0]
            n_val = max(1, int(N * args.val_frac))
            n_train = N - n_val
            perm   = torch.randperm(N)
            train_idx, val_idx = perm[:n_train], perm[n_train:]

            f_train, y_train = feat_all[train_idx], lp_all[train_idx]
            f_val,   y_val   = feat_all[val_idx],   lp_all[val_idx]

            train_dl = DataLoader(
                TensorDataset(f_train, y_train),
                batch_size=args.batch_size, shuffle=True, num_workers=0
            )
            val_dl = DataLoader(
                TensorDataset(f_val, y_val),
                batch_size=args.batch_size, shuffle=False, num_workers=0
            )

            # ── Linear baseline ───────────────────────────────────
            # Use only active vocab for speed
            freq   = (f_train != 0).sum(0)
            top_v  = min(args.max_vocab, int((freq > 0).sum().item()))
            _, top_idx = torch.topk(freq, top_v)
            lin_r2, _ = fit_linear_probe(
                f_train[:, top_idx].to(args.device),
                y_train.to(args.device),
                f_val[:, top_idx].to(args.device),
                y_val.to(args.device),
            )
            print(f"  Linear probe R² = {lin_r2:.4f}")

            # ── Build NIMO ────────────────────────────────────────
            model = SparseSAENIMO(
                d_sae      = args.d_sae,
                n_classes  = 1,
                hidden_dim = args.hidden_dim,
                lambda_reg = args.lambda_reg,
                mu_reg     = args.mu_reg,
                max_vocab  = args.max_vocab,
            ).to(args.device)

            # Vocabulary from training data
            def _vocab_iter():
                for i in range(0, len(f_train), 1024):
                    yield f_train[i:i+1024]

            model.build_vocabulary(_vocab_iter(), device=args.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )

            history = train_nimo(
                model, train_dl, val_dl, optimizer, scheduler,
                args.epochs, args.device, f_train, y_train,
            )

            # ── Extract interpretable coefficients ────────────────
            beta_0, beta, beta_full = model.extract_beta()
            G_mat = model.extract_G_matrix(feat_all, batch_size=512)
            # Active mask: only count G where feature is actually non-zero
            f_compact_all = feat_all[:, model.active_indices.cpu()]  # [N, V]
            f_active_mask = (f_compact_all != 0)                     # [N, V] bool
            LR    = model.linearity_ratio(G_mat, f_active_mask)

            n_active = (beta.abs() > 1e-4).sum().item()
            best_r2  = max(h["val_r2"] for h in history)
            print(f"  Final: LR={LR:.4f}  active_β={n_active}  "
                  f"best_val_R²={best_r2:.4f}  linear_R²={lin_r2:.4f}")

            # ── Save ──────────────────────────────────────────────
            out_path = os.path.join(args.out_dir, f"{tag}.pt")
            torch.save({
                "model_state":       model.state_dict(),
                "active_indices":    model.active_indices.cpu(),
                "beta_0":            beta_0.cpu(),
                "beta":              beta.cpu(),
                "beta_full":         beta_full.cpu(),
                "G_mat":             G_mat.cpu(),         # [N, V, 1]
                "history":           history,
                "LR":                LR,
                "linear_r2":         lin_r2,
                "layer":             layer,
                "timestep":          t,
                "n_active_beta":     n_active,
                "vocab_size":        model.V,
                "hyperparams": {
                    "hidden_dim":  args.hidden_dim,
                    "lambda_reg":  args.lambda_reg,
                    "mu_reg":      args.mu_reg,
                    "max_vocab":   args.max_vocab,
                    "epochs":      args.epochs,
                    "lr":          args.lr,
                },
            }, out_path)
            print(f"  Saved {out_path}")

            summary[(layer, t)] = {
                "LR": LR, "linear_r2": lin_r2, "best_r2": best_r2,
                "n_active_beta": n_active,
            }

    # ── Save summary ──────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "summary.pt")
    torch.save(summary, summary_path)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
