import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from convnimo import StudentModel
from convnimo.model import LinearStudentModel
from utils import get_dataset, IndexedDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["cifar10", "mnist"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--n-kernels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--hidden-channels", nargs="+", type=int, default=[64, 128, 64])
    parser.add_argument("--nimo-hidden-dim", type=int, default=64,
                        help="Hidden dim of the shared MLP inside NIMO")
    parser.add_argument("--lambda-reg", type=float, default=1.0,
                        help="Ridge regularisation strength on gamma")
    parser.add_argument("--mu-reg", type=float, default=1.0,
                        help="Sparsity penalty strength on C")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logits-path", type=str, required=True,
                        help="Path to logits file produced by calibrate.py")
    parser.add_argument("--save-path", type=str, default="ckpts/student.pt")
    parser.add_argument("--mode", type=str, default="multiclass",
                        choices=["multiclass", "1vsr"],
                        help="multiclass: regress all-class logits; "
                             "1vsr: regress a single binary logit for one class vs. the rest")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Class index to treat as positive (required for --mode 1vsr)")
    parser.add_argument("--model", type=str, default="nimo", choices=["nimo", "linear"],
                        help="Backend classifier: nimo (default) or linear")
    return parser.parse_args()


def get_dataloaders(args):
    train_ds, val_ds, in_channels, n_classes = get_dataset(args.dataset, args.data_dir)

    if args.mode == "1vsr":
        targets = torch.as_tensor(train_ds.targets)
        is_pos  = (targets == args.target_class)
        n_pos, n_neg = is_pos.sum().item(), (~is_pos).sum().item()
        weights = torch.where(is_pos, 1.0 / n_pos, 1.0 / n_neg)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"  Resampling: {n_pos} positives / {n_neg} negatives → balanced via WeightedRandomSampler")
        student_loader = DataLoader(
            IndexedDataset(train_ds), batch_size=args.batch_size, sampler=sampler, num_workers=4
        )
    else:
        student_loader = DataLoader(
            IndexedDataset(train_ds), batch_size=args.batch_size, shuffle=True, num_workers=4
        )

    # Dedicated loader for gamma re-solve: covers every training sample exactly
    # once with no resampling, regardless of mode.
    resolve_loader = DataLoader(
        IndexedDataset(train_ds), batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return val_loader, student_loader, resolve_loader, in_channels, n_classes


def train_one_epoch(model, loader, targets_all, optimizer, device, mode, target_class=None):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y, idx in loader:
        x, y = x.to(device), y.to(device)
        t_logits = targets_all[idx].to(device)
        optimizer.zero_grad()
        loss, y_hat = model.compute_loss(x, t_logits)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        if mode == "1vsr":
            correct += ((y_hat.squeeze(1) > 0) == (y == target_class)).sum().item()
        else:
            correct += (y_hat.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device, mode, target_class=None):
    model.eval()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if mode == "1vsr":
            y_bin = (y == target_class).float()
            total_loss += F.binary_cross_entropy_with_logits(logits.squeeze(1), y_bin).item() * x.size(0)
            correct += ((logits.squeeze(1) > 0) == y_bin.bool()).sum().item()
        else:
            total_loss += F.cross_entropy(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    args = get_args()
    if args.mode == "1vsr" and args.target_class is None:
        raise ValueError("--target-class is required when --mode 1vsr")

    val_loader, student_loader, resolve_loader, in_channels, n_classes = get_dataloaders(args)

    logits      = torch.load(args.logits_path, map_location="cpu")
    targets_all = logits["targets"]
    print(f"Loaded calibrated targets from {args.logits_path}  shape={tuple(targets_all.shape)}")

    n_out = 1 if args.mode == "1vsr" else n_classes
    if args.model == "linear":
        model = LinearStudentModel(
            in_channels=in_channels,
            n_kernels=args.n_kernels,
            n_classes=n_out,
            kernel_size=args.kernel_size,
            hidden_channels=args.hidden_channels,
        ).to(args.device)
    else:
        model = StudentModel(
            in_channels=in_channels,
            n_kernels=args.n_kernels,
            n_classes=n_out,
            kernel_size=args.kernel_size,
            hidden_channels=args.hidden_channels,
            nimo_hidden_dim=args.nimo_hidden_dim,
            lambda_reg=args.lambda_reg,
            mu_reg=args.mu_reg,
        ).to(args.device)

    mode_str = f"1vsr (target class={args.target_class})" if args.mode == "1vsr" else "multiclass"
    print(f"\nMode: {mode_str}")
    print(f"\n=== Student architecture ({args.model}) ===")
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, student_loader, targets_all, optimizer, args.device, args.mode, args.target_class
        )
        # Re-solve gamma on the full training set (NIMO only)
        if args.model == "nimo":
            model.recompute_gamma(resolve_loader, targets_all, args.device)
        val_loss, val_acc = evaluate(model, val_loader, args.device, args.mode, args.target_class)
        print(f"{epoch:>5} {train_loss:>11.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>9.4f}")
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

    print(f"\nBest val acc: {best_val_acc:.4f} — saved to {args.save_path}")


if __name__ == "__main__":
    main()
