import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from convnimo import TeacherModel, PreActResNet18
from utils import get_dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["cifar10", "mnist"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--arch", type=str, default="teacher", choices=["teacher", "preact_resnet"])
    # TeacherModel args
    parser.add_argument("--n-kernels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--hidden-channels", nargs="+", type=int, default=[64, 128, 64])
    parser.add_argument("--mlp-layers", type=int, default=1)
    parser.add_argument("--mlp-hidden-dim", type=int, default=128)
    # PreActResNet18 args
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--p-dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=str, default="ckpts/teacher.pt")
    parser.add_argument("--mode", type=str, default="multiclass",
                        choices=["multiclass", "1vsr"],
                        help="multiclass: cross-entropy over all classes; "
                             "1vsr: binary classifier for one class vs. the rest")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Class index to treat as positive (required for --mode 1vsr)")
    return parser.parse_args()


def get_dataloaders(args):
    train_ds, val_ds, in_channels, n_classes = get_dataset(args.dataset, args.data_dir)

    if args.mode == "1vsr":
        targets = torch.as_tensor(train_ds.targets)
        is_pos   = (targets == args.target_class)
        n_pos, n_neg = is_pos.sum().item(), (~is_pos).sum().item()
        weights  = torch.where(is_pos, 1.0 / n_pos, 1.0 / n_neg)
        sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        print(f"  Resampling: {n_pos} positives / {n_neg} negatives → balanced via WeightedRandomSampler")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, in_channels, n_classes


def compute_loss(logits, y, mode):
    if mode == "multiclass":
        return F.cross_entropy(logits, y)
    else:  # 1vsr: logits (B,1), y is already float binary labels (B,)
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), y)


def train_one_epoch(model, loader, optimizer, device, mode, target_class=None):
    model.train()
    total_loss, correct = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        if mode == "1vsr":
            y_bin = (y == target_class).float()
            loss = compute_loss(logits, y_bin, mode)
            correct += ((logits.squeeze(1) > 0) == y_bin.bool()).sum().item()
        else:
            loss = compute_loss(logits, y, mode)
            correct += (logits.argmax(1) == y).sum().item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
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
            total_loss += compute_loss(logits, y_bin, mode).item() * x.size(0)
            correct += ((logits.squeeze(1) > 0) == y_bin.bool()).sum().item()
        else:
            total_loss += compute_loss(logits, y, mode).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    args = get_args()
    if args.mode == "1vsr" and args.target_class is None:
        raise ValueError("--target-class is required when --mode 1vsr")

    train_loader, val_loader, in_channels, n_classes = get_dataloaders(args)

    n_out = 1 if args.mode == "1vsr" else n_classes
    if args.arch == "preact_resnet":
        model = PreActResNet18(
            out_dim=n_out,
            in_channels=in_channels,
            base_channels=args.base_channels,
            p_dropout=args.p_dropout,
        ).to(args.device)
    else:
        model = TeacherModel(
            in_channels=in_channels,
            n_kernels=args.n_kernels,
            n_out=n_out,
            kernel_size=args.kernel_size,
            hidden_channels=args.hidden_channels,
            mlp_layers=args.mlp_layers,
            mlp_hidden_dim=args.mlp_hidden_dim,
        ).to(args.device)

    print("\n=== Teacher architecture ===")
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    mode_str = f"1vsr (target class={args.target_class})" if args.mode == "1vsr" else "multiclass"
    print(f"Mode: {mode_str}\n")
    print(f"{'Epoch':>5} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, args.device, args.mode, args.target_class)
        val_loss,   val_acc   = evaluate(model, val_loader, args.device, args.mode, args.target_class)
        print(f"{epoch:>5} {train_loss:>11.4f} {train_acc:>10.4f} {val_loss:>10.4f} {val_acc:>9.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

    print(f"\nBest val acc: {best_val_acc:.4f} — saved to {args.save_path}")


if __name__ == "__main__":
    main()
