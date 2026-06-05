import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from convnimo import TeacherModel, PreActResNet18
from utils import get_dataset, IndexedDataset


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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--teacher-ckpt", type=str, required=True,
                        help="Path to teacher checkpoint produced by train_teacher.py")
    parser.add_argument("--logits-path", type=str, default="ckpts/teacher_logits.pt",
                        help="Where to save logits and temperature")
    parser.add_argument("--mode", type=str, default="multiclass", choices=["multiclass", "1vsr"],
                        help="Must match the mode used in train_teacher.py")
    parser.add_argument("--target-class", type=int, default=None,
                        help="Required when --mode 1vsr")
    return parser.parse_args()


def get_dataloaders(args):
    train_ds, val_ds, in_channels, n_classes = get_dataset(args.dataset, args.data_dir)

    val_loader     = DataLoader(val_ds,                   batch_size=args.batch_size, shuffle=False, num_workers=4)
    extract_loader = DataLoader(IndexedDataset(train_ds), batch_size=args.batch_size, shuffle=False, num_workers=4)
    return val_loader, extract_loader, in_channels, n_classes


def calibrate_temperature(model, val_loader, device) -> float:
    """Find scalar T that minimises NLL on the held-out validation set."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_logits.append(model(x.to(device)).cpu())
            all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    T = torch.tensor(1.5, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=500)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / T.clamp(min=0.01), labels)
        loss.backward()
        return loss

    opt.step(closure)
    T_val = T.item()
    print(f"  Calibrated temperature T = {T_val:.4f}")
    return T_val


@torch.no_grad()
def extract_logits(model, extract_loader, n_train, n_out, device) -> torch.Tensor:
    """Store raw teacher logits indexed by dataset position."""
    model.eval()
    logits_all = torch.zeros(n_train, n_out)
    for x, _, idx in extract_loader:
        logits_all[idx] = model(x.to(device)).cpu()
    return logits_all


def main():
    args = get_args()
    if args.mode == "1vsr" and args.target_class is None:
        raise ValueError("--target-class is required when --mode 1vsr")

    val_loader, extract_loader, in_channels, n_classes = get_dataloaders(args)

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

    print(f"Loading teacher from {args.teacher_ckpt}")
    model.load_state_dict(torch.load(args.teacher_ckpt, map_location=args.device))

    if args.mode == "multiclass":
        print("\n=== Temperature calibration ===")
        temperature = calibrate_temperature(model, val_loader, args.device)
    else:
        print("\n=== Skipping calibration (1vsr) ===")
        temperature = 1.0

    print("\n=== Logit extraction ===")
    n_train    = len(extract_loader.dataset)
    logits_all = extract_logits(model, extract_loader, n_train, n_out, args.device)

    # Apply calibration: divide by T so targets are already scaled
    targets = logits_all / temperature

    os.makedirs(os.path.dirname(args.logits_path), exist_ok=True)
    torch.save({"targets": targets}, args.logits_path)
    print(f"  Saved to {args.logits_path}")

    print(f"  Logit range: min={targets.min():.4f}  max={targets.max():.4f}")
    print(f"  Logit std:   {targets.std():.4f}")
    if args.mode == "multiclass":
        top2 = targets.topk(2, dim=1).values
        print(f"  Mean top1-top2 gap: {(top2[:, 0] - top2[:, 1]).mean():.4f}")
    else:
        print(f"  Mean logit (class={args.target_class}): {targets.squeeze(1).mean():.4f}")


if __name__ == "__main__":
    main()
