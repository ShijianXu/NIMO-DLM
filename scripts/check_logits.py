"""Sanity-check a saved logit file. Run: python scripts/check_logits.py"""

import torch
import torch.nn.functional as F

# ---- configure here ----
PATH = "ckpts/mnist_logits.pt"
N    = 5          # number of samples to print
# ------------------------


def main():
    path = PATH
    n    = N

    data   = torch.load(path, map_location="cpu")
    logits = data["targets"]            # [N, K]
    print(f"Shape: {logits.shape}  dtype: {logits.dtype}")
    print(f"Range: min={logits.min():.4f}  max={logits.max():.4f}  "
          f"mean={logits.mean():.4f}  std={logits.std():.4f}")

    probs = F.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)
    print(f"\nFirst {n} samples (idx | pred | logits | max_prob):")
    for i in range(min(n, len(logits))):
        lg = [f"{x:.3f}" for x in logits[i].tolist()]
        print(f"  [{i}]  pred={preds[i].item()}  logits={lg}  max_prob={probs[i].max():.4f}")

    counts = torch.bincount(preds, minlength=logits.size(1))
    print(f"\nPredicted class distribution: {counts.tolist()}")


if __name__ == "__main__":
    main()
