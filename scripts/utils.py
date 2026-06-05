import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class IndexedDataset(Dataset):
    """Wraps a dataset to return (x, y, idx) instead of (x, y)."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


def get_dataset(dataset: str, data_dir: str):
    """Load train/val datasets and return metadata.

    Returns:
        train_ds, val_ds, in_channels, n_classes
    """
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform_train)
        val_ds   = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_val)
        return train_ds, val_ds, 3, 10

    elif dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        val_ds   = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        return train_ds, val_ds, 1, 10

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
