import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class PreActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.downsample(x) if self.downsample is not None else x
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet18(nn.Module):
    def __init__(
        self,
        out_dim: int = 10,
        in_channels: int = 3,
        base_channels: int = 64,
        p_dropout: float = 0.0,
    ):
        super().__init__()
        self.out_dim = out_dim
        c = base_channels
        self.stem = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(c, c, stride=1)
        self.layer2 = self._make_layer(c, 2 * c, stride=2)
        self.layer3 = self._make_layer(2 * c, 4 * c, stride=2)
        self.layer4 = self._make_layer(4 * c, 8 * c, stride=2)
        self.bn = nn.BatchNorm2d(8 * c)
        self.dropout = nn.Dropout(p=p_dropout)
        self.head = nn.Linear(8 * c, out_dim)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            PreActBlock(in_channels, out_channels, stride=stride),
            PreActBlock(out_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(F.relu(self.bn(x)), 1)
        x = self.dropout(x.flatten(1))
        x = self.head(x)
        if self.out_dim == 1:
            return rearrange(x, "b 1 -> b")
        return x
    

if __name__ == "__main__":
    import torch
    # sanity check: test forward pass and output shape
    model = PreActResNet18(out_dim=10, in_channels=3, base_channels=64)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    print(logits.shape)  # should be (4, 10)