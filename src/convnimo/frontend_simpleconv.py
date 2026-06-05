import torch
import torch.nn as nn


class SimpleConv(nn.Module):
    """Stacked conv+ReLU layers followed by global max pooling and L2 normalization.

    Builds n_layers conv+ReLU blocks. Intermediate layers use hidden_channels;
    the final layer outputs n_kernels channels. Global max pooling collapses
    spatial dims to a fixed-length vector, which is then L2-normalized.
    """

    def __init__(
        self,
        in_channels: int,
        n_kernels: int,
        kernel_size: int = 3,
        hidden_channels: list = None,
    ):
        super().__init__()
        # hidden_channels: channel widths for all layers except the last.
        # e.g. [64, 128, 64] with n_kernels=32 gives a 4-layer network: 64 -> 128 -> 64 -> 32
        if hidden_channels is None:
            hidden_channels = []

        channel_sequence = hidden_channels + [n_kernels]   # full list of output channels per layer

        layers = []
        c_in = in_channels
        for c_out in channel_sequence:
            # bias=False: the final MLP has a bias term and can learn to shift features as needed
            layers.append(nn.Conv2d(c_in, c_out, kernel_size, bias=False, stride=1))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out

        self.conv_stack = nn.Sequential(*layers)
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> z: (B, n_kernels)
        z = self.conv_stack(x)  # (B, n_kernels, H', W')
        z = self.pool(z).squeeze(-1).squeeze(-1)
        z = nn.functional.normalize(z, p=2, dim=1)

        return z