import torch
import torch.nn as nn

from .frontend_simpleconv import SimpleConv
from .backend_mlp import MLP
from .backend_nimo import NIMO
from .backend_linear import LinearClassifier


class TeacherModel(nn.Module):
    """Conv feature extractor + MLP classifier."""

    def __init__(
        self,
        in_channels: int,
        n_kernels: int,
        n_out: int,              # n_classes for multiclass, 1 for 1vsR
        kernel_size: int = 3,
        hidden_channels: list = None,
        mlp_layers: int = 1,
        mlp_hidden_dim: int = 128,
    ):
        super().__init__()
        self.extractor  = SimpleConv(in_channels, n_kernels, kernel_size, hidden_channels)
        self.classifier = MLP(n_kernels, mlp_hidden_dim, n_out, mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extractor(x))


class StudentModel(nn.Module):
    """Conv feature extractor + NIMO classifier.

    The conv extractor can be initialised from a saved TeacherModel checkpoint
    via load_extractor(), sharing the same visual representation.
    """

    def __init__(
        self,
        in_channels: int,
        n_kernels: int,
        n_classes: int,
        kernel_size: int = 3,
        hidden_channels: list = None,
        nimo_hidden_dim: int = 64,
        lambda_reg: float = 1.0,
        mu_reg: float = 1.0,
    ):
        super().__init__()
        self.extractor  = SimpleConv(in_channels, n_kernels, kernel_size, hidden_channels)
        self.classifier = NIMO(
            num_features = n_kernels,
            num_classes  = n_classes,
            hidden_dim   = nimo_hidden_dim,
            lambda_reg   = lambda_reg,
            mu_reg       = mu_reg,
        )

    def forward(self, x: torch.Tensor, t_logits: torch.Tensor = None) -> torch.Tensor:
        """t_logits: calibrated teacher logits (B, K), required during training."""
        z = self.extractor(x)
        return self.classifier(z, t_logits)

    def compute_loss(self, x: torch.Tensor, t_logits: torch.Tensor):
        """Extract features, run NIMO closed-form solve, return (loss, y_hat)."""
        z     = self.extractor(x)
        y_hat = self.classifier(z, t_logits)
        loss, *_ = self.classifier.compute_loss(y_hat, t_logits)
        return loss, y_hat

    @torch.no_grad()
    def recompute_gamma(self, loader, targets_all, device):
        """Re-solve NIMO's gamma over the full training set after each epoch.

        Uses the current extractor weights to produce features, then delegates
        to NIMO.recompute_gamma() which accumulates normal equations batch-by-batch.

        Args:
            loader:      IndexedDataset DataLoader covering every training sample
                         exactly once (no resampling / WeightedRandomSampler).
            targets_all: full teacher-logit tensor [N_train, K] on CPU.
            device:      device string.
        """
        self.eval()

        def _iter():
            for x, _, idx in loader:
                yield self.extractor(x.to(device)), targets_all[idx].to(device)

        self.classifier.recompute_gamma(_iter())

    def load_extractor(self, ckpt_path: str, device: str = "cpu", freeze: bool = False):
        """Load conv extractor weights from a TeacherModel checkpoint.

        Args:
            ckpt_path: path to teacher state_dict saved by train_teacher.py
            freeze:    if True, fix extractor weights during student training
        """
        state = torch.load(ckpt_path, map_location=device)
        extractor_state = {
            k[len("extractor."):]: v
            for k, v in state.items()
            if k.startswith("extractor.")
        }
        self.extractor.load_state_dict(extractor_state)
        if freeze:
            for p in self.extractor.parameters():
                p.requires_grad = False


class LinearStudentModel(nn.Module):
    """Conv feature extractor + linear classifier.

    Same extractor architecture as StudentModel, but replaces NIMO with a
    single linear layer trained via MSE against teacher logits.
    """

    def __init__(
        self,
        in_channels: int,
        n_kernels: int,
        n_classes: int,
        kernel_size: int = 3,
        hidden_channels: list = None,
    ):
        super().__init__()
        self.extractor  = SimpleConv(in_channels, n_kernels, kernel_size, hidden_channels)
        self.classifier = LinearClassifier(n_kernels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extractor(x))

    def compute_loss(self, x: torch.Tensor, t_logits: torch.Tensor):
        y_hat = self.forward(x)
        loss, *_ = self.classifier.compute_loss(y_hat, t_logits)
        return loss, y_hat

    def load_extractor(self, ckpt_path: str, device: str = "cpu", freeze: bool = False):
        """Load conv extractor weights from a TeacherModel checkpoint."""
        state = torch.load(ckpt_path, map_location=device)
        extractor_state = {
            k[len("extractor."):]: v
            for k, v in state.items()
            if k.startswith("extractor.")
        }
        self.extractor.load_state_dict(extractor_state)
        if freeze:
            for p in self.extractor.parameters():
                p.requires_grad = False
