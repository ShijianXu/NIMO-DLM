import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(nn.Module):
    """Simple linear classifier: Y = X @ W^T + b.

    Drop-in replacement for NIMO that learns a single linear layer on top of
    the conv features, trained with MSE against teacher logits.
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(num_features, num_classes)

    @property
    def beta(self) -> torch.Tensor:
        """[K, d] weight matrix — same attribute name as NIMO for compatibility."""
        return self.fc.weight

    @property
    def beta_0(self) -> torch.Tensor:
        """[K] bias — same attribute name as NIMO for compatibility."""
        return self.fc.bias

    def forward(self, X: torch.Tensor, T_logits=None) -> torch.Tensor:
        """T_logits is accepted but unused; kept for API compatibility with NIMO."""
        return self.fc(X)

    def compute_loss(self, Y_hat: torch.Tensor, T_logits: torch.Tensor):
        """MSE loss against teacher logits, same interface as NIMO.compute_loss."""
        loss = F.mse_loss(Y_hat, T_logits, reduction='mean')
        return loss, loss, torch.tensor(0.0), torch.tensor(0.0)

    @torch.no_grad()
    def extract_interpretable_coefficients(self):
        """Return linear coefficients, same interface as NIMO.extract_interpretable_coefficients.

        Returns:
            beta_0 [K]: per-class bias (intercept).
            beta   [K, d]: per-class feature weights.
        """
        return self.fc.bias.clone(), self.fc.weight.clone()
