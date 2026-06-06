"""
SparseSAENIMO: NIMO adapted for high-dimensional sparse SAE features.

Design rationale
----------------
The original NIMO builds an [N, d, d] tensor in forward_correction, which is
infeasible at d_SAE = 16 384.  Here we use a compressed context encoding:

    context_sum_i  =  Σ_{j' active}  f_{j',i} * binary_map[j']  ∈ ℝ^{n_bits}
    c_{-j,i}       =  context_sum_i  −  f_{j,i} * binary_map[j] ∈ ℝ^{n_bits}

So the correction network g_u takes input [c_{-j} | binary_j] ∈ ℝ^{2·n_bits}
(≈ 30 dims for d_SAE = 16 384), instead of x_{-j} ∈ ℝ^{d_SAE}.

The key NIMO properties are preserved:
  • g_u(0, E_j) = 0 by explicit centering → linear baseline when context is absent
  • β_j is the global linear coefficient (MEM_j = β_j)
  • Closed-form γ solve via parameter elimination

Feature vocabulary
------------------
We first scan the dataset and keep the V_active (≤ max_vocab) most frequently
activated features, forming a compact index mapping.  NIMO's Lasso (via the
C regulariser) will further zero out most β within that vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterator


# ─────────────────────────────────────────────────────────────────
#  Correction network
# ─────────────────────────────────────────────────────────────────

class ContextCorrectionMLP(nn.Module):
    """Shared g_u network for sparse context.

    Input:  [c_{-j} | binary_j]  ∈ ℝ^{2·n_bits}
    Output: scalar correction (before centering/bounding) per (sample, feature, class)
    """

    def __init__(self, n_bits: int, hidden_dim: int, n_classes: int,
                 noise_std: float = 0.1, dropout: float = 0.0):
        super().__init__()
        in_dim = 2 * n_bits
        self.noise_std = noise_std

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + n_bits, hidden_dim)     # re-inject binary_j
        self.fc3 = nn.Linear(2 * hidden_dim + n_bits, n_classes)
        self.drop = nn.Dropout(p=dropout)
        self._n_bits = n_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., 2·n_bits]
        binary_j = x[..., self._n_bits:]               # [..., n_bits]

        z1 = self.fc1(x)
        if self.training and self.noise_std > 0:
            z1 = z1 + self.noise_std * torch.randn_like(z1)
        z1 = torch.tanh(0.3 * z1)

        z1_cat = torch.cat([z1, binary_j], dim=-1)     # [..., hidden + n_bits]
        z2 = torch.sin(2 * torch.pi * self.fc2(z1_cat))
        z2 = self.drop(z2)

        z  = torch.cat([z2, z1_cat], dim=-1)           # [..., 2·hidden + n_bits]
        return self.fc3(z)                              # [..., n_classes]


# ─────────────────────────────────────────────────────────────────
#  Sparse NIMO
# ─────────────────────────────────────────────────────────────────

class SparseSAENIMO(nn.Module):
    """NIMO for sparse SAE features at dLLM scale.

    Usage
    -----
    1. Instantiate with d_sae and desired max_vocab.
    2. Call ``build_vocabulary(feature_iter)`` once to register the active
       feature vocabulary from the dataset.
    3. Train with ``forward(f, y_target)`` (training mode) — closed-form γ
       solve happens inside.
    4. After each epoch call ``recompute_gamma(feature_target_iter)`` for a
       full-dataset γ update.
    5. ``extract_beta()`` returns the interpretable coefficients.
    """

    def __init__(
        self,
        d_sae:       int   = 16_384,
        n_classes:   int   = 1,
        hidden_dim:  int   = 64,
        lambda_reg:  float = 1.0,
        mu_reg:      float = 1.0,
        max_vocab:   int   = 2_048,
        noise_std:   float = 0.1,
    ):
        super().__init__()
        self.d_sae      = d_sae
        self.K          = n_classes
        self.lambda_reg = lambda_reg
        self.mu_reg     = mu_reg
        self.max_vocab  = max_vocab

        # Binary positional encodings for features 0..d_sae-1
        n_bits = int(torch.floor(torch.log2(torch.tensor(float(d_sae)))).item()) + 1
        self.n_bits = n_bits
        indices = torch.arange(d_sae, dtype=torch.int64).unsqueeze(1)
        powers  = 2 ** torch.arange(n_bits - 1, -1, -1, dtype=torch.int64)
        binary_map = (indices.bitwise_and(powers) > 0).float() - 0.5  # [d_sae, n_bits]
        self.register_buffer("binary_map_full", binary_map)  # [d_sae, n_bits]

        # Will be set by build_vocabulary()
        self.register_buffer("active_indices", torch.zeros(0, dtype=torch.long))
        self._vocab_built = False

        # Correction network: input is [c_{-j} | binary_j] ∈ ℝ^{2·n_bits}
        self.shared_mlp = ContextCorrectionMLP(n_bits, hidden_dim, n_classes, noise_std)

        # Learnable output scale
        self.alpha2 = nn.Parameter(torch.tensor(4.0))

        # These will be resized after vocabulary is built
        # C: [K, V]  (V set after build_vocabulary)
        self.C = nn.Parameter(torch.empty(0))

        # Ridge penalty matrix (built after vocabulary)
        self.register_buffer("W", torch.zeros(0, 0))

        # Cached gamma and interpretable beta (set after solve)
        self.register_buffer("_gamma",  torch.zeros(0))
        self.register_buffer("beta_0",  torch.zeros(n_classes))
        self.register_buffer("beta",    torch.zeros(n_classes, 0))

    # ──────────────────────────────────────────────
    #  Vocabulary construction
    # ──────────────────────────────────────────────

    def build_vocabulary(
        self,
        feature_iter: Iterator[torch.Tensor],
        device: str = "cpu",
    ) -> None:
        """Scan sparse feature tensors and pick the max_vocab most active features.

        Args:
            feature_iter: yields [N_i, d_sae] sparse feature tensors (float, most zeros)
            device: target device for the vocabulary buffers
        """
        freq = torch.zeros(self.d_sae, dtype=torch.long)
        for f_batch in feature_iter:
            nonzero_mask = (f_batch != 0)          # [N, d_sae]
            freq += nonzero_mask.sum(dim=0).cpu()

        V = min(self.max_vocab, int((freq > 0).sum().item()))
        _, top_idx = torch.topk(freq, V)
        top_idx, _ = top_idx.sort()                # keep natural order

        self.register_buffer("active_indices", top_idx.to(device))
        self._vocab_built = True
        V = top_idx.shape[0]

        # Build the binary map for the active vocabulary
        self.register_buffer(
            "binary_map",
            self.binary_map_full[top_idx].to(device)  # [V, n_bits]
        )

        # Re-initialise learnable C and ridge matrix for the new vocabulary size
        self.C = nn.Parameter(torch.empty(self.K, V, device=device).normal_(0.5, 0.1))

        d1 = V + 1
        W  = torch.eye(d1, device=device)
        W[0, 0] = 0.0
        self.register_buffer("W", W)

        self.register_buffer("_gamma", torch.zeros(self.K, d1, 1, device=device))
        self.register_buffer("beta_0", torch.zeros(self.K, device=device))
        self.register_buffer("beta",   torch.zeros(self.K, V, device=device))

        print(f"[SparseSAENIMO] Vocabulary built: {V} active features "
              f"(max_vocab={self.max_vocab}, d_sae={self.d_sae})")

    # ──────────────────────────────────────────────
    #  Core computation
    # ──────────────────────────────────────────────

    @property
    def V(self) -> int:
        return self.active_indices.shape[0]

    def _remap(self, f: torch.Tensor) -> torch.Tensor:
        """Project [N, d_sae] → [N, V] by selecting active vocabulary columns."""
        return f[:, self.active_indices]

    def _forward_correction(self, f_compact: torch.Tensor) -> torch.Tensor:
        """Compute nonlinear corrections G for all vocabulary features.

        Args:
            f_compact: [N, V]
        Returns:
            G: [N, V, K]  (bounded, G(0)=0 enforced)
        """
        N, V = f_compact.shape
        bmap = self.binary_map                              # [V, n_bits]

        # Context sum per sample: c_i = Σ_j f_{ij} * binary_j  [N, n_bits]
        context_sum = f_compact @ bmap                     # [N, n_bits]

        # For each vocab feature j: c_{-j,i} = context_sum_i - f_{ij} * binary_j
        # f_compact: [N, V] → [N, V, 1]
        # bmap:      [V, n_bits] → [1, V, n_bits]
        c_minus_j = (
            context_sum.unsqueeze(1)                       # [N, 1, n_bits]
            - f_compact.unsqueeze(2) * bmap.unsqueeze(0)   # [N, V, n_bits]
        )                                                  # [N, V, n_bits]

        # MLP input: [c_{-j} | binary_j]  [N, V, 2·n_bits]
        bmap_exp  = bmap.unsqueeze(0).expand(N, -1, -1)   # [N, V, n_bits]
        mlp_input = torch.cat([c_minus_j, bmap_exp], dim=-1)  # [N, V, 2·n_bits]

        G_raw = self.shared_mlp(mlp_input)                 # [N, V, K]

        # Enforce G(0, E_j) = 0 by centering
        zero_ctx   = torch.zeros(V, self.n_bits, device=f_compact.device,
                                 dtype=f_compact.dtype)
        zero_input = torch.cat([zero_ctx, bmap], dim=-1)  # [V, 2·n_bits]
        G_zero     = self.shared_mlp(zero_input)           # [V, K]
        G_centered = G_raw - G_zero.unsqueeze(0)           # [N, V, K]

        # Bound to (-1, 1) neighbourhood
        bound  = 0.5 * (1.0 + torch.tanh(self.alpha2))
        return torch.tanh(G_centered) * bound              # [N, V, K]

    def _build_X_tilde_aug(self, f_compact: torch.Tensor) -> torch.Tensor:
        """Build augmented feature matrix for the closed-form γ solve.

        Returns:
            X_tilde_aug: [K, N, V+1]  (intercept column prepended)
        """
        N  = f_compact.shape[0]
        G  = self._forward_correction(f_compact)           # [N, V, K]
        C_pos = F.softplus(self.C)                         # [K, V]

        # B[n,v,k] = f_{nv} * (1 + G[n,v,k])
        B   = f_compact.unsqueeze(-1) * (1.0 + G)         # [N, V, K]
        B_k = B.permute(2, 0, 1)                           # [K, N, V]

        X_tilde = B_k * C_pos.unsqueeze(1)                 # [K, N, V]
        ones    = torch.ones(self.K, N, 1,
                             device=f_compact.device, dtype=f_compact.dtype)
        return torch.cat([ones, X_tilde], dim=2)           # [K, N, V+1]

    # ──────────────────────────────────────────────
    #  Forward / loss / gamma solve
    # ──────────────────────────────────────────────

    def forward(self, f: torch.Tensor, y_target: torch.Tensor | None = None) -> torch.Tensor:
        """Predict logit targets.

        Args:
            f:        [N, d_sae] sparse SAE features
            y_target: [N, K] targets (required during training for γ solve)
        Returns:
            y_hat: [N, K] predictions
        """
        assert self._vocab_built, "Call build_vocabulary() before forward()."
        f_compact   = self._remap(f)
        X_tilde_aug = self._build_X_tilde_aug(f_compact)  # [K, N, V+1]

        if self.training and y_target is not None:
            A     = torch.bmm(X_tilde_aug.transpose(1, 2), X_tilde_aug)   # [K, V+1, V+1]
            A_reg = A + self.lambda_reg * self.W.unsqueeze(0)
            T_k   = y_target.t().unsqueeze(-1)             # [K, N, 1]
            b     = torch.bmm(X_tilde_aug.transpose(1, 2), T_k)  # [K, V+1, 1]
            gamma = torch.linalg.solve(A_reg, b)           # [K, V+1, 1]
            self._gamma = gamma
        else:
            gamma = self._gamma

        Y_hat = torch.bmm(X_tilde_aug, gamma).squeeze(-1).t()  # [N, K]
        return Y_hat

    def compute_loss(
        self, f: torch.Tensor, y_target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward + loss computation."""
        y_hat = self.forward(f, y_target)
        N     = y_hat.shape[0]

        mse          = F.mse_loss(y_hat, y_target, reduction="mean")
        gamma_pen    = self.lambda_reg * self._gamma[:, 1:, :].pow(2).sum() / (N * self.K)
        C_pos        = F.softplus(self.C)
        c_pen        = self.mu_reg * C_pos.pow(2).sum() / (self.K * self.V)
        total        = mse + gamma_pen + c_pen
        return total, mse, gamma_pen, c_pen

    @torch.no_grad()
    def recompute_gamma(
        self, feature_target_iter: Iterator[tuple[torch.Tensor, torch.Tensor]]
    ) -> None:
        """Re-solve γ on the full dataset (should be called after each epoch).

        Args:
            feature_target_iter: yields (f [N, d_sae], y [N, K]) pairs
        """
        V  = self.V
        d1 = V + 1
        dev = self.binary_map.device
        A_acc = torch.zeros(self.K, d1, d1, device=dev)
        b_acc = torch.zeros(self.K, d1,  1, device=dev)

        for f_batch, y_batch in feature_target_iter:
            f_batch = f_batch.to(dev, dtype=self.C.dtype)
            y_batch = y_batch.to(dev, dtype=self.C.dtype)
            f_c     = self._remap(f_batch)
            Xt      = self._build_X_tilde_aug(f_c)          # [K, N, d1]
            A_acc  += torch.bmm(Xt.transpose(1, 2), Xt)
            T_k     = y_batch.t().unsqueeze(-1)              # [K, N, 1]
            b_acc  += torch.bmm(Xt.transpose(1, 2), T_k)

        A_reg        = A_acc + self.lambda_reg * self.W.unsqueeze(0)
        self._gamma  = torch.linalg.solve(A_reg, b_acc)     # [K, d1, 1]
        self._update_beta()

    @torch.no_grad()
    def _update_beta(self) -> None:
        C_pos         = F.softplus(self.C)                   # [K, V]
        self.beta_0.copy_(self._gamma[:, 0, 0])
        self.beta.copy_(C_pos * self._gamma[:, 1:, 0])

    # ──────────────────────────────────────────────
    #  Interpretability extraction
    # ──────────────────────────────────────────────

    @torch.no_grad()
    def extract_beta(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return interpretable coefficients.

        Returns:
            beta_0:  [K]                intercept per class
            beta:    [K, V]             feature importance (compact vocab)
            beta_full: [K, d_sae]       β mapped back to full SAE space (zeros elsewhere)
        """
        beta_full = torch.zeros(self.K, self.d_sae,
                                device=self.beta.device, dtype=self.beta.dtype)
        beta_full[:, self.active_indices] = self.beta
        return self.beta_0, self.beta, beta_full

    @torch.no_grad()
    def extract_G_matrix(
        self, f_dataset: torch.Tensor, batch_size: int = 512
    ) -> torch.Tensor:
        """Compute G corrections for every (sample, active feature) pair.

        Args:
            f_dataset: [N, d_sae] feature tensor
        Returns:
            G_mat: [N, V, K]
        """
        was_training = self.training
        self.eval()
        results = []
        for i in range(0, len(f_dataset), batch_size):
            f_b   = self._remap(f_dataset[i:i+batch_size].to(self.binary_map.device,
                                                               dtype=self.C.dtype))
            G     = self._forward_correction(f_b)       # [B, V, K]
            results.append(G.cpu())
        if was_training:
            self.train()
        return torch.cat(results, dim=0)                # [N, V, K]

    @torch.no_grad()
    def linearity_ratio(
        self, G_mat: torch.Tensor, f_active_mask: torch.Tensor = None
    ) -> float:
        """Compute LR = ||β||_1 / (||β||_1 + mean_n ||G_n(active)||_1).

        Only active features (f_j ≠ 0) contribute to G_norm, because inactive
        features have zero contribution to the NIMO prediction regardless of G.

        Args:
            G_mat:         [N, V, K]  correction matrix from extract_G_matrix()
            f_active_mask: [N, V] bool — True where feature j is non-zero for sample n.
                           If None, falls back to summing over all V features (biased).
        Returns:
            LR ∈ [0, 1]:  1 means purely linear, 0 means purely nonlinear
        """
        beta_norm = self.beta.abs().sum().item()
        if f_active_mask is not None:
            # [N, V] — G contribution only where feature is active
            G_flat = G_mat.squeeze(-1)          # [N, V]  (K=1)
            G_active = G_flat.abs() * f_active_mask.float().to(G_flat.device)
            G_norm_avg = G_active.sum(dim=1).mean().item()
        else:
            G_norm_avg = G_mat.abs().sum(dim=(1, 2)).mean().item()
        denom = beta_norm + G_norm_avg
        return beta_norm / denom if denom > 0 else 1.0
