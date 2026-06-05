import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionMLP(nn.Module):
    """Shared nonlinear correction network g_theta with tricks from original NIMO.

    Architecture:
        fc1 : Linear(d + n_bits, hidden) → scaled Tanh (+ noise during training)
        skip: re-inject binary encoding after fc1
        fc2 : Linear(hidden + n_bits, hidden) → sin(2π··) for periodic capacity
        fc3 : Linear(2·hidden + n_bits, K)

    The G(0)=0 centering and output bounding (tanh · learnable_alpha)
    are applied externally in forward_correction().
    """

    def __init__(self, input_dim, n_bits, hidden_dim, n_classes, noise_std=0.2, dropout=0.1):
        super().__init__()
        self.d_feat    = input_dim   # number of feature dims (d), binary dims come after
        self.noise_std = noise_std

        self.fc1 = nn.Linear(input_dim + n_bits, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + n_bits, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim + n_bits, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [..., d + n_bits]
        binary = x[..., self.d_feat:]                  # [..., n_bits]

        z1 = self.fc1(x)
        if self.training and self.noise_std > 0:
            z1 = z1 + self.noise_std * torch.randn_like(z1)
        z1 = torch.tanh(0.3 * z1)                      # scaled tanh; stays near linear at init

        z1_cat = torch.cat([z1, binary], dim=-1)        # [..., hidden + n_bits]  (re-inject binary)
        z2 = torch.sin(2 * torch.pi * self.fc2(z1_cat)) # [..., hidden]  (periodic capacity)
        z2 = self.dropout(z2)

        z = torch.cat([z2, z1_cat], dim=-1)             # [..., 2·hidden + n_bits]
        return self.fc3(z)                              # [..., K]


class NIMO(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, lambda_reg=1.0, mu_reg=1.0, noise_std=0.2):
        super().__init__()
        self.d = num_features
        self.K = num_classes
        self.lambda_reg = lambda_reg
        self.mu_reg = mu_reg

        # 1. Binary map as positional encoding (replaces nn.Embedding)
        # Number of bits needed: n_bits = floor(log2(d)) + 1
        self.n_bits = int(torch.floor(torch.log2(torch.tensor(float(self.d))))) + 1

        # Vectorised construction of binary matrix [d, n_bits]
        # Encode indices 1..d, centred at 0 (values ±0.5)
        indices = torch.arange(1, self.d + 1, dtype=torch.int64).unsqueeze(1)
        powers = 2 ** torch.arange(self.n_bits - 1, -1, -1, dtype=torch.int64)
        binary_map = (indices.bitwise_and(powers) > 0).float() - 0.5

        # Register as buffer so it moves with the model but receives no gradients
        self.register_buffer('binary_map', binary_map)

        # 2. Shared nonlinear correction network g_theta
        # Input: [x_{-j} (d dims) | binary_j (n_bits dims)]
        self.shared_mlp = CorrectionMLP(
            input_dim  = self.d,
            n_bits     = self.n_bits,
            hidden_dim = hidden_dim,
            n_classes  = self.K,
            noise_std  = noise_std,
        )

        # Learnable output scale for G: G_bounded = tanh(G) * 0.5 * (1 + tanh(alpha2))
        # Initialised to 4.0 → initial bound ≈ 0.5 * (1 + tanh(4)) ≈ 0.9993 ≈ 1.0
        self.alpha2 = nn.Parameter(torch.tensor(4.0))

        # 3. Continuously reparameterised scaling matrix C
        self.C = nn.Parameter(torch.empty(self.K, self.d).normal_(0.5, 0.1))

        # 4. Auxiliary buffers
        self.register_buffer('mask_matrix', 1.0 - torch.eye(self.d))

        W = torch.eye(self.d + 1)
        W[0, 0] = 0.0  # do not penalise the intercept
        self.register_buffer('W', W)

        # 5. Persistent gamma and beta (updated after each solve; saved in state_dict)
        self.register_buffer('_cache_gamma', torch.zeros(self.K, self.d + 1, 1))
        self.register_buffer('beta_0', torch.zeros(self.K))       # intercept per class [K]
        self.register_buffer('beta', torch.zeros(self.K, self.d)) # feature contributions [K, d]

    def forward_correction(self, X):
        """Compute the nonlinear correction term G, fusing binary-map positional encoding."""
        N = X.size(0)

        # [N, d, d]: dim-1 is target feature j, dim-2 is the context x_{-j}
        X_masked = X.unsqueeze(1) * self.mask_matrix.unsqueeze(0)

        # Expand binary_map to batch dimension: [d, n_bits] -> [N, d, n_bits]
        bin_embeds = self.binary_map.unsqueeze(0).expand(N, -1, -1)

        # [N, d, d + n_bits]
        mlp_input = torch.cat([X_masked, bin_embeds], dim=-1)

        G_raw = self.shared_mlp(mlp_input)  # [N, d, K]

        # Enforce G(0) = 0
        zero_masked = torch.zeros(self.d, self.d, device=X.device)
        zero_input = torch.cat([zero_masked, self.binary_map], dim=-1)  # [d, d + n_bits]
        G_zero = self.shared_mlp(zero_input)  # [d, K]

        G_centered = G_raw - G_zero.unsqueeze(0)  # [N, d, K]  — enforce G(0)=0

        # Bound G: tanh(G) * 0.5 * (1 + tanh(alpha2)), same as original NIMO
        bound = 0.5 * (1.0 + torch.tanh(self.alpha2))
        G_final = torch.tanh(G_centered) * bound
        return G_final

    def _build_X_tilde_aug(self, X):
        """Construct the augmented scaled feature matrix used by forward() and recompute_gamma()."""
        N = X.size(0)
        G = self.forward_correction(X)

        B = X.unsqueeze(-1) * (1.0 + G)                    # [N, d, K]
        B_k = B.permute(2, 0, 1)                            # [K, N, d]
        C_pos = F.softplus(self.C)
        X_tilde = B_k * C_pos.unsqueeze(1)                  # [K, N, d]
        ones = torch.ones(self.K, N, 1, device=X.device, dtype=X.dtype)
        return torch.cat([ones, X_tilde], dim=2)             # [K, N, d+1]

    def forward(self, X, T_logits=None):
        """
        Decoupled forward pass:
        - During training, pass T_logits to solve for gamma via closed form.
        - During inference, only pass X; prediction uses the last cached gamma.
        """
        X_tilde_aug = self._build_X_tilde_aug(X)            # [K, N, d+1]

        if self.training and T_logits is not None:
            # === Training: closed-form solution ===
            A = torch.bmm(X_tilde_aug.transpose(1, 2), X_tilde_aug)  # [K, d+1, d+1]
            A_reg = A + self.lambda_reg * self.W.unsqueeze(0)         # broadcast regularisation

            T_k = T_logits.t().unsqueeze(-1)                          # [K, N, 1]
            b = torch.bmm(X_tilde_aug.transpose(1, 2), T_k)          # [K, d+1, 1]

            # Numerically stable solver
            gamma = torch.linalg.solve(A_reg, b)                      # [K, d+1, 1]

            # Cache for inference (beta is updated only via recompute_gamma for accuracy)
            self._cache_gamma = gamma

            Y_hat_k = torch.bmm(X_tilde_aug, gamma)                   # [K, N, 1]
            Y_hat = Y_hat_k.squeeze(-1).t()                            # [N, K]
            return Y_hat

        else:
            # === Inference ===
            if not hasattr(self, '_cache_gamma'):
                raise RuntimeError("Model has not yet completed a forward pass with targets; gamma is uninitialised.")

            gamma = self._cache_gamma
            Y_hat_k = torch.bmm(X_tilde_aug, gamma)
            Y_hat = Y_hat_k.squeeze(-1).t()
            return Y_hat

    def compute_loss(self, Y_hat, T_logits):
        """
        Compute the training loss.
        Uses reduction='mean' for MSE so the loss scale is independent of batch size.
        Regularisation terms are divided by N·K to keep the same relative weighting
        regardless of batch size or number of classes.
        """
        N = Y_hat.size(0)

        # Primary loss: per-sample per-class mean
        mse_loss = F.mse_loss(Y_hat, T_logits, reduction='mean')

        # Gamma penalty — normalised by N so lambda_reg is batch-size independent
        gamma = self._cache_gamma
        gamma_penalty = self.lambda_reg * torch.sum(gamma[:, 1:, :] ** 2) / (N * self.K)

        # Sparsity-driving penalty on C — fixed size, normalise by K·d
        C_pos = F.softplus(self.C)
        c_penalty = self.mu_reg * torch.sum(C_pos ** 2) / (self.K * self.d)

        total_loss = mse_loss + gamma_penalty + c_penalty
        return total_loss, mse_loss, gamma_penalty, c_penalty

    @torch.no_grad()
    def recompute_gamma(self, feature_target_iter):
        """Re-solve gamma analytically over the full dataset using current weights.

        Mini-batch training caches gamma from the last batch only, which is a
        noisy estimate.  Call this after each epoch (before evaluation) to
        accumulate the normal equations across all batches and solve once:

            (Σ X̃ᵢᵀX̃ᵢ + λW) γ = Σ X̃ᵢᵀTᵢ

        Memory cost: accumulators are [K, d+1, d+1] and [K, d+1, 1] — tiny.
        Per-batch cost equals a regular forward pass (no gradients stored).

        Args:
            feature_target_iter: iterable of (Z [N, d], T_logits [N, K]) pairs
                                  where Z are post-extractor features.
        """
        d1 = self.d + 1
        dev = self.binary_map.device
        A_accum = torch.zeros(self.K, d1, d1, device=dev)
        b_accum = torch.zeros(self.K, d1, 1,  device=dev)

        for Z, T_logits in feature_target_iter:
            Z, T_logits = Z.to(dev), T_logits.to(dev)
            X_tilde_aug = self._build_X_tilde_aug(Z)                   # [K, N, d+1]
            A_accum += torch.bmm(X_tilde_aug.transpose(1, 2), X_tilde_aug)
            T_k      = T_logits.t().unsqueeze(-1)                       # [K, N, 1]
            b_accum += torch.bmm(X_tilde_aug.transpose(1, 2), T_k)

        A_reg = A_accum + self.lambda_reg * self.W.unsqueeze(0)
        self._cache_gamma = torch.linalg.solve(A_reg, b_accum)          # [K, d+1, 1]
        self._update_beta()

    @torch.no_grad()
    def _update_beta(self):
        """Compute and store interpretable beta from current gamma and C.
        beta_0 [K]: per-class intercept.
        beta   [K, d]: per-class feature contributions = C_pos * gamma[1:].
        Called automatically after each gamma solve (training or recompute_gamma).
        """
        C_pos = F.softplus(self.C)
        self.beta_0.copy_(self._cache_gamma[:, 0, 0])
        self.beta.copy_(C_pos * self._cache_gamma[:, 1:, 0])

    @torch.no_grad()
    def extract_interpretable_coefficients(self):
        """Return per-class interpretable coefficients stored in buffers.
        beta_0 [K]: intercept per class.
        beta   [K, d]: feature contributions per class.
        """
        return self.beta_0, self.beta



if __name__ == "__main__":
    # Quick test to verify the forward and loss computations run without error
    torch.manual_seed(0)
    model = NIMO(num_features=4, num_classes=1)
    X = torch.tensor([[1, 1, 1, 1],
                     [2, 2, 2, 2],
                     [3, 3, 3, 3],
                     [1, 2, 3, 4],
                     [2, 2, 3, 3],
                     [3, 3, 4, 4]], dtype=torch.float32)
    model.forward(X, T_logits=torch.randn(6, 1))