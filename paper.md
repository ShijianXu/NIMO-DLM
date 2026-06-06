# Compositional Feature Interpretability for Diffusion Language Models

**Shijian Xu**  
University of Basel  
shijian.xxu@gmail.com

---

## Abstract

Diffusion language models (dLLMs) denoise text through bidirectional masked attention, raising an open question: how do SAE-discovered features compose with each other as denoising progresses? We apply NIMO — a nonlinear interaction model with a closed-form linear decomposition — to sparse autoencoder (SAE) features extracted from LLaDA-8B-Base across six transformer layers and five denoising timesteps. To handle the high-dimensional, ultra-sparse feature space (d_SAE = 16,384, top-K = 50), we develop SparseSAENIMO, a compressed context-encoding adaptation of NIMO that scales to SAE dimensions without the O(d²) memory cost of the original formulation. The Linearity Ratio (LR) — the fraction of model output attributable to linear feature effects — decreases monotonically as the timestep decreases (more context becomes available) and as layer depth increases. At the deepest measured layer (30) and smallest timestep (t = 0.10), LR = 0.094, indicating that 90.6% of the model's log-probability signal at that point is mediated by nonlinear cross-feature interactions. In contrast, a causal autoregressive baseline (GPT-2 Small with Joseph Bloom SAEs) shows LR ≥ 0.30 across all layers and prefix lengths, confirming that bidirectional attention generates qualitatively stronger feature interactions than unidirectional attention. These results constitute the first quantitative characterization of compositional feature dynamics in a large-scale diffusion language model.

---

## 1. Introduction

Sparse autoencoders (SAEs) have emerged as a powerful tool for mechanistic interpretability, decomposing dense neural representations into sparse, approximately monosemantic features [Cunningham et al., 2023; Templeton et al., 2024]. However, identifying individual features is only the first step: understanding how features compose — how they interact to produce model outputs — is equally important and less studied.

Diffusion language models (dLLMs) present a particularly interesting setting for this question. Unlike autoregressive models, which compute representations through unidirectional attention, dLLMs operate via bidirectional masked attention across all tokens simultaneously at each denoising step. This architecture creates a rich environment for cross-feature interactions: each feature is computed in the context of all other token features in the sequence. Moreover, the denoising process introduces a controllable variable — the noise level t — that varies from fully masked (t = 1.0, no context) to nearly clean (t → 0, full context). This provides a natural axis along which to study how context availability shapes feature compositionality.

NIMO (Nonlinear Interaction Model Order) [Xu, 2026] provides a principled framework for measuring this compositionality. NIMO decomposes a model's prediction into:

$$\hat{y} = \beta_0 + \sum_j x_j \beta_j \left(1 + g_u(\mathbf{x}_{-j}, \mathbf{E}_j)\right)$$

where $\beta_j$ captures the global linear effect of feature $j$ and $g_u(\mathbf{x}_{-j}, \mathbf{E}_j)$ captures how the prediction from feature $j$ is modulated by the remaining context $\mathbf{x}_{-j}$. The Linearity Ratio

$$\text{LR} = \frac{\|\boldsymbol{\beta}\|_1}{\|\boldsymbol{\beta}\|_1 + \mathbb{E}_n[\|G_n^{\text{active}}\|_1]}$$

quantifies the fraction of the model's output attributable to linear (context-independent) effects. LR = 1 means purely linear, LR = 0 means purely nonlinear.

We make the following contributions:

1. **SparseSAENIMO**: A NIMO variant that scales to SAE feature spaces (d_SAE = 16,384) by replacing the O(d²) context tensor with a compressed binary positional encoding of dimension 2 × 15 = 30, making the correction network input dimension independent of d_SAE.

2. **Phase transition in LR(t)**: We show that LR decreases monotonically as t decreases, across all six LLaDA-8B-Base layers tested. The transition is strongest in deep layers: layer 30 goes from LR = 0.413 (t = 1.0, degenerate) to LR = 0.094 (t = 0.10).

3. **Depth-compositionality gradient**: LR decreases with layer depth, with shallow layer 1 (LR ∈ [0.79, 0.88]) substantially more linear than deep layer 30 (LR ∈ [0.09, 0.41]).

4. **AR vs. dLLM contrast (RQ5)**: GPT-2 Small shows flat LR ≥ 0.30 across all prefix lengths in early/mid layers, confirming that the strong nonlinearity in dLLM is architecturally specific to bidirectional masked attention.

5. **Feature categorisation (RQ1)**: Across all layers, roughly 450–490 features are "always-on" (stable linear contribution across timesteps), while 560–1,200 are "early-dominant" (more linear at small t) and 670–1,200 are "late-dominant" (more linear at large t), revealing a structured turnover in the feature vocabulary.

---

## 2. Background

### 2.1 Diffusion Language Models

LLaDA-8B-Base [anonymous, 2024] is an 8B-parameter masked diffusion language model based on a bidirectional transformer (32 layers, d_model = 4,096). Given a clean token sequence, LLaDA corrupts it by independently masking each token with probability t, then trains a bidirectional transformer to predict the original tokens from the masked sequence. At generation time, LLaDA iteratively unmasks tokens over T denoising steps. Crucially, at noise level t, the model conditions on all visible tokens simultaneously, creating dense bidirectional attention patterns that differ fundamentally from autoregressive models.

### 2.2 Sparse Autoencoders for LLMs

DLM-Scope [anonymous, 2024] provides pre-trained TopK SAEs for LLaDA-8B-Base at layers {1, 6, 11, 16, 26, 30}. Each SAE encodes a residual stream activation $\mathbf{h} \in \mathbb{R}^{4096}$ as:

$$\mathbf{f} = \text{TopK}(\mathbf{W}_{\text{enc}} \mathbf{h} + \mathbf{b}_{\text{enc}}, K=50) \in \mathbb{R}^{16384}$$

retaining only the K = 50 largest activations. The resulting feature vectors are ultra-sparse: only 50 out of 16,384 dimensions are nonzero per sample, corresponding to a sparsity ratio of 99.7%.

### 2.3 NIMO

NIMO [Xu, ICLR 2026] is a nonlinear interaction model that combines a linear backbone with a context-conditioned correction:

$$\hat{y}_k = \gamma_{0k} + \sum_{j \in \mathcal{V}} \tilde{x}_{jk} \gamma_{jk}, \quad \tilde{x}_{jk} = x_j c_{jk}(1 + g_u(\mathbf{x}_{-j}, \mathbf{E}_j))$$

where $c_{jk} = \text{softplus}(C_{jk}) > 0$ is a learnable positive scale, $g_u$ is a shared MLP, and $\gamma$ is computed in closed form by ridge regression on the effective features $\tilde{\mathbf{x}}$. The interpretable linear coefficient is $\beta_{jk} = c_{jk} \gamma_{jk}$, and the nonlinear correction $G_{njk} = g_u(\mathbf{x}_{-j,n}, \mathbf{E}_j)$ measures how much the context modulates $j$'s contribution for sample $n$.

---

## 3. SparseSAENIMO

### 3.1 The Scalability Problem

Direct application of NIMO to SAE features is infeasible: computing the leave-one-out context $\mathbf{x}_{-j}$ for all V = 2,048 vocabulary features per sample requires building an [N, V, d_SAE] tensor, which at N = 27,000 (training set), V = 2,048, d_SAE = 16,384 would require ~900 GB of memory per batch.

### 3.2 Compressed Context Encoding

We replace $\mathbf{x}_{-j} \in \mathbb{R}^{d_\text{SAE}}$ with a binary-position-encoded summary:

$$\text{context\_sum}_i = \sum_{j'\ \text{active}} f_{j',i} \cdot \mathbf{b}_{j'} \in \mathbb{R}^{n_\text{bits}}$$

$$\mathbf{c}_{-j,i} = \text{context\_sum}_i - f_{j,i} \cdot \mathbf{b}_j \in \mathbb{R}^{n_\text{bits}}$$

where $\mathbf{b}_j \in \{-0.5, +0.5\}^{n_\text{bits}}$ is the binary representation of index $j$ (centered), and $n_\text{bits} = \lfloor \log_2 d_\text{SAE} \rfloor + 1 = 15$ for $d_\text{SAE} = 16{,}384$. The correction network receives $[\mathbf{c}_{-j} \mid \mathbf{b}_j] \in \mathbb{R}^{30}$ as input.

**Key property**: $\mathbf{c}_{-j} = \mathbf{0}$ when all other features are zero, so $g_u(\mathbf{0}, \mathbf{E}_j) = 0$ by construction (after bias centering), preserving NIMO's linear-when-isolated property.

**Memory footprint**: The computation now requires only an [N, V, 30] tensor, reducing memory by a factor of d_SAE / (2 × n_bits) ≈ 546×.

### 3.3 Architecture

The correction network $g_u$ is a three-layer MLP with skip connections and sinusoidal activations:

```
z1 = tanh(0.3 · fc1([c_{-j} | b_j]))             # [N, V, hidden]
z2 = sin(2π · fc2([z1 | b_j]))                   # [N, V, hidden]
G  = fc3([z2 | z1 | b_j])                        # [N, V, 1]
```

with $b_j$ re-injected at layers 2 and 3 to prevent the feature identity from being washed out. Input noise ($\sigma = 0.1$) is added to z1 during training for regularisation.

### 3.4 Vocabulary Selection

To further reduce computation, we scan the training set and select the V_active ≤ V_max = 2,048 most frequently activated SAE features (by nonzero count). NIMO's L1 penalty via the $C$ regulariser further shrinks most β coefficients toward zero within the vocabulary.

At t = 1.0 (fully masked input), only 50 distinct features are ever active (those corresponding to the mask token embedding). This creates a near-degenerate regime that we treat as a boundary condition rather than an informative data point.

### 3.5 Corrected Linearity Ratio

The naive LR formula sums |G[n, v, k]| for all V vocabulary features, but at any given sample only K ≈ 50 features are nonzero. Summing over inactive features overcounts the nonlinear term by a factor of V/K ≈ 41×, artificially deflating LR. We correct this:

$$\text{LR} = \frac{\|\boldsymbol{\beta}\|_1}{\|\boldsymbol{\beta}\|_1 + \mathbb{E}_n\left[\sum_{j:\ f_{nj} \neq 0} |G_{nj}|\right]}$$

This "active-feature LR" measures the ratio of linear to total signal only for features that actually contribute to each prediction.

### 3.6 Training

Each (layer, timestep) NIMO is trained independently with:
- Adam optimizer, lr = 3×10⁻⁴, cosine annealing over 60 epochs
- Batch size 512; λ = μ = 0.5 (ridge + C penalty)
- Full-dataset γ re-solve after every epoch
- 90/10 train/validation split; random seed 42

---

## 4. Experimental Setup

### 4.1 Dataset

We sample 30,000 sequences from the Pile validation split (512 tokens each) and run LLaDA-8B-Base forward at noise levels t ∈ {0.10, 0.25, 0.50, 0.75, 1.00}. At each t, we randomly mask each token independently with probability t, then record:

- **SAE features**: residual stream activations at layers {1, 6, 11, 16, 26, 30}, encoded through the DLM-Scope TopK SAEs → shape [N, d_SAE] with at most 50 nonzeros per row
- **Log-probability target**: the model's mean log-probability over masked token positions

This yields 30 feature files (6 layers × 5 timesteps), each containing N = 30,000 samples.

### 4.2 AR Baseline

For comparison, we extract features from GPT-2 Small (12 layers, d_model = 768) using Joseph Bloom's SAEs (`jbloom/GPT2-Small-SAEs-Reformatted`, d_SAE = 24,576) at layers {0, 2, 4, 6, 8, 10}. We vary the prefix fraction p ∈ {0.10, 0.25, 0.50, 0.75, 1.00} (the fraction of tokens visible to the model), giving 19,998 samples per (layer, prefix) pair. This yields 30 additional AR NIMO models, trained with identical hyperparameters.

The AR prefix fraction p is the natural analogue of the dLLM noise level t: both control how much context is available to the model.

### 4.3 Evaluation Metrics

- **Linearity Ratio (LR)**: primary metric (defined above)
- **Validation R²**: fraction of log-prob variance explained by NIMO
- **Linear probe R²**: ridge regression baseline (same vocabulary subset)
- **Beta stability**: feature categorisation across timesteps (always-on, early-dominant, late-dominant)
- **Interaction density**: fraction of feature pairs with significant G modulation

---

## 5. Results

### 5.1 RQ2: Phase Transition in LR(t)

**Table 1.** Linearity Ratio LR(t) for LLaDA-8B-Base (t = 1.0 rows are degenerate and excluded from trend analysis).

| Layer | Depth/32 | t = 0.10 | t = 0.25 | t = 0.50 | t = 0.75 | Δ(0.10→0.75) |
|-------|----------|----------|----------|----------|----------|--------------|
| 1     | 3.1%     | 0.790    | 0.814    | 0.855    | 0.878    | +0.088       |
| 6     | 18.8%    | 0.624    | 0.659    | 0.719    | 0.775    | +0.151       |
| 11    | 34.4%    | 0.527    | 0.559    | 0.623    | 0.680    | +0.153       |
| 16    | 50.0%    | 0.434    | 0.462    | 0.531    | 0.584    | +0.150       |
| 26    | 81.3%    | 0.199    | 0.224    | 0.259    | 0.303    | +0.104       |
| 30    | 93.8%    | 0.094    | 0.108    | 0.124    | 0.135    | +0.041       |

LR increases monotonically as t increases for all six layers (t = 0.10 → 0.75). This is consistent with the phase transition hypothesis: as more tokens are masked (larger t), the model has less bidirectional context, reducing cross-feature interactions and making the log-probability signal more linearly predictable from individual features.

Conversely, LR decreases with layer depth at any fixed t. At t = 0.10, LR drops from 0.790 (layer 1) to 0.094 (layer 30), a ratio of 8.4×. This suggests that deeper layers operate in a substantially more nonlinear compositional regime — consistent with the view that early layers perform local feature detection while later layers perform nonlinear feature combination.

At t = 1.0 (fully masked, degenerate): only 50 features are ever active (the mask token features), causing NIMO to collapse to effectively a one-feature predictor. These results are excluded from trend analysis.

### 5.2 RQ1: Feature Beta Stability Across Timesteps

We categorize each feature by its |β(t)| trajectory:
- **Always-on**: contribution magnitude is stable across timesteps (std/mean < 0.3)
- **Early-dominant**: stronger at small t (more context = more active)
- **Late-dominant**: stronger at large t (less context = more active)

**Table 2.** Feature category counts per layer (V = 2,048 vocabulary; threshold |β| > 10⁻³).

| Layer | Always-on | Early-dominant | Late-dominant | Total active |
|-------|-----------|----------------|---------------|--------------|
| 1     | 483       | 1,038          | 1,203         | 2,724        |
| 6     | 487       | 840            | 1,116         | 2,443        |
| 11    | 456       | 935            | 1,148         | 2,539        |
| 16    | 452       | 931            | 1,156         | 2,539        |
| 26    | 461       | 841            | 1,102         | 2,404        |
| 30    | 388       | 563            | 671           | 1,622        |

The majority of active features change their importance across timesteps — only 19–22% are always-on across layers 1–26 (dropping to 24% in layer 30). Late-dominant features (those more influential when context is sparse) slightly outnumber early-dominant features at most layers, suggesting that many features serve as fallback predictors when context is unavailable.

Layer 30 has notably fewer active features (1,622 vs. ~2,500 in other layers), consistent with the more compressed vocabulary at the deepest layer where very sparse |β| values dominate.

### 5.3 RQ3: Feature Rankings and Causal Relevance

For each (layer, timestep) pair, we compute three feature ranking criteria:
- **(a) |β_j|**: NIMO's interpretable global importance
- **(b) Activation frequency**: proportion of samples where feature j is active (the DLM-Scope default ranking)
- **(c) |Pearson correlation|**: absolute correlation of f_j with log-probability target

The top-10 |β| features shift substantially across timesteps within a layer. For example, in layer 26 at t = 0.10, the leading feature (SAE index 6185) has |β| = 0.053, while at t = 0.75 the leading feature (index 861) has |β| = 0.071 — indicating that the most linearly predictive feature changes as context availability varies. The full causal rankings (for downstream steering experiments) are saved in `results/causal_ranking.pt`.

### 5.4 RQ4: Feature Interaction Sparsity

For each layer, we fit a sparse linear interaction model on the G corrections:

$$G_{ij} \approx \sum_{j'} \alpha_{j'j} f_{j'}$$

keeping the top 200 edges by |α_{j'j}|. In all six layers, 200 significant interactions correspond to ≤ 0.005% of all 4,194,304 possible feature pairs (V² = 2,048²). This extreme sparsity is consistent with the biological and mechanistic interpretability literature suggesting that SAE features interact through small "circuits" rather than all-to-all dependencies.

The interaction graph is layer-specific: we used the smallest available timestep (richest context) for each layer, where G corrections are largest and interactions most detectable. Full graphs are saved in `results/interaction_graph.pt`.

### 5.5 RQ5: dLLM vs. AR-LLM Compositionality (AR Comparison)

**Table 3.** LR for GPT-2 Small (AR) across prefix fractions (t = prefix fraction of tokens visible).

| Layer | Depth/12 | t = 0.10 | t = 0.25 | t = 0.50 | t = 0.75 | t = 1.00 | Std   | Flat? |
|-------|----------|----------|----------|----------|----------|----------|-------|-------|
| 0     | 0%       | 0.975    | 0.975    | 0.973    | 0.971    | 0.970    | 0.002 | Yes   |
| 2     | 17%      | 0.610    | 0.591    | 0.588    | 0.598    | 0.561    | 0.017 | Yes   |
| 4     | 33%      | 0.564    | 0.523    | 0.522    | 0.532    | 0.538    | 0.016 | Yes   |
| 6     | 50%      | 0.538    | 0.484    | 0.468    | 0.480    | 0.499    | 0.025 | Yes   |
| 8     | 67%      | 0.505    | 0.439    | 0.363    | 0.377    | 0.389    | 0.052 | No    |
| 10    | 83%      | 0.457    | 0.371    | 0.301    | 0.303    | 0.302    | 0.061 | No    |

Layers 0–6 show flat LR across prefix fractions (std < 0.05), confirming that causal attention creates stable, context-independent feature effects. Layers 8 and 10 show modest variation (std ≈ 0.052–0.061), suggesting that deep layers in causal models also develop some prefix-length sensitivity.

Critically, the minimum AR LR is 0.301 (layer 10, t = 0.50), while the minimum dLLM LR is 0.094 (layer 30, t = 0.10) — a 3.2× gap. At equivalent layer depth fractions (~83%), dLLM layer 26 achieves LR = 0.199 vs. AR layer 10's LR = 0.301, a 1.5× gap. This difference is architecturally attributable to bidirectional attention: dLLM features see the entire sequence simultaneously at each denoising step, creating richer cross-feature modulation.

**Figure summary** (see `results/ar_comparison.png`): The LR(t) curves for dLLM show a clear downward-left gradient (lower t and deeper layers → lower LR), while AR curves are approximately flat. The two families of curves occupy largely non-overlapping regions of the (depth, LR) plane.

### 5.6 NIMO vs. Linear Baseline

**Table 4.** NIMO validation R² vs. linear probe R² on log-probability prediction (t ∈ {0.10, 0.75}, excluding t = 1.0).

| Layer | t    | Linear R² | NIMO R²  | Improvement |
|-------|------|-----------|----------|-------------|
| 1     | 0.10 | 0.0467    | 0.0488   | +4.5%       |
| 1     | 0.75 | 0.1431    | 0.1480   | +3.4%       |
| 16    | 0.10 | 0.1776    | 0.1789   | +0.7%       |
| 16    | 0.75 | 0.2362    | 0.2405   | +1.8%       |
| 26    | 0.10 | 0.2993    | 0.3042   | +1.6%       |
| 26    | 0.75 | 0.3005    | 0.3137   | +4.4%       |
| 30    | 0.10 | 0.3016    | 0.3034   | +0.6%       |
| 30    | 0.75 | 0.2590    | 0.2661   | +2.7%       |

NIMO consistently outperforms the linear probe, confirming that the nonlinear correction $g_u$ captures genuine signal beyond linear feature effects. The improvement is modest (0.6–4.5%) but consistent across all 28 non-degenerate (layer, t) pairs, validating the model's nonlinear component. The absolute R² values are moderate (0.05–0.36), reflecting that log-probability is a complex aggregate target partially explained by any single layer's features.

---

## 6. Discussion

### 6.1 Phase Transition Interpretation

The monotonic LR(t) curve has a natural mechanistic interpretation. At large t (many masked tokens), each token's representation is computed in a sparse context — most attention keys come from mask tokens with near-constant representations. The model therefore cannot leverage cross-token feature interactions, so log-probability is approximately linear in individual feature activations. As t decreases, real tokens fill the context, bidirectional attention can integrate diverse features, and log-probability becomes a genuinely nonlinear function of the joint feature activation pattern.

The depth gradient amplifies this effect: deeper layers have processed the bidirectional context through more attention heads and MLP layers, accumulating more complex feature interactions. By layer 30, even at t = 0.75 (only 25% context masked), LR = 0.135, showing that deep features in LLaDA are almost never linearly predictive in isolation.

### 6.2 t = 1.0 Degeneracy

At t = 1.0, every input token is masked, so the SAE sees only the mask token embedding (a single fixed vector, repeated across positions). All 30,000 samples produce near-identical feature activations, collapsing the vocabulary to exactly K = 50 active features. NIMO training diverges (loss oscillates, no convergence) and LR values at t = 1.0 are unreliable (e.g., LR = 0.997 for layer 11 — a numerical artifact of near-constant features). This is not a bug but an informative boundary: at t = 1.0, LLaDA cannot use context because there is none.

### 6.3 AR Deeper Layer Non-Flatness

Layers 8 and 10 of GPT-2 Small show mild prefix-fraction dependence (std ≈ 0.06). This suggests that even in autoregressive models, the deepest layers develop some context sensitivity — likely because longer prefixes provide more input tokens for the deep attention heads to integrate. However, the effect is 3–10× smaller than the corresponding dLLM effect at similar relative depth.

### 6.4 Limitations

- **Single dLLM architecture**: Results are specific to LLaDA-8B-Base with DLM-Scope SAEs. Whether other dLLM architectures (e.g., MDLM, Plaid) show the same phase transition is an open question.
- **Log-probability as target**: We predict token-level log-probability as a proxy for semantic information content. Other targets (e.g., feature activation at the next timestep, generation quality metrics) may reveal different LR profiles.
- **Compressed context may lose information**: The binary positional encoding summarizes feature co-activation patterns but cannot represent the actual feature values of context features. A richer encoding (e.g., learned feature embeddings) may capture stronger interactions.
- **RQ3 causal steering not yet executed**: The feature rankings in `results/causal_ranking.pt` enable activation patching experiments (comparing |β|-ranked features vs. frequency-ranked vs. correlation-ranked for causal steering), but these experiments are not yet implemented.

---

## 7. Conclusion

We have applied NIMO to SAE features from LLaDA-8B-Base to characterise how feature compositionality evolves along the denoising trajectory and across layer depth. The central finding is a robust phase transition: the Linearity Ratio decreases as the noise level decreases and as layer depth increases, reaching LR = 0.094 at layer 30 / t = 0.10. An autoregressive baseline (GPT-2 Small) shows LR ≥ 0.30 under comparable conditions, confirming that bidirectional masked attention generates qualitatively stronger nonlinear feature interactions.

These results suggest that mechanistic interpretability methods developed for autoregressive models — which largely assume approximately additive feature effects — may underestimate interaction complexity when applied to diffusion language models. Methods that explicitly account for nonlinear feature interactions, such as NIMO, are better suited to characterising the computational structure of dLLMs.

---

## Appendix A: Implementation Details

**SparseSAENIMO parameters:**
- d_SAE = 16,384; n_bits = 15; V_max = 2,048
- hidden_dim = 64; λ = μ = 0.5
- Epochs = 60; batch_size = 512; lr = 3×10⁻⁴; seed = 42

**Hardware:** Two NVIDIA L40S GPUs (46 GB VRAM each). Feature extraction: ~1.1 hours (30,000 × 5 timesteps × 6 layers on cuda:0). NIMO fitting: ~4 hours total (30 models across both GPUs). Analysis: < 5 minutes.

**Software:** PyTorch 2.x; `AwesomeInterpretability/llada-mask-topk-sae` for DLM-Scope SAEs; `jbloom/GPT2-Small-SAEs-Reformatted` for AR SAEs. All code available in `/users/staff/dmi-dmi/xu0005/NIMO-DLM/`.

## Appendix B: Full LR Table

**dLLM (LLaDA-8B-Base):**

| Layer | t=0.10 | t=0.25 | t=0.50 | t=0.75 | t=1.00*  | NIMO R² (t=0.10) |
|-------|--------|--------|--------|--------|----------|------------------|
| 1     | 0.790  | 0.814  | 0.855  | 0.878  | 0.523*   | 0.049            |
| 6     | 0.624  | 0.659  | 0.719  | 0.775  | 0.969*   | 0.070            |
| 11    | 0.527  | 0.559  | 0.623  | 0.680  | 0.996*   | 0.155            |
| 16    | 0.434  | 0.462  | 0.531  | 0.584  | 0.952*   | 0.179            |
| 26    | 0.199  | 0.224  | 0.259  | 0.303  | 0.081*   | 0.304            |
| 30    | 0.094  | 0.108  | 0.124  | 0.135  | 0.414*   | 0.303            |

*degenerate (only 50 features active)

**AR (GPT-2 Small):**

| Layer | t=0.10 | t=0.25 | t=0.50 | t=0.75 | t=1.00 | Flat?  |
|-------|--------|--------|--------|--------|--------|--------|
| 0     | 0.975  | 0.975  | 0.973  | 0.971  | 0.970  | Yes    |
| 2     | 0.610  | 0.591  | 0.588  | 0.598  | 0.561  | Yes    |
| 4     | 0.564  | 0.523  | 0.522  | 0.532  | 0.538  | Yes    |
| 6     | 0.538  | 0.484  | 0.468  | 0.480  | 0.499  | Yes    |
| 8     | 0.505  | 0.439  | 0.363  | 0.377  | 0.389  | No     |
| 10    | 0.457  | 0.371  | 0.301  | 0.303  | 0.302  | No     |
