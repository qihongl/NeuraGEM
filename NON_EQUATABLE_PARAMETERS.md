# Non-Equatable Parameters: NeuraGEM vs. EGO on Beukers et al. Task

**Date:** 2026-06-06
**Goal:** Compare NeuraGEM (Hummos et al., 2026) and EGO (Giallanza et al., 2024) on the Beukers et al. (2024) sequence learning task.

## Parameters Successfully Equated

| Parameter | Value | Status |
|-----------|-------|--------|
| State representation | 10D one-hot (10 states, 0–9) | ✓ |
| Story length | 6 steps per story | ✓ |
| Block size (blocked) | 120 timesteps = 20 stories | ✓ |
| Block size (interleaved) | 6 timesteps = 1 story | ✓ |
| Blocked phase length (pure) | 1800 timesteps | ✓ |
| Interleaved phase length (pure) | 1800 timesteps | ✓ |
| Interleaved→Blocked: interleaved | 500 timesteps | ✓ |
| Interleaved→Blocked: blocked | 1000 timesteps | ✓ |
| Testing phase | 240 timesteps (40 stories, frozen weights) | ✓ |
| Seeds | 20 | ✓ |
| Online learning | batch_size=1, 1 epoch | ✓ |

## Non-Equatable Parameters

### 1. Learning Rate Magnitudes

| Model | Fast Updates | Slow/Weight Updates |
|-------|-------------|---------------------|
| **NeuraGEM** | Z: Adam, lr = 0.3 | W (LSTM): Adam, lr = 0.001 |
| **EGO** | Context: SGD, lr = 0.5 | N/A (only context module trainable) |

**Why non-equatable:** The EGO context module uses SGD with bias-based persistence gating; NeuraGEM's Z uses Adam with explicit L2 decay. The effective learning dynamics depend on the optimizer's adaptive mechanisms (momentum, adaptive scaling) and the architecture's gradient magnitudes (gated LSTM vs. MGRU). A numerically equal lr would produce different effective step sizes.

**Mitigation:** Sweep EGO's `episodic_lr` across {0.1, 0.5, 1.0, 2.0} and report the regime that produces stable learning. Use the same lr for all curricula.

### 2. Effective Model Capacity

| Aspect | NeuraGEM | EGO |
|--------|----------|-----|
| Parameter count (learnable) | ~4K (LSTM weights) | ~40 (hidden_to_context.weight: 10×4) |
| Memory capacity | Fixed: 2D Z + LSTM hidden state (32D) | Unbounded: EM stores every past (state, context, next_state) tuple |
| Information storage | Compressed in weights + latent variable | Exact copies in key-value store |
| Retrieval | Recurrent dynamics | Cosine-similarity weighted average |

**Why non-equatable:** EGO's episodic memory grows linearly with experience (one row per timestep — ~1800–3000 rows), providing near-perfect recall of past observations. NeuraGEM must compress this information into a 2-dimensional latent Z and 32-dimensional LSTM hidden state. This is a fundamental architectural claim of each paper, not a tuneable parameter.

### 3. Gradient Propagation Depth

| Aspect | NeuraGEM | EGO |
|--------|----------|-----|
| BPTT depth | Full (seq_len=10 or 18 steps) | Truncated (1 step back) |
| Gradient flow | Through LSTM unrolled across full sequence | Through single-step context update only |

**Why non-equatable:** NeuraGEM backpropagates error through the entire sequence window, allowing gradients to flow across multiple timesteps. EGO deliberately truncates BPTT after one step, as a biological-plausibility constraint. Matching this would require fundamentally changing either model's architecture.

### 4. Context Maintenance Mechanism

| Aspect | NeuraGEM | EGO |
|--------|----------|-----|
| Mechanism | Z (2D softmax, L2 decay = 0.00004) optimized at inference time | Context (4D) via persistent MGRU with bias=1.0 gating |
| Decay | Explicit L2 regularization | Implicit through gating weight dynamics |
| Update signal | Prediction error gradient through LSTM | Prediction error gradient through single linear layer |

### 5. Hidden Unit Architecture

| Aspect | NeuraGEM | EGO |
|--------|----------|-----|
| Recurrent unit | LSTM (32 hidden units) | Minimal Gated Recurrent Unit (10 hidden units) |
| Gating mechanism | Input/forget/output gates | Single update gate |
| Z modulation | Multiplicative gating of LSTM hidden state via sigmoid(Z) × binary mask | Context vector concatenated into query |

### 6. Prediction Mechanism

| Aspect | NeuraGEM | EGO |
|--------|----------|-----|
| Prediction | LSTM output → Linear(32, 10) → one-step-ahead state | EM query(state, context) → weighted average of stored next states |
| Error signal | Direct comparison with true next state | Direct comparison with true next state |
| Generalization | Through LSTM weight interpolation | Through similarity-based retrieval from exemplars |

## Implications for Comparison

These non-equatable parameters are not bugs — they are the *research question*. The comparison tests whether two fundamentally different architectures for latent-state inference (gradient-based Z optimization vs. episodic memory retrieval) produce similar or divergent behavioral signatures on the same task with matched environmental statistics. Any difference in results should be interpreted through the lens of these architectural differences.

## LR Sweep Plan

| episodc_lr | Expected behavior |
|-----------|-------------------|
| 0.1 | Slow learning, may not reach ceiling under blocked training |
| 0.5 | Baseline (matches EGO paper values) |
| 1.0 | Faster learning, may overfit |
| 2.0 | Very fast, risk of instability |

Default: 0.5. If blocked performance is sub-ceiling, increase to 1.0.
