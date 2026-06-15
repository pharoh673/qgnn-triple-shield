# Results — Does a permutation-equivariant quantum tagger act as a "triple shield"?

**Task.** Binary jet tagging on JetNet, **gluon vs top**, particle-cloud inputs
(`n_const=16`, features `[Δη, Δφ, log pT, log E]` + validity mask). Models all consume
the same input:

| Model | Description | Trainable params (nq=4) |
|---|---|---|
| **Equivariant QNN** | shared per-constituent circuit + masked-mean pool (Quantum Deep Sets) | 61 |
| **Non-equivariant QNN** | same circuit + sinusoidal positional encoding (symmetry broken), budget-matched | 61 |
| **Classical PFN (matched)** | Deep Sets MLP shrunk to equal parameter budget | 61 |
| **Classical PFN (reference)** | competent Deep Sets MLP (honest classical ceiling) | 641 |

**Design.** One-Axis-At-a-Time sweep anchored at `nq=4, N=200, noiseless`; 10 replicates
per cell (20 for the learning-curve block), mean ± std reported. Seeds derived
deterministically from `(global_seed, N, replicate)`; fixed held-out test set across all
cells; standardizer fit on the train split only. Quantum sims on `default.qubit`
(noiseless / correlated) and `default.mixed` (iid), trained with Adam + early stopping +
random restarts to escape vanishing-gradient initializations.

The hypothesis: the same permutation symmetry that shrinks the hypothesis space should
let the equivariant model (1) generalize from fewer samples, (2) work at fewer qubits,
and (3) tolerate more (especially correlated) noise — than a budget-matched
non-equivariant control.

---

## 1. Data efficiency — learning curves  ·  `figures/learning_curve.png`

Test ROC-AUC vs training size N (nq=4, noiseless, 20 replicates):

| N | Equivariant | Non-equiv | Classical (ref) | Classical (matched) |
|---|---|---|---|---|
| 20 | 0.579 | 0.574 | 0.676 | 0.537 |
| 50 | **0.786** | 0.701 | 0.805 | 0.558 |
| 100 | **0.820** | 0.760 | 0.844 | 0.599 |
| 200 | 0.838 | 0.820 | 0.860 | 0.605 |
| 500 | 0.853 | 0.845 | 0.868 | 0.629 |
| 1000 | 0.862 | 0.850 | 0.870 | 0.635 |

**Verdict: shield holds vs the matched control.** The equivariant model beats the
budget-matched non-equivariant QNN at every N≥50, with the gap largest in the
data-scarce regime (N=50: **0.786 vs 0.701**) — the non-equivariant model needs roughly
4× more data to match what equivariance buys for free. Both quantum models vastly
outperform the parameter-matched classical net (≤0.64), confirming the quantum models use
their 61 parameters far more effectively.

**Honest caveat.** The *unconstrained* classical PFN (641 params) remains the top curve at
every N. The equivariant QNN is competitive with — not superior to — a competent classical
baseline. The clean, defensible claim is **symmetry > no-symmetry at equal budget**, not
"quantum beats classical".

## 2. Generalization gap  ·  `figures/generalization_gap.png`

Train AUC − test AUC (nq=4, noiseless):

| N | Equivariant | Non-equiv |
|---|---|---|
| 20 | **0.027** | 0.108 |
| 100 | **0.040** | 0.069 |
| 200 | 0.037 | 0.045 |
| 1000 | 0.031 | 0.031 |

**Verdict: clean win for the symmetry constraint.** At small N the equivariant model
overfits dramatically less (4× smaller gap at N=20). This is the mechanism behind the
learning-curve advantage: the symmetry-restricted hypothesis class generalizes from less
data. The gap converges as N grows, exactly as expected.

## 3. Qubit frugality  ·  `figures/qubit_frugality.png`

Test AUC vs working qubits (N=200, noiseless):

| n_qubits | Equivariant | Non-equiv |
|---|---|---|
| 2 | **0.825** | 0.804 |
| 4 | **0.838** | 0.820 |
| 6 | 0.835 | 0.844 |
| 8 | 0.838 | 0.838 |

**Verdict: partial.** The equivariant model is genuinely qubit-frugal — it holds **0.825
AUC at just 2 qubits** and leads at nq=2–4. But the non-equivariant control does *not*
collapse at low qubit counts (0.804 at nq=2); it catches up by nq=6. So the "frugal"
half of the claim holds; the "control collapses" half does not.

## 4. Noise robustness  ·  `figures/noise_robustness.png`

Test AUC vs noise strength p (N=200, nq=4):

- **iid (amplitude damping):** both models stay flat (~0.83–0.84) up to p=0.35 — incoherent
  single-qubit noise is benign here (consistent with the known near-invariance of ROC-AUC
  to uniform channel contraction).
- **Correlated (OU-correlated coherent RX):**

| p | Equivariant | Non-equiv |
|---|---|---|
| 0.00 | 0.838 ± 0.012 | 0.820 ± 0.029 |
| 0.05 | 0.844 | 0.828 |
| 0.10 | 0.834 | 0.817 |
| 0.20 | **0.774 ± 0.081** | 0.715 ± 0.162 |

**Verdict: shield holds for correlated noise.** Under temporally+spatially correlated
coherent noise the equivariant model degrades more gracefully (drop of 0.064 vs 0.105 from
p=0→0.2), and the advantage *widens* with noise (0.018 → 0.059). Note also the
non-equivariant model's variance explodes at high p (std 0.162 vs 0.081): it is not just
lower but far less *reliable* under heavy correlated noise. iid noise — the "easy mode" —
separates the models very little.

## 5. Shield surface  ·  `figures/shield_surface.png`

2-D heatmap of (equivariant − non-equivariant) test AUC over `n_qubits × correlated-p`
(N=200). The equivariant advantage (red) is **concentrated in the doubly-stressed corner:
few qubits (2–4) AND high correlated noise (p=0.2)** — precisely where the triple-shield
argument predicts symmetry should matter most. At 6 qubits the advantage vanishes/reverses.

## 6. Trainability — barren-plateau probe  ·  `figures/gradient_variance.png`

Variance of ∂L/∂θ₀ over random initializations vs qubit count (log scale):

| n_qubits | Equivariant (local cost) | Generic (global cost) |
|---|---|---|
| 2 | 2.4e-5 | 1.5e-4 |
| 8 | 2.3e-6 | 1.4e-6 |
| **2→8 decay** | **~10×** | **~107×** |

**Verdict: clean Cerezo-style contrast.** The deep generic ansatz with a global cost shows
~10× steeper gradient-variance decay (barren-plateau onset) than the shallow, local-cost
equivariant model, whose variance stays comparatively flat — evidence that the structured,
local construction is more trainable as the register grows. (The non-equivariant A1 shares
the equivariant ansatz and behaves like it, as expected.)

---

## Overall verdict

The "triple shield" **partially holds — and the nuance is the result, not a failure**:

- **Strongest:** data efficiency / generalization (Figs 1–2) and **correlated-noise
  robustness** (Figs 4–5), both vs a rigorously budget-matched control.
- **Weakest:** the qubit-frugality "collapse" claim — equivariance helps at low qubits but
  the control does not break down.
- **Trainability** (Fig 6) independently favors the structured construction.

The honest framing for an application: *a permutation-equivariant inductive bias delivers
measurable, controlled advantages in the data-scarce and correlated-noise regimes at equal
parameter budget, while a competent classical baseline remains a strong ceiling.*

## Limitations & honest notes

- iid noise required the density-matrix simulator (`default.mixed`, cost ~4ⁿ), so the iid
  axis is studied only at nq≤4; high-qubit noise uses correlated (statevector) sims.
- Small-N points (N=20) have wide error bands (high variance is intrinsic to the regime).
- The equivariant vs non-equivariant comparison is at exactly equal trainable-parameter
  count (61); A1 breaks symmetry via input positional encoding only.
- All results are noiseless/noisy **simulation** (no hardware), gluon-vs-top only.
- Reproduce: `python experiments/sweep.py --grid full --workers 9 && python experiments/run_diagnostics.py && python analysis/plots.py`.
