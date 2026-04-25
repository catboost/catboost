# S36-LATENT-P11 T0a — Math Derivation: Hessian-vs-SampleWeight Semantics

**Mode:** Pragmatic
**Date:** 2026-04-25
**Branch:** `mlx/sprint-36-hess-vs-weight` (no commits yet)
**Cuts from:** master `3bc02bc3cb`
**Authors:** mathematician
**Scope:** code-grounded derivation; no source files modified

---

## 0. TL;DR

- **Verdict: variable swap, not architectural.** CPU and MLX use **mathematically identical score-function formulas**: both compute `score_per_leaf = (Σ g)² / (W + λ)`. They differ in *what* the codebase plugs into `W`. CPU plugs the per-document **sample weight** (`Σ w_i`); MLX plugs the per-document **hessian** (`Σ h_i · w_i`). For RMSE/MAE/Quantile/Huber the two coincide because `h_i ≡ 1`; for Logloss/Poisson/Tweedie/Multiclass they do not.
- **Confidence: high.** The CPU side is unambiguous (`scoring.cpp:217` populates `SumWeight += sampleWeights[doc]`; `score_calcers.cpp:20-33` and `short_vector_ops.h:67-81` consume `SumWeight` as the score denominator). The MLX side is also unambiguous (`csv_train.cpp:1801` populates the histogram weight channel from `dimHess[k][doc]`; `csv_train.cpp:1894-1897, 1979-1984, 1997-2014` consume that channel as `weightLeft/weightRight` in the gain formula).
- **Predicted Logloss iter=50 drift magnitude: O(10–30 %)** — same order as DEC-036 (52.6 % was Cosine; L2 should be smaller but still well outside 4-gate parity). At iter=1 with depth=0 the drift is **zero by construction**. At iter=1 depth>0 the drift is small (≤ a few percent) because σ(0) = ½ ⇒ h_i ≈ 0.25 uniformly. The drift compounds across iterations as the hessian field becomes inhomogeneous.
- **Recommended next step: run T1 empirical probe before fixing.** The math closes; what the math cannot tell us is whether MLX's "wrong-but-stable" formula has an *implicit-bias* benefit on real benchmarks. A single Logloss-on-Adult or Logloss-on-Higgs run with both formulas will tell us in minutes.

---

## 1. Setup — what the "weight slot" means in each codebase

### 1.1. CPU (CatBoost)

Per-bucket statistic struct:

```cpp
// catboost/private/libs/algo/calc_score_cache.h (TBucketStats)
struct TBucketStats {
    double SumWeightedDelta;   // Σ (w_i · weight_i · der1_i)   over docs in bucket  ("plain mode")
    double SumWeight;          // Σ (w_i · weight_i)             over docs in bucket  ("plain mode")
    double SumDelta;           // Σ (weight_i · der1_i)          over docs in bucket  ("ordered mode")
    double Count;              // bucket doc count               over docs in bucket  ("ordered mode")
};
```

Notation: `der1_i` is the *negative* gradient of the loss at sample `i` (CatBoost convention — see § 2.1). `weight_i` is the user-provided learn weight (1 by default). `w_i` is the bootstrap weight (1 with no bootstrap; Bernoulli ∈ {0,1}; Bayesian Exp(1/T); MVS [0,1]).

Population site (`catboost/private/libs/algo/scoring.cpp:213-218`, `UpdateWeighted` lambda body):

```cpp
for (int doc : docIndexRange.Iter()) {
    auto& leafStats0 = stats[indexer.GetIndex<isOneNode>(doc, quantizedValues)];
    leafStats0.SumWeightedDelta += weightedDer[doc];   // weightedDer[d] = w_d · weight_d · der1_d
    leafStats0.SumWeight        += sampleWeights[doc]; // sampleWeights[d] = w_d · weight_d
}
```

The arrays come from `tensor_search_helpers.cpp:467-484`:

- `bt.SampleWeightedDerivatives[d] = bt.WeightedDerivatives[d] · sampleWeights[d]` (line 472),
- `sampleWeights[d] *= learnWeights[d]` after Bootstrap (line 482).

So `sampleWeights[d]` is **exactly** the product `w_d · weight_d` — no hessian component. `SumWeight` in the score function is the sum of *sample weights only*.

### 1.2. MLX

Per-(k, partition, bin) histogram channel layout (`csv_train.cpp:1796-1801`):

```cpp
const float* hd = perDimHist[k].data() + p * 2 * totalBinFeatures;
size_t base = (k * numPartitions + p) * stride;
for (int b = static_cast<int>(folds) - 1; b >= 0; --b) {
    suffGrad[base + b] = suffGrad[base + b + 1] + hd[feat.FirstFoldIndex + b];                        // grad channel
    suffHess[base + b] = suffHess[base + b + 1] + hd[totalBinFeatures + feat.FirstFoldIndex + b];      // weight channel
}
```

The variable `suffHess` is named "hess" because it actually holds the hessian sum. The histogram kernel (`kernel_sources.h`, dispatched at `csv_train.cpp:1482`) consumes two payload arrays per dimension:

```cpp
// csv_train.cpp:4070-4097 — RMSE / Logloss derivative blocks
dimGrads[0] = ...;   // negative gradient (or pos gradient — see § 2.2)
dimHess[0]  = ones / σ(1−σ) / exp(approx) / ...;   // hessian for the loss
```

After bootstrap and sample-weighting, **both** arrays are multiplied by per-doc sample weight (`csv_train.cpp:4168-4169`, `4236-4237`):

```cpp
dimGrads[k] = mx::multiply(dimGrads[k], sampleWeightsArr);
dimHess[k]  = mx::multiply(dimHess[k],  sampleWeightsArr);
```

Hence MLX's "weight channel" stores **`Σ over docs in bin of (h_d · w_d · weight_d)`** — i.e., **hessian × sample_weight**, not sample_weight alone.

### 1.3. Side-by-side at the stat-slot level

| Slot | CPU `SumWeight` | MLX `weightLeft / weightRight` |
|------|-----------------|-------------------------------|
| Per-doc contribution | `w_d · weight_d` | `h_d · w_d · weight_d` |
| Per-bucket value     | `Σ (w_d · weight_d)` | `Σ (h_d · w_d · weight_d)` |
| Multiplicative gap   | 1 | `h_d` (per-doc hessian factor inside the sum) |

For RMSE/MAE/Quantile/Huber: `h_d ≡ 1` ⇒ slots are equal. For Logloss/Poisson/Tweedie/Multiclass: `h_d` is data-dependent, and the slots diverge.

---

## 2. CPU's gain formula — explicit per-leaf form

### 2.1. Score function — `TL2ScoreCalcer::AddLeafPlain`

`catboost/private/libs/algo/score_calcers.cpp:20-33`:

```cpp
void TL2ScoreCalcer::AddLeafPlain(int splitIdx,
                                  const TBucketStats& leftStats,
                                  const TBucketStats& rightStats) {
    const double rightAvrg = CalcAverage(rightStats.SumWeightedDelta,
                                         rightStats.SumWeight,
                                         L2Regularizer);
    const double leftAvrg  = CalcAverage(leftStats.SumWeightedDelta,
                                         leftStats.SumWeight,
                                         L2Regularizer);
    AddLeaf(splitIdx, rightAvrg, rightStats);
    AddLeaf(splitIdx, leftAvrg,  leftStats);
}
```

with (`online_predictor.h:112-119`)

```cpp
inline double CalcAverage(double sumDelta, double count, double scaledL2Regularizer) {
    double inv = count > 0 ? 1. / (count + scaledL2Regularizer) : 0;
    return sumDelta * inv;
}
```

and (`score_calcers.h:87-89`)

```cpp
void AddLeaf(int splitIdx, double leafApprox, const TBucketStats& leafStats) override {
    Scores[splitIdx] += leafApprox * leafStats.SumWeightedDelta;
}
```

Letting `S_L = SumWeightedDelta_L`, `W_L = SumWeight_L` (and similarly for the right side), with `λ_s := L2Regularizer · (sumAllWeights / allDocCount)` the scaled regulariser:

$$
\boxed{\text{gain}_{\text{CPU,L2}} \;=\; \frac{S_L^2}{W_L + \lambda_s} \cdot \mathbb{1}[W_L>0] \;+\; \frac{S_R^2}{W_R + \lambda_s} \cdot \mathbb{1}[W_R>0]}
$$

(The `mask · 1/(W+λ)` form in `CalcAverage` returns 0 when `W=0`, so the contribution from an empty side vanishes — cf. DEC-042 per-side mask.)

### 2.2. Cosine

`TCosineScoreCalcer::AddLeaf` (`score_calcers.h:63-66`) and `UpdateScoreBinKernelPlain` (`short_vector_ops.h:67-81`) accumulate two state variables per (feat, bin):

$$
\text{num}_{\text{CPU}} = \sum_{\ell \in \{L,R\}} \frac{S_\ell^2}{W_\ell + \lambda_s},
\qquad
\text{den}_{\text{CPU}} = \sum_{\ell \in \{L,R\}} \frac{S_\ell^2 \, W_\ell}{(W_\ell + \lambda_s)^2}.
$$

(`SumWeightedDelta` appears in both the average and the dotProduct summand; `SumWeight` appears in both the denominator and the d2-summand.) The reported score is `num / sqrt(den)` (`score_calcers.h:55-61`). For our purposes the structure is identical to L2 with respect to the question of *what `W_\ell` denotes*.

### 2.3. The semantic content of `S_\ell` and `W_\ell` in CPU

- `WeightedDerivatives[d] = weight_d · der1_d` populated by `IDerCalcer::CalcDersRangeImpl` (`error_functions.cpp:46-60`): the per-doc loop multiplies `der1_d` by `weight_d` *if* a weights pointer is supplied. `der1_d` is the per-loss first derivative as defined by the loss class (e.g. RMSE: `target - approx`; CrossEntropy/Logloss: `target - σ(approx)`; Poisson: `target - exp(approx)`).
- `SampleWeightedDerivatives[d] = WeightedDerivatives[d] · w_d`. **The hessian never enters this product** — only the first derivative and the user/bootstrap weights.
- `SampleWeights[d] = w_d · weight_d`. Pure sample weight; no hessian.

So in CPU, for a leaf `\ell`:

$$
S_\ell = \sum_{d \in \ell} w_d \, \text{weight}_d \, \tilde g_d, \qquad
W_\ell = \sum_{d \in \ell} w_d \, \text{weight}_d,
$$

where `\tilde g_d := der1_d` is the (sign-conventional) per-doc first derivative. **`W_\ell` does not depend on the loss function's hessian.**

### 2.4. Leaf-value step — important contrast

`CalcDeltaNewton(ss, ...)` (`online_predictor.h:172-179`) reads:

```cpp
return CalcDeltaNewtonBody(ss.SumDer, ss.SumDer2, l2Regularizer, sumAllWeights, allDocCount);
// sumDer / (-sumDer2 + λ_s)
```

`SumDer2` is populated via `AddDerDer2` from per-doc second derivatives. The leaf-value Newton step uses the **hessian** in the denominator, while the **score function** uses the **sample weight**. CPU has architecturally separated the two roles. This is the asymmetry MLX has flattened.

---

## 3. MLX's gain formula — explicit per-(p,k) form

Ordinal branch, `csv_train.cpp:1889-1985` (the L2 case is at lines 1965-1985; the Cosine case mirrors it at 1987-2053). For each partition `p`, dimension `k`, and bin offset `b`:

```cpp
float totalSum    = perDimPartStats[k][p].Sum;       // Σ over docs in p of g̃_d
float totalWeight = perDimPartStats[k][p].Weight;    // Σ over docs in p of h̃_d
size_t base = (k * numPartitions + p) * stride;
float sumRight    = suffGrad[base + bin];            // Σ over docs in p with bin > b of g̃_d
float weightRight = suffHess[base + bin];            // Σ over docs in p with bin > b of h̃_d
float sumLeft     = totalSum    - sumRight;
float weightLeft  = totalWeight - weightRight;
```

where `g̃_d := dimGrads[k][d]` after sample-weight + bootstrap multiplication, and `h̃_d := dimHess[k][d]` after the same. From § 1.2:

$$
g̃_d = w_d \, \text{weight}_d \, g_d, \qquad
h̃_d = w_d \, \text{weight}_d \, h_d.
$$

(MLX does not track sign convention separately — see § 4 caveat — but the squared form makes the sign irrelevant for L2 score and Cosine score.)

L2 case (`csv_train.cpp:1965-1985`):

$$
\boxed{\text{gain}_{\text{MLX,L2}} = \frac{S_L^2}{W_L^{\text{MLX}} + \lambda} \mathbb{1}[W_L^{\text{MLX}}>\epsilon] + \frac{S_R^2}{W_R^{\text{MLX}} + \lambda} \mathbb{1}[W_R^{\text{MLX}}>\epsilon] - \frac{(S_L+S_R)^2}{W_L^{\text{MLX}} + W_R^{\text{MLX}} + \lambda}}
$$

(MLX subtracts a parent-leaf term — line 1984; CPU's score does not because gain is computed as a *change* relative to the parent only at the optimizer level. This is a separate point and has been DEC-042-style validated.)

Cosine case (`csv_train.cpp:1997-2014`) is structurally identical with the same `W_\ell^{\text{MLX}}` substitution.

The structural difference vs. CPU is **one substitution**:

$$
W_\ell^{\text{CPU}} = \sum_{d \in \ell} w_d \, \text{weight}_d
\qquad \xrightarrow{\text{MLX bug}} \qquad
W_\ell^{\text{MLX}} = \sum_{d \in \ell} w_d \, \text{weight}_d \, h_d.
$$

`S_\ell` is the same in both codebases (numerator unaffected). `λ` differs slightly (`l2RegLambda` raw vs `λ_s = λ · sumAllWeights/allDocCount` in CPU) — this is a separate, already-classified divergence carry-forward (see § 8).

---

## 4. RMSE invariance — algebraic confirmation

For RMSE: `h_d = 1` for all `d` (`error_functions.h:381` `RMSE_DER2 = -1.0`; CatBoost negates per its hessian sign convention so the *magnitude* is 1). MLX assigns `dimHess[0] = ones` directly (`csv_train.cpp:4070`). Hence:

$$
W_\ell^{\text{MLX}}\big|_{h \equiv 1} = \sum_{d \in \ell} w_d \, \text{weight}_d \cdot 1 = \sum_{d \in \ell} w_d \, \text{weight}_d = W_\ell^{\text{CPU}}.
$$

Therefore `gain_{MLX,L2} - parent_term = gain_{CPU,L2}` whenever `h_d ≡ 1` and the λ-scaling difference is absorbed (see § 8). MLX and CPU **agree by accident** for RMSE — the bug exists, but it is silent.

The same holds for MAE, Quantile (because their hessians are also 1, line `csv_train.cpp:4074, 4080`), and Huber on the dominant branch (line 4088: `dimHess = 1` for `|residual| ≤ δ`; on the tail branch MLX uses 1e-6 while a true Huber hessian is 0 in the gradient region — small additional divergence, masked by the 1e-6 floor).

---

## 5. Logloss derivation — explicit divergence term

Loss: `ℓ(a, y) = -y log σ(a) - (1-y) log(1-σ(a))`. Then

$$
g_d = \sigma(a_d) - y_d, \qquad h_d = \sigma(a_d)\,(1 - \sigma(a_d)) \in (0, 1/4].
$$

MLX (`csv_train.cpp:4090-4097`) computes `dimGrads[0] = σ(a) - target` and `dimHess[0] = max(σ(1-σ), 1e-16)` — sign convention here matches positive-gradient (note this contradicts the CatBoost CPU sign convention; not relevant for the L2/Cosine score, which sees `S_\ell^2` only).

For unit sample weights (`w_d = weight_d = 1`), the per-leaf denominators are:

$$
W_\ell^{\text{CPU}} = |\ell|, \qquad W_\ell^{\text{MLX}} = \sum_{d \in \ell} \sigma(a_d)\,(1-\sigma(a_d)).
$$

Define the per-leaf *mean hessian* `\bar h_\ell := |ℓ|^{-1} \sum_{d\in\ell} h_d ∈ (0, 1/4]`. Then `W_\ell^{\text{MLX}} = \bar h_\ell · W_\ell^{\text{CPU}}`.

CPU vs MLX gain at a single split (ignoring the λ-scaling difference for clarity):

$$
\text{gain}_{\text{CPU,L2}} = \frac{S_L^2}{|L| + \lambda_s} + \frac{S_R^2}{|R| + \lambda_s},
$$

$$
\text{gain}_{\text{MLX,L2}} = \frac{S_L^2}{\bar h_L |L| + \lambda} + \frac{S_R^2}{\bar h_R |R| + \lambda} - \text{parent-term}.
$$

### 5.1. The divergence term

Consider two candidate splits A and B at a single node, both with the same `S_L, S_R, |L|, |R|`. CPU's gain ranks them identically. **MLX's ranking depends on `(\bar h_L, \bar h_R)`** for each candidate — i.e., on the *distribution* of `σ(a_d)(1-σ(a_d))` within each side. Splits that happen to produce sides with low `\bar h` (predictions confidently saturated, near 0 or 1) get *amplified* gain (small denominator), while splits with high `\bar h` (predictions near 0.5, where the model is least confident) get *suppressed* gain.

This is **not** a bias-preserving transformation: it is an implicit prior in MLX's split-ranking that says "prefer splits whose children contain confidently-predicted points." That is the opposite of what gradient boosting wants — boosting *should* prefer to split where the model is uncertain, i.e., where the gradients are large and informative.

### 5.2. Why iter=1 depth>0 drift is small but nonzero

At iter=1, all `a_d = a_0` (the single starting approx, typically the global mean logit). Hence `\sigma(a_d)(1-\sigma(a_d))` is **constant in `d`** ⇒ `\bar h_L = \bar h_R = h_0`. Then

$$
\frac{S_L^2}{h_0 |L| + \lambda} \neq \frac{S_L^2}{|L| + \lambda},
$$

so the *value* of `gain_{MLX}` differs from `gain_{CPU}` by an O(1) factor. **But the argmax over candidate splits is preserved at iter=1** because the multiplicative `h_0`-rescaling of the denominator commutes with the argmax (apart from the additive λ, which is the same for every split candidate). Concretely, ranking A vs B at iter=1 only depends on the relative magnitudes `S²/W`, which are scaled identically across candidates.

Caveat: the *additive* λ can mildly perturb the argmax, but for default λ=3 vs `|L|+|R| ≈ N_train ≫ 1` the perturbation is < 0.1 % per split. So **drift at iter=1 ≈ 0** (split tree is ε-identical), with model-output drift kicking in via the leaf-value step and accumulating from iter=2 onward as `\sigma(a_d)` becomes inhomogeneous.

### 5.3. Predicted iter=50 magnitude

At iter=50 with default lr=0.03, predictions span a non-trivial range; `\bar h_L` and `\bar h_R` are no longer constant across leaves or across split candidates. The `(\bar h_L, \bar h_R)` term begins to **flip orderings** between split candidates that CPU would rank close. Empirically, for the analogous DEC-036 phenomenon (also a denominator-asymmetry bug, also Cosine), iter=50 drift was 52.6 % in the affected ST+Cosine config. P11 is structurally similar (a denominator misuse) but L2-on-Logloss should be *somewhat smaller* because:

- DEC-036's per-side mask asymmetry could create whole-partition skips (qualitative effect).
- P11's hess-vs-weight is a *smooth* multiplicative perturbation of the denominator (quantitative effect).

A reasonable point estimate: **iter=50 drift in [10 %, 30 %]** for Logloss with default config. Could be lower (≤ 5 %) on small datasets where λ's additive contribution dominates `\bar h · |ℓ|`. Could be higher (≥ 50 %) on datasets where σ saturates rapidly. **Empirical T1 measurement is needed to resolve this range.**

### 5.4. Configurations where drift is exactly zero

- `depth = 0` at any iteration (no split ever happens; only leaf-value step runs, which is correct).
- Any iter where the argmax is preserved (most likely iter=1 on standard configs; not iter ≥ 2 in general).
- Single-document-per-leaf splits where `\bar h_\ell = h_{d^*}` is a single value (still affects gain *value*, but not necessarily *rank*).

---

## 6. Poisson, Tweedie, Multiclass — same shape, different magnitudes

### 6.1. Poisson

`g_d = \exp(a_d) - y_d`, `h_d = \exp(a_d)`. MLX (`csv_train.cpp:4098-4102`) sets `dimHess = max(exp(a), 1e-6)`. Unlike Logloss, `h_d` is **unbounded above** — for high counts, `\bar h_\ell` can be O(10²–10³). The MLX denominator then becomes O(10²–10³) larger than the CPU denominator, dramatically suppressing gain.

But again, at iter=1 with `a_0 = log(\bar y)`, `\bar h_\ell = \bar y` is constant across leaves and the argmax is preserved up to λ-perturbation. **Drift kicks in at iter ≥ 2.** Magnitude is expected to be **larger than Logloss drift** because `h_d` varies more aggressively with `a_d` (Poisson hessian has range up to `e^{a_{\max}}`).

### 6.2. Tweedie

Hessian (`csv_train.cpp:4109-4114`):

$$
h_d = -y_d \, (1-p) \, e^{(1-p) a_d} + (2-p) \, e^{(2-p) a_d}.
$$

For `p ∈ (1, 2)` (typical use), both terms are positive when `y_d ≥ 0`, and `h_d` is approximately quadratic in `\exp(a_d)`. Behaviour matches Poisson qualitatively but with a `p`-dependent magnitude. **No structural difference** from the Logloss/Poisson story.

### 6.3. Multiclass

For class-`k` softmax cross-entropy, `g_{k,d} = p_{k,d} - \mathbb{1}[y_d = k]` and `h_{k,d} = p_{k,d}(1 - p_{k,d}) ∈ (0, 1/4]`. MLX (`csv_train.cpp:4140-4159`) tracks `K` separate `(grad, hess)` channels and the score function sums per-class contributions over `k = 0..K-1` (`csv_train.cpp:1889`). The bug therefore replicates **independently per class**, with each class's score using `\bar h_{k,\ell}` instead of `|\ell|`.

A subtle multi-class twist: if class `k_0` has very imbalanced incidence (e.g. 1 % positive), `p_{k_0, d}` is uniformly small, so `h_{k_0, d} ≈ p_{k_0, d}` is small. CPU's gain treats all classes' `|\ell|` identically; MLX's gain *down-weights* rare classes' contribution to the joint score. This is a **systemic bias against rare-class signal**, not just a numerical drift. For balanced multi-class problems the effect is comparable to Logloss; for imbalanced it is plausibly larger.

### 6.4. The other affected losses

By the same construction, every loss for which `h_d \not\equiv 1` exhibits the bug:

| Loss family | `h_d` | Bug active? |
|-------------|-------|-------------|
| RMSE / MultiRMSE | 1 (or weight·1) | No — `\bar h_\ell ≡ 1` |
| MAE / Quantile | 1 | No — same reason |
| Huber (small-residual branch) | 1 | No |
| Huber (large-residual branch) | 1e-6 (MLX clamp; true is 0) | Yes (small magnitude) |
| LogCosh | sech²(a−y) ∈ (0,1] | Yes |
| Logloss / CrossEntropy | σ(1−σ) ∈ (0,1/4] | Yes |
| Poisson | exp(a) | Yes (large magnitude) |
| Tweedie | as above | Yes |
| MultiClass (softmax) | p(1−p) per class | Yes |
| MAPE | 1/|y| | Yes |
| RMSPE | varies | Yes |
| pairlogit / yetirank | scattered Hessian, varies | Yes |

So roughly **half of CatBoost's loss zoo** is affected, corresponding to the half that has a non-trivial Newton step.

---

## 7. Verdict

**Same role, swapped values.** The score function in both codebases is the textbook "gain = Σ_ℓ S_ℓ²/(W_ℓ + λ)" — they are computing the same expression, with the same parametric form. The semantic role of `W_ℓ` *should be* the *sum of sample weights in the leaf*, because the score formula is derived from the L2 leaf-value Newton step **specialized to the case `h_d ≡ 1`** (see Friedman 2001 §3 — for L2 loss the Hessian is constant and the formula reduces to gradient-sum-squared over leaf-size). For L2 loss the two coincide; the score formula was canonized for that case and CatBoost reuses it across all loss functions, with `SumWeight` understood to mean "leaf size" in the weighted sense.

MLX's substitution of `Σ h_d` for `Σ w_d` is a **variable swap** (option 1 in the prompt), not a different mathematical formulation. The fix changes which array is plugged into `weightLeft / weightRight`; it does not change the gain formula's algebraic shape.

**Confidence: high** (≥ 0.95). The grep evidence is unambiguous and the math is elementary. The remaining ≤ 5 % uncertainty covers the question of whether CatBoost upstream **historically** intended `Σ w_d` (sample weight) or `Σ h_d` (hessian) as the score denominator — i.e., whether MLX's choice has any pedigree in older CatBoost or in the Russian-language internal docs. Even if upstream's *original* intent was `Σ h_d`, the *current* code irrefutably uses `Σ w_d`, and bit-exact CPU parity is the binding constraint for catboost-mlx.

---

## 8. Recommended fix shape

### 8.1. Variable swap — minimal, isolated change

Add a third channel to the histogram payload: `sampleWeightChannel[bin] = Σ_{d ∈ bin} w_d · weight_d`. Route this channel into `weightLeft / weightRight` in FindBestSplit. Keep `dimHess` for the leaf-value Newton step at `csv_train.cpp:5036-5049`.

**Concrete file:line edits (do not apply yet — T0a is analysis-only):**

1. **Histogram producer** (`csv_train.cpp` ~line 4376-4387 and similar): the histogram kernel currently takes `(grads, hess)` for each dimension. Add a parallel scatter for `sampleWeightsArr` (or `ones` when no sample weights) into a third bin-aligned buffer.
2. **Histogram layout** (`csv_train.cpp:1796-1801`): widen `2 * totalBinFeatures` to `3 * totalBinFeatures` and populate a third `suffSampleWeight[base + b]` buffer alongside `suffGrad` and `suffHess`. This is a structural change to the histogram tensor's last dimension; downstream consumers must be updated.
3. **FindBestSplit consumers** (`csv_train.cpp:1689-1696, 1894-1897, 2301-2304, 2385-2388`): replace `weightRight = suffHess[...]` with `weightRight = suffSampleWeight[...]`. Keep `suffHess` available because `perDimPartStats[k][p].Weight` is used for monotone-constraint pre-checks (`csv_train.cpp:1872-1886`) — that check **is** a Newton-step approximation and **should** keep using hessian.
4. **Histogram Metal kernel** (`kernel_sources.h`): widen the per-thread atomic-add fan-out from 2 channels to 3. This is a kernel-signature change. **Kernel md5 invariant `9edaef45b99b9db3e2717da93800e76f` will change** — orchestrator must whitelist the new md5 when this fix lands. (DEC-012 atomic remains intact: each of the three channels is atomically updated independently.)
5. **Reference paths** (`csv_train.cpp` reference histogram CPU path used for parity testing): update to match the 3-channel layout.

**Estimated diff size:** ~150 lines across `csv_train.cpp` + `kernel_sources.h` + the histogram CPU reference. Bounded, mechanical, no new mathematical content.

### 8.2. Why not the other framings

- **Architectural change (separate hess + sample-weight throughout):** that's exactly what § 8.1 describes (one new channel). "Architectural" is the correct adjective only if you frame the histogram tensor's channel count as architecture; we'd call it a layout change.
- **Score-formula change (use h instead of w in CPU's formula):** would require changing CatBoost's CPU code, which is forbidden by project rules ("Do NOT modify the original CatBoost CPU/CUDA code"). Also, the CPU formula is the canonical one in the gradient boosting literature for the L2-loss-special-case interpretation, so it would be mathematically wrong to flip CPU.
- **No-fix-needed:** ruled out by the algebra in §§ 5.1, 6.1–6.3. The two formulas are not equivalent up to argmax for inhomogeneous-hessian losses.

### 8.3. Numerical / DEC-012 considerations

- The new sample-weight channel sums a non-negative quantity (`w_d · weight_d ∈ [0, ∞)`). At fp32 it is well-behaved up to `N · max(w·weight)` ≤ 2²⁴ before precision loss; for typical `N ≤ 10⁶` and `w·weight ≤ 1`, fp32 is ample. No K4-style Kahan widening expected to be needed.
- DEC-012 atomic (per-bin atomic add in the Metal histogram kernel) extends naturally: the third channel is a third independent atomic add on the same `[bin]` index, so atomicity is preserved trivially.
- Bit-exact 4-gate parity (G4a/G4b/G4d) gates that currently pass on RMSE will continue to pass on RMSE because § 4 shows RMSE is invariant. New parity gates must be added for Logloss/Poisson — see § 10.

---

## 9. Predicted empirical signature (for T1)

If T0b code reading confirms this analysis, then T1 (empirical probe) should observe:

| Iter | Loss = RMSE | Loss = Logloss | Loss = Poisson |
|------|-------------|----------------|----------------|
| 1 (depth=0) | drift = 0 | drift = 0 | drift = 0 |
| 1 (depth=6) | drift ≤ 1e-6 | drift ≤ 1 % (argmax-stable) | drift ≤ 1 % |
| 10 | drift ≤ 1e-6 | drift ~ 1–5 % | drift ~ 5–15 % |
| 50 | drift ≤ 1e-6 | drift ~ 10–30 % | drift ~ 20–50 % |
| 100 | drift ≤ 1e-6 | drift ~ 15–40 % | drift ~ 30–70 % |

Drift = `|MLX_logloss − CPU_logloss| / CPU_logloss` on a held-out split, default config (lr=0.03, depth=6, λ=3, no bootstrap, sampleWeights=1).

**Falsification criterion for T1:** if Logloss iter=50 drift ≤ 1 % on three different datasets (Adult, Higgs-100k, Click-Prediction), then either (a) this analysis missed a downstream cancellation, or (b) MLX's bug is empirically masked by some other counteracting force — both worth investigating before shipping a fix.

---

## 10. Recommended next step

**Run T1 empirical probe before committing the fix.**

Why not close on math alone:

1. **Implicit bias.** The bug, while mathematically wrong, may be empirically *favourable* on some data distributions. Gradient boosting's success rests partly on the implicit regularization of the optimizer; an "incorrect" denominator that systematically prefers high-`\bar h_\ell` splits could, in principle, act as a regularizer akin to early stopping. The math cannot rule this out.
2. **Magnitude verification.** Whether the iter=50 drift is 5 % or 50 % matters for severity classification (latent vs critical) and for sprint scope (single-iteration L4 fix vs multi-sprint refactor). T1 settles this in <1 hour.
3. **Affected-loss enumeration.** Some non-trivial-hessian losses (e.g. Huber tail with `1e-6` clamp, MAPE) may show drift at noise level; T1 measures empirically rather than relying on dimensional analysis.

Suggested T1 spec:
- Datasets: Adult (binary classification, ~32 k rows), Higgs-100k (binary, 100 k rows), Click-Prediction (binary, ~150 k rows after subset).
- Losses: Logloss, Poisson (on a count target), MultiClass (using a 3-class subset of Higgs).
- Metric: Logloss (or NLL) on held-out 20 % split, MLX vs CPU CatBoost, default config.
- Sweep: iter ∈ {1, 10, 50, 100, 500}.
- Output: drift table; if any cell shows > 1 % drift on Logloss, P11 is confirmed and the variable-swap fix is justified.

After T1 confirms (or surprises) the math, T2 = orchestrator-led variable-swap fix per § 8.1.

---

## 11. Open questions (cannot resolve from code reading alone)

1. **Upstream historical intent.** Did CatBoost upstream *ever* use `Σ h_d` for the score denominator, or has `SumWeight` always meant sample weight? Answering would inform whether MLX's choice was an inherited convention or an independent error. Likely answerable via `git log -- score_calcers.cpp scoring.cpp` on the upstream repo.
2. **Pairwise / Querywise losses.** YetiRank and PairLogit scatter pairwise hessians (`csv_train.cpp:2670-2690`). The *pairwise hessian* there is an artifact of pair-construction, not a per-doc Newton hessian. Whether the variable-swap fix changes pairwise behaviour at all needs a closer reading of `algo/yetirank.cpp` (referenced but not surveyed here).
3. **MVS bootstrap interaction.** With MVS sampling the per-doc weight `w_d` is gradient-magnitude-dependent (`csv_train.cpp:4189-4231`). The fix must keep `w_d` in the new sample-weight channel — but MVS multiplies BOTH `dimGrads[k]` and `dimHess[k]` by the same `w_d`. The variable swap correctly splits "MVS weight" out of "hessian"; T1 should cover MVS-on to confirm.
4. **Sample-weight scaling for λ.** CPU uses `λ_s = λ · sumAllWeights / allDocCount` (`online_predictor.h:121-127`); MLX uses `l2RegLambda` raw (no scaling). This is a *separate* divergence not covered by P11 fix. After the P11 fix, the residual `λ`-scaling difference will surface and may require its own L4. (Currently masked by RMSE on equal-weight benchmarks where `sumAllWeights / allDocCount = 1`.)
5. **DEC-042 interaction.** DEC-042 added per-side mask in MLX. With the current MLX, `weightLeft / weightRight = Σ h_d`; the per-side mask checks `weight > 1e-15`. After fix, the mask checks `Σ w_d > 1e-15`. Semantics of "empty side" become *strictly* "no documents present" rather than "no hessian mass present" — the latter could also fire when all docs in a side have `h_d ≈ 0` (saturated Logloss). This is the **correct** mask semantics, but it changes which splits are skipped in pathological cases. Empirically should be a tightening (fewer skips), but worth flagging in the fix PR.

---

## 12. Summary table

| Question | Answer |
|----------|--------|
| Same role / different roles? | **Same role, swapped values** |
| Confidence | High (≥ 0.95) |
| Affected losses | All non-RMSE-family (Logloss, Poisson, Tweedie, MultiClass, MAPE, LogCosh, RMSPE, pairlogit, yetirank, Huber tail) |
| Predicted Logloss iter=50 drift | 10–30 % (range; T1 needed to pin) |
| Predicted Poisson iter=50 drift | 20–50 % |
| Configs with zero drift | RMSE/MAE/Quantile (any iter), depth=0 (any iter), iter=1 with constant initial approx |
| Fix shape | Variable swap: add 3rd histogram channel (sample-weight); reroute `weightLeft/Right` |
| Fix complexity | ~150 lines, kernel-md5-changing, structurally bounded |
| Recommended next step | **T1 empirical probe** before fix to verify magnitude and rule out implicit-bias benefit |

---

## Appendix A. Key code citations

| Fact | File:line |
|------|-----------|
| CPU populates `SumWeight` from sample weights | `catboost/private/libs/algo/scoring.cpp:217` |
| `sampleWeights[d]` = bootstrap × learnWeight (no hessian) | `catboost/private/libs/algo/tensor_search_helpers.cpp:467-484` |
| CPU L2 score formula | `catboost/private/libs/algo/score_calcers.cpp:20-33` |
| CPU `CalcAverage` mask | `catboost/private/libs/algo_helpers/online_predictor.h:112-119` |
| CPU SSE2 Cosine score reference | `catboost/libs/helpers/short_vector_ops.h:67-81` |
| CPU `SumDer2` (hessian) used only for leaf-value Newton | `catboost/private/libs/algo_helpers/online_predictor.h:162-179` |
| `WeightedDerivatives = weight·der1` | `catboost/private/libs/algo_helpers/error_functions.cpp:46-60` |
| RMSE hessian = 1 | `catboost/private/libs/algo_helpers/error_functions.h:381` |
| CrossEntropy/Logloss hessian = σ(1−σ) (CPU class) | `catboost/private/libs/algo_helpers/error_functions.h:350-377` |
| Poisson hessian = exp(approx) | `catboost/private/libs/algo_helpers/error_functions.h:670-672` |
| MLX populates histogram weight channel from `dimHess` | `catboost/mlx/tests/csv_train.cpp:1801` |
| MLX `dimHess` per loss | `catboost/mlx/tests/csv_train.cpp:4067-4159` |
| MLX sample weight applied to BOTH grad and hess | `catboost/mlx/tests/csv_train.cpp:4168-4169, 4236-4237` |
| MLX FindBestSplit ordinal L2 formula | `catboost/mlx/tests/csv_train.cpp:1965-1985` |
| MLX FindBestSplit ordinal Cosine formula | `catboost/mlx/tests/csv_train.cpp:1997-2014` |
| MLX FindBestSplit one-hot formula | `catboost/mlx/tests/csv_train.cpp:1689-1732` |
| MLX leaf-value Newton step (CORRECT) | `catboost/mlx/tests/csv_train.cpp:5036-5049` |

End of derivation.
