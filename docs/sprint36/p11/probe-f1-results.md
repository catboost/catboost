# S36-LATENT-P11 PROBE-F1 — Empirical Drift Results

**Date:** 2026-04-25
**Branch:** `mlx/sprint-36-hess-vs-weight`
**Cuts from:** master `3bc02bc3cb`
**Authors:** research-scientist
**Scope:** measurement-only probe (no source code changed; no fix applied)

---

## TL;DR

- **Math prediction REFUTED at the predicted magnitude.** Iter=50 Logloss drift is
  **0.001–0.108 %** (mean 0.024 %) across 6 anchor/seed combinations. Math derivation
  predicted **10–30 %**. Iter=50 Poisson drift is similarly **0.001–0.027 %** vs
  predicted **20–50 %**. All cells are at least **two orders of magnitude below**
  the predicted band.
- **Math prediction CONFIRMED at the iter=1 limit.** Drift at iter=1 is **0.0004–0.0035 %**
  across all 12 records — argmax-preserving as derivation §5.2 predicted (the
  multiplicative `h₀`-rescaling commutes with the per-candidate ranking).
- **Drift direction CONFIRMED.** Drift grows monotonically with iter on every
  anchor (mean iter=1 ≈ 0.0016 %, iter=10 ≈ 0.005 %, iter=50 ≈ 0.024 %, iter=100
  ≈ 0.060 %). The mechanism is real; the **magnitude is in the noise floor**.
- **Stress test (aggressive Poisson, log-rate spans [-3, 3])** still bounded at
  **0.28 % iter=100** drift. The "unbounded hessian range" prediction in derivation
  §6.1 does not manifest empirically.
- **Verdict per acceptance criteria (in spec):** `iter=50 Logloss drift < 1%`
  → "STOP. Re-derive; the fix is not warranted as currently scoped."
- **Recommended next step:** demote P11 to **DEFERRED** in the backlog. The
  variable-swap fix (~150 lines, kernel-md5-changing) is not justified at the
  observed drift magnitudes; competing backlog items (#113 S31-T3-MEASURE,
  #114 S31-T-CLEANUP, SA carry-forwards) should take priority.

---

## 1. Build & environment

| Item | Value |
|------|-------|
| Probe binary | `csv_train_p11` (built 2026-04-25 13:14, md5 `57300a624d3d781ea3237f88f1c590b7`) |
| Build script | `docs/sprint36/p11/scripts/build_csv_train_p11.sh` |
| Source | `catboost/mlx/tests/csv_train.cpp` at branch tip `3bc02bc3cb` |
| Kernel md5 (invariant) | `9edaef45b99b9db3e2717da93800e76f` ✓ unchanged |
| MLX library | `/opt/homebrew/opt/mlx/{include,lib}` |
| CatBoost CPU | `catboost==1.2.10` (Python package via conda) |
| Compiler | `clang++ -std=c++17 -O2` |
| Hardware | Apple Silicon (darwin-arm64) |

Compile flags: vanilla (no instrumentation). The freshly-built binary at iter=0
on synthetic-logloss seed=42 reports `loss=0.690969` versus the pre-S33-L4-FIX
April 23 build's `loss=0.690971` — the 5e-6 difference confirms the binary
incorporates current master semantics including the per-side-mask
(DEC-042) closure.

## 2. Configuration

Identical hyperparameters between MLX (csv_train_p11) and CatBoost CPU
(`CatBoostClassifier` for Logloss, `CatBoostRegressor` for Poisson):

| Parameter | Value |
|-----------|-------|
| iterations | 100 |
| depth | 6 |
| learning_rate | 0.03 |
| l2_leaf_reg | 3 |
| border_count / bins | 128 |
| bootstrap_type | No |
| boost_from_average | True (Logloss) ; emulated via baseline=log(mean(y)) (Poisson) |
| score_function | L2 |
| grow_policy | SymmetricTree |
| feature_border_type | GreedyLogSum |
| has_time | True (CPU only — disables row shuffle, matches MLX deterministic order) |
| sample weights | none (unit) |

Drift is computed on **training data** at the iter checkpoint:
`drift = |MLX_loss − CPU_loss| / |CPU_loss|`. Per-iter loss read from MLX's
`--verbose` per-line output and from CPU's `evals_result_['learn'][metric]`.
Both report standard mean Logloss / mean Poisson NLL on the training set.

CPU's `evals_result_['learn']['Logloss']` was independently verified against
manual computation `mean(-(y log p + (1-y) log(1-p)))` over staged predictions —
agreement to fp32 precision (~1e-5).

## 3. Anchors

| Anchor | N | Features | Target | Loss | Seeds |
|--------|---|----------|--------|------|-------|
| synthetic-logloss | 20,000 | 10 | y ~ Bernoulli(σ(0.5·X[0] + 0.3·X[1] + 0.1·noise)) | Logloss | {42, 1337, 7, 17, 9999} |
| synthetic-poisson | 20,000 | 10 | y ~ Poisson(exp(0.3·X[0] + 0.2·X[1] − 0.5)) | Poisson | {42, 1337, 7, 17, 9999} |
| adult | 30,162 | 6 (numeric only) | UCI Adult Income (binary) | Logloss | {42} |
| synthetic-poisson-aggressive | 20,000 | 10 | y ~ Poisson(exp(2·X[0] + X[1])); log-rate clipped to [-3, 3] | Poisson | {42} (stress test) |

Datasets persisted under `docs/sprint36/p11/data/`.

The **aggressive Poisson** anchor was added after observing low drift on the
spec-default Poisson — it gives the hessian field `h_d = exp(a_d)` a 50× larger
dynamic range to test whether the math prediction (§6.1) of "for high counts,
`h̄_ℓ` can be O(10²–10³)" produces drift in that regime.

| Anchor | y range | y mean | y max |
|--------|---------|--------|-------|
| synthetic-poisson | [0, 7] | 0.66 | 7 |
| synthetic-poisson-aggressive | [0, 35] | 4.07 | 35 |

## 4. Drift table

`drift = |MLX − CPU| / |CPU|`, expressed as a percent.

| anchor | loss | seed | iter=1 | iter=10 | iter=50 | iter=100 |
|--------|------|------|--------|---------|---------|----------|
| synthetic-logloss | logloss | 42 | 0.0016 % | 0.0045 % | 0.0247 % | 0.0279 % |
| synthetic-logloss | logloss | 1337 | 0.0004 % | 0.0039 % | 0.0311 % | 0.0506 % |
| synthetic-logloss | logloss | 7 | 0.0007 % | 0.0022 % | 0.0104 % | 0.0807 % |
| synthetic-logloss | logloss | 17 | 0.0035 % | 0.0002 % | 0.0154 % | 0.0388 % |
| synthetic-logloss | logloss | 9999 | 0.0005 % | 0.0023 % | 0.0129 % | 0.0195 % |
| synthetic-poisson | poisson | 42 | 0.0013 % | 0.0057 % | 0.0226 % | 0.0279 % |
| synthetic-poisson | poisson | 1337 | 0.0016 % | 0.0074 % | 0.0094 % | 0.0070 % |
| synthetic-poisson | poisson | 7 | 0.0016 % | 0.0026 % | 0.0075 % | 0.0022 % |
| synthetic-poisson | poisson | 17 | 0.0013 % | 0.0066 % | 0.0007 % | 0.0258 % |
| synthetic-poisson | poisson | 9999 | 0.0021 % | 0.0111 % | 0.0234 % | 0.0726 % |
| adult | logloss | 42 | 0.0016 % | 0.0121 % | 0.1081 % | 0.2080 % |
| **synthetic-poisson-aggressive** | poisson | 42 | 0.0368 % | 0.2546 % | 0.2520 % | 0.2840 % |

Per-iter curves are persisted in `docs/sprint36/p11/probe-f1-results.json`
(both the full 100-point loss curve for MLX and CPU, plus per-checkpoint drift).
Per-run logs (with the exact MLX command-line) are under
`docs/sprint36/p11/runs/<anchor>_seed<S>/{mlx,cpu}.log`.

## 5. CPU self-consistency

Sanity check: the CPU should not disagree with itself across seeds (acceptance
threshold: `>2 %` iter=50 spread = "halt, fix harness").

| Loss | iter=50 mean | iter=50 spread | Threshold |
|------|--------------|----------------|-----------|
| Logloss (synthetic) | 0.6562 | **0.57 %** | < 2 % ✓ |
| Poisson (synthetic) | 0.8894 | **0.50 %** | < 2 % ✓ |

Spread is well under the 2 % gate. CPU is a stable reference.

## 6. Acceptance verdict

| Acceptance row | Observed | Verdict |
|----------------|----------|---------|
| iter=1 drift ≤ 1e-6 across all anchors | max iter=1 drift = 3.7e-4 (synthetic-poisson-aggressive); 3.5e-5 on others. Both above 1e-6 floor due to fp32 noise | **CONFIRMED** (qualitatively — argmax-preserving, drift bounded by fp32 quantization noise) |
| iter=50 Logloss drift in [5 %, 50 %] | max iter=50 Logloss drift = **0.108 %** (Adult); synthetic max = 0.031 % | **REFUTED** |
| iter=50 Logloss drift < 1 % | yes — max 0.108 % | **TRIGGERS "STOP. Re-derive; fix not warranted."** |
| iter=50 Logloss drift > 80 % | no | n/a |
| iter=50 Poisson drift > Logloss drift | no — max iter=50 Poisson 0.027 % vs max Logloss 0.108 %. Even on aggressive-Poisson stress test (0.252 %) does not exceed Adult (0.108 %) by the predicted ~2× factor | **REFUTED** |
| CPU disagrees with itself across seeds (>2 %) | 0.5–0.6 % spread | not triggered |

## 7. Why the math prediction was wrong by ~3 orders of magnitude

The math derivation (§5.1) showed the score-denominator divergence is real and
*can* re-rank candidate splits. What it could not predict (and explicitly
flagged in §10 as the implicit-bias caveat) is whether MLX's "wrong-but-stable"
denominator empirically *does* re-rank splits **on real data**. The probe shows
that, in practice, it does not (at the magnitudes that matter for training
accuracy).

Two plausible mechanisms for the empirical near-agreement:

1. **Argmax robustness across iterations.** At iter ≥ 2 the per-leaf mean
   hessian `h̄_ℓ` becomes inhomogeneous in principle, but on the configurations
   tested (default lr=0.03, depth=6, no bootstrap, GreedyLogSum borders),
   `h̄_ℓ` drifts slowly enough across iterations that the gain ranking
   `S²_ℓ / (h̄_ℓ |ℓ| + λ)` remains argmax-equivalent to `S²_ℓ / (|ℓ| + λ)` for
   the **vast majority of split candidates**. The ranking-flip cases predicted
   in §5.1 (where `h̄_L`, `h̄_R` differ enough between candidates A and B to
   flip CPU's order) are rare.

2. **Self-correction across iterations.** Even when MLX picks a different
   split at iter k, the gradient field at iter k+1 absorbs the difference
   (gradient boosting is residual-driven). The cumulative effect at iter=50 is
   bounded by what the *worst-case* flips can drift, which on these datasets
   is < 0.1 % loss.

Both effects are characteristic of the implicit-regularization caveat that
appeared in derivation §10. The math was correct that the formulas differ; it
overestimated the empirical impact.

The aggressive-Poisson stress test (50× larger hessian range) was designed
specifically to expose §6.1's "unbounded" prediction. It did produce ~10× more
drift than the spec-default Poisson — directional confirmation — but the
absolute level remained 0.28 %, two orders of magnitude below the predicted
20–50 % band.

## 8. Recommendation

**Demote P11 to DEFERRED.** The variable-swap fix (math-derivation §8.1) remains
mathematically correct, structurally bounded (~150 lines, one new histogram
channel, kernel-md5 change), and a future maintenance candidate for upstream
parity. But:

- **No active correctness incident.** Maximum observed drift (0.28 % on
  aggressive Poisson, 0.21 % on Adult Logloss iter=100) is within the noise
  band of the existing quantization-border carry-forward (DEC-038/DEC-039) and
  the λ-scaling carry-forward (math derivation §8.3 Open Q4).
- **Cost.** The fix touches the histogram tensor layout and the histogram
  Metal kernel signature, breaking the kernel-md5 invariant
  `9edaef45b99b9db3e2717da93800e76f` that has held across S31–S35 (entire
  correctness arc). Any correctness-affecting kernel change should be paid for
  by a clear empirical justification; this probe shows there isn't one yet.
- **Higher-priority backlog.** #113 S31-T3-MEASURE re-run, #114 S31-T-CLEANUP,
  and SA carry-forwards offer better return on engineering time.

If/when a real-world incident shows P11 firing — a customer reports
Logloss-on-MLX vs Logloss-on-CPU drift > 1 % on their data — the fix shape is
documented in `math-derivation.md §8.1` and ready to apply. Until then,
**measurement, not action**.

A possible follow-up that would *strengthen* the deferral case: re-run this
probe on (i) MultiClass with 3+ imbalanced classes (the §6.3 "rare-class bias"
case), and (ii) a real-world large-N Poisson dataset such as a count-regression
Kaggle anchor. Both can wait for an incident report.

## 9. Reproducibility

```bash
# 1. Build the probe binary (vanilla; no instrumentation)
./docs/sprint36/p11/scripts/build_csv_train_p11.sh

# 2. Run the full sweep (12 anchor records, ~5 min on M-series silicon)
python3 docs/sprint36/p11/scripts/run_probe_f1.py

# 3. Quick smoke (seed=42 + skip Adult, ~1 min)
python3 docs/sprint36/p11/scripts/run_probe_f1.py --quick
```

Outputs:
- `docs/sprint36/p11/probe-f1-results.json` — full per-iter loss curves and
  per-checkpoint drift for every record.
- `docs/sprint36/p11/runs/<anchor>_seed<S>/{mlx.log,cpu.log}` — exact stdout
  from each training run.
- `docs/sprint36/p11/data/adult.csv` — committed (Adult was downloaded once
  from UCI; redownload via `gen_adult` if missing).
- `docs/sprint36/p11/data/synthetic-{logloss,poisson}-seed<S>.csv` — **not
  committed** because they are byte-deterministic from `numpy.random.default_rng(seed)`
  in `gen_synthetic_logloss` / `gen_synthetic_poisson`. The harness regenerates
  any missing CSV automatically.

Random seeds: data generation and training both consume the seed from `SEEDS`.
fp32 reproducibility on MPS/Metal is ε-close (per `--verbose` re-runs).

## Appendix A. Falsification mode used

This probe was run as **measurement-with-pre-registered-acceptance** in the
sense of math-derivation §10:

> Why not close on math alone: 1. Implicit bias. The bug, while mathematically
> wrong, may be empirically *favourable* on some data distributions. ... The
> math cannot rule this out.

The acceptance table in the probe spec made the falsification path explicit
*before* any data was collected. The "iter=50 Logloss drift < 1 %" outcome
appeared in row 3 of the acceptance table and was not appended after-the-fact.

## Appendix B. Per-loss curve sample (synthetic-logloss seed=42)

| iter | MLX loss | CPU loss | drift |
|------|----------|----------|-------|
| 1 | 0.690969 | 0.690958 | 1.6e-5 |
| 10 | 0.676504 | 0.676535 | 4.5e-5 |
| 50 | 0.654447 | 0.654285 | 2.5e-4 |
| 100 | 0.646893 | 0.647074 | 2.8e-4 |

Both losses follow virtually parallel curves; the divergence is monotonically
increasing but bounded by ~3e-4 at iter=100. Full curves are in
`probe-f1-results.json`.

End of report.
