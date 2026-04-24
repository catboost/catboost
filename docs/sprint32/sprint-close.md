# Sprint 32 — Close Report: COSINE-GAIN-TERM-AUDIT

**Branch**: `mlx/sprint-32-cosine-gain-term-audit`  
**Date closed**: 2026-04-24  
**Tip at close**: `1aaf92497b` (5 commits on branch above S31 tip `9b3a5238a7`)  
**Authoritative records**: `docs/sprint32/t1-codepath/verdict.md`, `docs/sprint32/t2-terms/verdict.md`, `docs/sprint32/t3-fix/verdict.md`, `docs/sprint32/t4-validate/`

---

## Summary

S32 shipped **three correctness fixes** in `csv_train.cpp`. The DEC-036 structural divergence (53% ST+Cosine drift) is **not closed**. Mechanism is reframed for S33.

### What shipped

| DEC | Fix | Effect |
|-----|-----|--------|
| DEC-037 (S31 T3b, co-fixed) | `maxBordersCount = maxBins` (was `maxBins-1`); restored greedy unweighted `GreedyLogSumBestSplit` | Border count off-by-one + wrong DP algorithm; closed S31 |
| DEC-038 (S32 T3) | `GreedyLogSumBestSplit` receives `allVals` (with duplicates) instead of `uniqueVals` | Border grid shifted ~2 indices; gain ratio 0.946 → 0.9999 |
| DEC-039 (S32 T3) | `maxBordersCount = std::min(maxBins, 127u)` in `QuantizeFeatures` | VALID_BIT aliasing at `fold_count=128` silently dropped 391 docs/posInWord=0 feature |

All three fixes are in `catboost/mlx/tests/csv_train.cpp`. Kernel sources (`kernel_sources.h`) are **untouched** — identical to v5 (`784f82a891`).

### What S32 did NOT close

**DEC-036 (ST+Cosine structural divergence) remains OPEN.**

Post-fix measurement at iter=50: MLX RMSE = 0.2956, CPU RMSE = 0.1937, drift = **52.6%** (pre-fix: 53.30%). The 0.7pp reduction is noise.

The original "GAIN-FORMULA" framing (ratio 0.946) was a surface symptom of the border grid bugs. With borders fixed, depth=0 gain ratio is now 1.000000. But the iter=50 drift is unchanged.

**Reframe for S33**: The iter≥2 compounding rate is ~9% per iteration (52.6% at 50 iters from a 0.75% iter=1 residual implies 70× amplification — not geometric). This is **runaway divergence** characteristic of trajectory lock-in: a 0.75% split-selection error at iter=1 flips splits at iter=2, compounding ~12× faster than pure 0.75%^50 geometric would predict (1.0075^50 = 1.45; observed 1.526). Three candidate frames:

1. **Trajectory lock-in cascade**: iter=1 0.75% split flip → iter=2 split sequence diverges catastrophically (chaotic GBDT search; residuals diverge)
2. **Per-iter persistent bug**: leaf value computation, approx update, or gradient computation has a recurring error (would appear at iter=1 but accumulate)
3. **Config/RNG mismatch**: bootstrap or noise path differs between MLX and CPU per-iter

S33 opens with an L0-L4 scaffold: L0 config audit → L1 determinism shift → L2 graft experiment (replace MLX iter=1 model with CPU iter=1 model, continue with MLX) → L3 iter=2 instrumentation → L4 fix+gates.

---

## Gate Results

| Gate | Criterion | Result | Evidence |
|------|-----------|--------|----------|
| G3a | Depth=0 gain ratio = 1.000 ± 1e-4 (3 seeds) | **PASS** | seed=42: 1.000000 (δ=4.4e-7); seed=43: 1.000000 (δ=3.5e-7); seed=44: 1.000000 (δ=7.2e-8) |
| G3b | ST+Cosine drift ≤ 2% | **FAIL** | 52.6% at iter=50, N=50k (from T3-FIX verdict); DEC-036 mechanism reframed to iter≥2 for S33 |
| G3c | bench_boosting v5 ULP=0 preserved | **PASS** | `BENCH_FINAL_LOSS=0.48231599` at AN-009 anchor; kernel md5=9edaef45b99b9db3e2717da93800e76f (byte-identical to v5) |
| G3d | 18-config L2 SymmetricTree non-regression | **PASS** | 18/18 cells in [0.98, 1.02] envelope |

**G3b FAIL** — drift target unmet; mechanism reframed to iter≥2 for S33. No progress framing applied. 52.6% is the honest number.

---

## DEC-012 Atomicity Violations This Sprint

S32 recorded **two DEC-012 atomicity violations**:

1. **DEC-037 bundled with T3b verdict** (`746d5090b5`, S31): the border count + greedy algorithm restoration was committed inside the T3b verdict doc commit. One structural code change and a doc update in the same commit.

2. **DEC-038 + DEC-039 bundled in single commit** (`901bc760ac`, S32 T3-FIX): two independent structural fixes (`allVals` and `fold_count` cap) landed in the same commit because both bugs were found during the same T3 fix session.

**S33 hard rule**: "If you find a second structural issue while fixing the first, STOP and commit the first atomically before continuing the investigation." No exceptions. Bundling obscures the causal chain and makes bisection harder.

---

## S31 carry-forward items (status at S32 close)

| Item | Status |
|------|--------|
| S31-T3-MEASURE (G3a/G3b/G3c gate matrix) | **Absorbed into S32 G3a/G3b/G3c/G3d**; not a separate deliverable |
| S31-T4a (ST-guard removal) | **Blocked** by G3b FAIL → carried to S33 (gate on drift closure) |
| S31-T4b (LG-guard removal) | **Blocked** by G3b/G3c FAIL → carried to S33 |
| S31-T-LATENT-P11 | Not addressed; documented in TODOS as low-priority carry-forward |
| S31-T-CLEANUP (SA-I2 + CR nits) | Not addressed; documented in TODOS |
| S31-CR / S31-SA / S31-CLOSE | **Subsumed into S32 close** — S31 closes jointly here |

---

## Forward pointer: S33

**DEC-036 remains OPEN.** S33 will open with a new DEC-040 and a layered SCAFFOLD:

- **L0** — Config audit: verify `bootstrap_type`, `bagging_temperature`, RNG seeding, `l2_leaf_reg` scaling are identical between MLX and CPU sides.
- **L1** — Determinism shift: measure iter=1 RMSE across 5 seeds; confirm 0.75% is stable and not seed-dependent.
- **L2** — Graft experiment: replace MLX iter=1 model (tree structure + leaf values) with CPU's iter=1 model; let MLX continue from iter=2 onward. If iter=50 drift collapses → bug is in leaf estimation or approx update on the MLX side.
- **L3** — Iter=2 instrumentation: dump per-partition split decisions at iter=2; compare split sequence MLX vs CPU after one iteration of shared history.
- **L4** — Fix + gates.

**DEC-040** (the S33 DEC) will be authored by the S33 kickoff agent. Not pre-authored here.

---

## Commits this sprint (oldest → newest on branch above S31 tip)

| SHA | Tag | Description |
|-----|-----|-------------|
| `ee6edfe426` | S32-00 | Sprint kickoff; DEC-037 formalized; DEC-038 OPEN |
| `0e24e7f8b7` | S32-T1 | T1-CODEPATH verdict — SAME-PATH; H1 eliminated |
| `5d3899090c` | S32-T2a | COSINE_TERM_AUDIT flag + per-bin dump harness |
| `1762e8d49c` | S32-T2b | T2-INSTRUMENT verdict — FORMULA CORRECT, root cause = border grid |
| `901bc760ac` | S32-T3 | T3-FIX: DEC-038 (allVals) + DEC-039 (fold_count cap) — DEC-012 violation noted |
| `2428419596` | S32-T4a | T4-VALIDATE gate scripts + raw outputs (G3a/G3d) |
| `1aaf92497b` | S32-T4b | DEC-038 + DEC-039 formalized in DECISIONS.md |

---

## Files of record

- `docs/sprint32/t1-codepath/verdict.md` — H1 eliminated; SAME-PATH confirmed
- `docs/sprint32/t2-terms/verdict.md` — formula correct; root cause = border grid
- `docs/sprint32/t3-fix/verdict.md` — DEC-038 + DEC-039 fix description and verification
- `docs/sprint32/t4-validate/run_g3a.py` — G3a gate harness (3-seed gain ratio)
- `docs/sprint32/t4-validate/run_g3d.py` — G3d gate harness (18-config L2 parity)
- `docs/sprint32/t4-validate/data/g3a_gain_ratio.csv` — G3a raw data
- `docs/sprint32/t4-validate/data/g3d_l2_parity.csv` — G3d raw data
- `.claude/state/DECISIONS.md` — DEC-038 CLOSED; DEC-039 added and CLOSED
