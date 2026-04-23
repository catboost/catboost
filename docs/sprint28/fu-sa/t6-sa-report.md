# S28-SA — Sprint 28 Security Audit Report

**Date:** 2026-04-23
**Auditor:** @security-auditor
**Commits audited:** `83f30c3677`, `0ea86bde21`, `c07e895f7c`, `dca62f0d72`, `4083add248`, `b9577067ef`
**Scope:** numerical bounds, input validation, guard-completeness on Cosine dispatch

**Note:** report persisted by orchestrator; audit agent returned findings inline due to Write-tool restriction.

---

## Summary

**Verdict: PASS-WITH-FINDINGS.** Counts: Critical 0 / High 1 / Informational 3.

Five gates (G-SA1..G-SA5) pass with one non-blocking High. Cosine numerical guards are present and strictly more conservative than the CPU reference `TCosineScoreCalcer`. Input validation is layered correctly (Python → C++ `ParseScoreFunction`). Combination guards (Cosine+LG, Cosine+ST) are correct on the canonical `CatBoostMLX.fit()` / `cross_val_score` Python surface, but bypassable via direct `_core.train()` invocation and the `csv_train` CLI — both explicitly scoped to S29.

---

## Gate-by-gate findings

### G-SA1 Cosine numerical bounds — **PASS**

`ComputeCosineGain` / `ComputeCosineGainKDim` (csv_train.cpp:1026-1056):

- `cosDen` seeded to `1e-20f` at every accumulation site (lines 1259, 1476, 1553, OBLIV equivalents); `sqrt(1e-20f) = 1e-10f`, well above FP32 subnormal (~1.175e-38).
- `weightLeft/Right < 1e-15f → continue` guard precedes every L2/Cosine accumulator (1155, 1307, 1486, 1562) — strictly more conservative than CPU `TCosineScoreCalcer` (score_calcers.h:47-74) which relies on `L2Regularizer=1e-20` inside `CalcAverage` without a pre-skip.
- Scalar `ComputeCosineGain` returns `-inf` (not NaN) on zero-weight inputs.
- `num`, `den` are sums of non-negative terms — no `sqrt(negative)` path.
- Empty-partition case: `ComputeCosineGainKDim(0, 1e-20f) = 0.0f` — well-defined, not competitive vs `-inf` initial bestGain.

### G-SA2 Overflow / Inf-NaN poisoning on DW+Cosine — **PASS**

At 50k docs × K=3 × 64 partitions, `cosNum` ≤ ~2e22 vs FP32 max 3.4e38; 1e16× margin. DW+Cosine path is precision-degraded (~1.6%, documented) but not at risk of overflow or Inf/NaN poisoning. Inf/NaN can only enter from upstream gradient layer; out of scope here. API-layer guards correctly gate the two paths (LG, ST) where degradation exceeds acceptable envelope.

### G-SA3 Enum-string input validation — **PASS**

`ParseScoreFunction` (csv_train.cpp:984-997): known values return; reserved `NewtonL2`/`NewtonCosine` throw `std::invalid_argument` with a distinct message; unknown strings throw. No silent default-to-L2. Python layer (core.py:615-626) enforces identical rejection before C++ entry. `switch/case` dispatch sites (1174, 1326, 1506, 1579) throw `std::logic_error` on default — defensive canary.

### G-SA4 Combination-guard completeness — **PASS-WITH-FINDING (High, SA-H1)**

Coverage matrix:

| Entry path | Reaches `_validate_params`? | Guarded? |
|---|---|---|
| `CatBoostMLX.fit()` | yes (core.py:1313) | yes |
| `CatBoostMLX.cross_val_score` | yes (core.py:2204) | yes |
| Mutate `clf.score_function` then `fit()` | yes (re-validated) | yes |
| `_core.TrainConfig()` + `_core.train()` direct | **no** | **no — BYPASS** |
| `csv_train` CLI `--score-function Cosine --grow-policy {SymmetricTree,Lossguide}` | n/a | **no — BYPASS** |

Both bypass paths silently produce ~47% aggregate-metric drift vs CPU-Cosine baseline. Not memory-safety; correctness-of-ML-output. Advanced-user / developer-workflow surfaces only. Commit `b9577067ef` message explicitly reserves CLI guard for S29.

### G-SA5 Information disclosure in error messages — **INFORMATIONAL**

`ValueError` texts at core.py:634, 644 mention `TODO-S29-LG-COSINE-RCA`, `TODO-S29-ST-COSINE-KAHAN`, and the 47% drift rationale. Appropriate for an open-source research project; flagged per spec but not actionable.

---

## Critical (BLOCK sprint close)

None.

## High (should-fix, non-blocking)

### [SA-H1] Combination guards bypassable via nanobind and CLI entry points

- **Category:** A04 Insecure Design — incomplete validation coverage
- **Location:** `python/catboost_mlx/_core/bindings.cpp:54,77-` (score_function/grow_policy as unchecked `def_rw` fields; single `train` entry); `catboost/mlx/tests/csv_train.cpp:234,236` (CLI flags accepted without cross-check); `catboost/mlx/train_api.cpp:24-60` (`TrainConfigToInternal` copies ScoreFunction without combination check)
- **Impact:** silent ~47% aggregate-metric drift on direct-C++/CLI callers; no memory-safety risk
- **Remediation for S29:** port the two `_validate_params` guards (core.py:628-647) into `TrainConfigToInternal` (train_api.cpp:24) and into `ParseArgs` after csv_train.cpp:239
- **Verification for S29:** unit test that `_core.train()` with forbidden combo throws `std::invalid_argument`; CLI test that exit-1s on same combo
- **Does NOT block Sprint 28 close** — commit message explicitly scopes to S29; Python (canonical customer surface) is fully guarded

## Informational

- **[SA-I1]** MLX `weightLeft/Right < 1e-15f` pre-skip is strictly more conservative than CPU `TCosineScoreCalcer`. Defensive; source of documented 1-ULP drift in t2-gate-report. No action.
- **[SA-I2]** Error messages reference internal sprint tickets — appropriate for open-source research project. No action.
- **[SA-I3]** `_core.TrainConfig` fields `score_function` / `grow_policy` are `def_rw` strings with no validation setter. Unknown values are eventually caught by `ParseScoreFunction`; combination bypass is the actual gap (tracked under SA-H1).

---

## S29 follow-ups confirmed

1. **S29-CLI-GUARD** (commit `b9577067ef` message): combination rejection in csv_train CLI + `TrainConfigToInternal`. Tracks SA-H1.
2. **TODO-S29-LG-COSINE-RCA** (core.py:634): priority-queue × Cosine magnitude interaction RCA.
3. **TODO-S29-ST-COSINE-KAHAN** (core.py:644): Kahan/Neumaier port to ST+Cosine denominator.

## Positive findings

- `ParseScoreFunction` fails loud on reserved values rather than silent L2 fallback — exactly what DEC-032 mandated.
- `switch/case` default-throw canaries in all four dispatch sites (csv_train.cpp:1174, 1326, 1506, 1579).
- `cosDen = 1e-20f` seed has inline rationale (csv_train.cpp:1016-1018).
- Bool-as-int footgun guard (core.py:549-565) — defence-in-depth.
- Python-side combination rejection precedes data conversion — fail-fast.
