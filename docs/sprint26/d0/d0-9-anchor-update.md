# S26-D0-9 — Sprint 4 Anchor Update (post-DEC-028)

**Date**: 2026-04-22
**Branch**: `mlx/sprint-26-python-parity`
**PR**: #23
**Scope**: `python/tests/test_qa_round9_sprint4_partition_layout.py` — 5 numeric constants across 3 tests

## Trigger

CI on PR #23 failed on `test_rmse_final_loss_matches_sprint4_anchor`:

```
AssertionError: RMSE anchor mismatch: expected ~0.432032, got 0.306348
```

Pytest `-x` exited at first failure; local run without `-x` revealed 2 more anchors in the same file drifted identically.

## Reproduction

Local run on D0 tip `66a4b5e869` matches CI exactly:

| Test | Old anchor | New value (local) | CI value |
|---|---|---|---|
| `test_rmse_final_loss_matches_sprint4_anchor` | 0.432032 | 0.306348 | 0.306348 |
| `test_specific_predictions_match_anchor` preds[0] | 0.414606 | 0.317253 | (not reached due to `-x`) |
| same, preds[1] | −0.545893 | −0.568259 | — |
| same, preds[99] | 1.356884 | 1.598960 | — |
| `test_multiclass_k3_proba_anchor` | [0.35687973, 0.36606121, 0.27705906] | [0.37227302, 0.36382151, 0.26390547] | — |

## Stability evidence

**Regression seed=0, 3 runs**: RMSE 0.30634809 ± ~6e-9; preds[0] 0.31725318 ± ~4e-9. Well under the 1e-3 test tolerance and under the pre-existing ~2.75e-7 Metal non-determinism floor documented in the test module's FINDING-002.

**Across seeds 0 / 1 / 7 / 42 / 99** (same data=seed0): RMSE range [0.304121, 0.308569], mean ≈ 0.306. Seed=0's 0.306348 sits central in the distribution — not a lucky draw.

**Multiclass seed=0, 3 runs**: proba[0] stable to ~1e-8.

## Attribution: DEC-028 alone

The anchor shift is fully attributable to **DEC-028** (RandomStrength noise scale: `hessian_sum` → `sqrt(sum(g²)/N)` gradient RMS, matching CPU's `CalcDerivativesStDevFromZeroPlainBoosting`). **DEC-029** (non-oblivious tree serialization) is not exercised because these tests use the default `grow_policy="SymmetricTree"` (verified in `python/catboost_mlx/core.py:474`).

### Evidence: `random_strength` ablation (same config, seed=0)

| `random_strength` | RMSE | preds[0] |
|---:|---:|---:|
| 0.0 | 0.282508 | 0.314329 |
| 1.0 (default) | **0.306348** (new anchor) | **0.317253** |
| 2.0 | 0.317802 | 0.235238 |

Smooth monotone scaling of RMSE with noise magnitude is the expected shape when the noise formula is correct. The pre-fix RMSE 0.432032 lies far above this clean curve — consistent with the buggy `hessian_sum` scale producing noise of wrong magnitude for the gain being perturbed, degrading split selection.

## Historical precedent

Same pattern as **TODO-022** (Sprint 8) in which the bench_boosting K=10 anchor was updated from `2.22267818` → `1.78561831` after a correctness fix exposed the old anchor as stale. Recorded in `.claude/state/MEMORY.md#Resolved / mitigated (historical)`:

> **bench_boosting K=10 anchor** (RESOLVED Sprint 8, TODO-022): `1.78561831` is the canonical anchor at `20k × 30 × depth 5 × 50 iters`; the prior `2.22267818` "expected" value was captured from a mismatched-param run.

The general lesson — stale numeric anchors survive quietly until a correctness fix exposes them — is worth remembering as a parity-gate pattern (next to the "Kernel-ULP=0 ≠ full-path parity" note already in MEMORY.md).

## Decision

One atomic commit on `mlx/sprint-26-python-parity` updating the 5 anchor constants plus inline comment notes. Commit preserves DEC-012 (one structural change per commit). PR #23 auto-rebuilds; the 3 anchor tests flip PASS, `-x` proceeds past them, CI clears.

## Non-goals

- No change to the test tolerance (remains 1e-3 for RMSE/preds, 1e-3 atol for proba).
- No change to any DEC entry. This is a test-data refresh; DEC-028 itself already documents the underlying fix.
- No broader test audit. Local scan for numeric-pattern asserts yielded only the 5 values in this file; other anchor tests (`test_qa_round10_sprint5_bench_and_scan.py`) use the standalone `bench_boosting` binary, which is not on the DEC-028 code path (separate CLI — exercises the kernel only, per the `bench_boosting coverage gap` memory).
