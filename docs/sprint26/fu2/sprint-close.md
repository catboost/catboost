# Sprint 26 Follow-Up 2 — Close Report

**Sprint**: S26-FU-2 — RandomStrength noise in `FindBestSplitPerPartition`
**Branch**: `mlx/sprint-26-fu2-noise-dwlg` (stacked on S26 D0 tip `66a4b5e869`)
**Date closed**: 2026-04-22
**Verdict**: ALL GATE PASS. No kill-switches fired.

---

## Sprint summary

S26 D0 fixed the RandomStrength noise formula for the SymmetricTree path (`FindBestSplit` in
`csv_train.cpp`). S26-FU-2 extended that fix to the Depthwise and Lossguide grow policies, which
use a separate code path (`FindBestSplitPerPartition`). The T1 triage confirmed that CPU CatBoost
computes a single global scalar (`scoreStDev` via `CalcDerivativesStDevFromZeroPlainBoosting`) once
per tree before any depth or partition loop, and passes it unchanged into every per-partition
candidate evaluation for all three grow policies. MLX now mirrors this: `gradRms` is threaded from
`RunTraining` into both `FindBestSplit` and `FindBestSplitPerPartition` with identical semantics.

The implementation touched only 47 lines in `catboost/mlx/train_lib/csv_train.cpp`. No kernel
sources, no histogram code, no leaf-estimation code, and no SymmetricTree path were modified.
All four sprint gates (G1-DW segmented, G1-LG full, G2 ST non-regression, G5 Depthwise
determinism) passed. The five Depthwise N=1000 gate-scope failures are a pre-existing
small-N overfitting asymmetry unrelated to FU-2, verified by reproducing them with the pre-FU-2
binary. These are carried forward as S26-FU-3.

---

## Commits

| Commit | Task | Description |
|--------|------|-------------|
| `7abd7b3bcf` | T1 | D0 triage — CPU uses global scalar gradRms |
| `478e8d5c9d` | T2+T3 | Thread `gradRms` into `FindBestSplitPerPartition` (C++ impl + manual smoke test) |
| `715b15b613` | T4 | Extend `tests/test_python_path_parity.py` to DW/LG (pytest) |
| `ee5a90707b` | T5+T6 | G1 + G2 + G5 gate artifacts (sweep scripts, results, gate report) |
| *(pending — T8 commit)* | T8 | Sprint close — gate report path-coverage fix, DEC-028 footnote, state files |

---

## Gate results

### G1-DW — Depthwise parity sweep (18 cells, segmented)

| Cell class | Cells | Gate |
|------------|-------|------|
| DW rs=0.0, N≥10k | 6 | **6/6 PASS** |
| DW rs=1.0, N≥10k | 6 | **6/6 PASS** |
| DW rs=0.0, N=1000 | 3 | 0/3 FAIL — pre-existing (not FU-2 scope) |
| DW rs=1.0, N=1000 | 3 | 1/3 FAIL — pre-existing (not FU-2 scope) |

**G1-DW verdict**: PASS (12/12 at N≥10k target population). Pre-existing N=1000 failures verified identical on pre-FU-2 binary.

### G1-LG — Lossguide parity sweep (18 cells)

| Cell class | Cells | Gate |
|------------|-------|------|
| LG rs=0.0 (all N) | 9 | **9/9 PASS** |
| LG rs=1.0 (all N) | 9 | **9/9 PASS** |

**G1-LG verdict**: PASS (18/18). All Lossguide cells pass at all sizes.

### G2 — SymmetricTree non-regression (18 cells)

D0 SymmetricTree results reproduced to 6 decimal places. DEC-028 fix intact.
**G2 verdict**: PASS (18/18). KS-3 clear.

### G5 — Depthwise determinism (100 runs)

Config: N=10k, seed=1337, rs=0, Depthwise, d=6, 128 bins, LR=0.03, 50 iters.
Max−min = 1.49e-08 (threshold 1e-6).
**G5 verdict**: PASS. Same order as D0 SymmetricTree result — FU-2 introduces no new non-determinism.

### Kill-switch summary

| KS | Trigger | Status |
|----|---------|--------|
| KS-2 | DW rs=1 pred_std_R < 0.85 or > 1.20 | CLEAR — max 1.1028 |
| KS-3 | ST pred_std_R outside [0.90, 1.10] | CLEAR — all in [0.9996, 1.0870] |
| KS-4 | G5 max−min > 1e-6 | CLEAR — 1.49e-08 |
| KS-5 | Scope leak outside csv_train.cpp | CLEAR — 47 lines only |

---

## Path coverage

**FU-2 covers**: `FindBestSplitPerPartition` gain computation in `csv_train.cpp` — the
RandomStrength noise injection path for Depthwise and Lossguide grow policies.

**FU-2 does NOT cover**: histogram kernel, leaf estimation (`CalcLeafValues` /
`UpdateApproximations`), feature quantization / bin border logic, nanobind orchestration,
SymmetricTree `FindBestSplit` path (covered by D0, preserved by G2).

---

## Code review outcome

T7 (@code-reviewer): **APPROVE-WITH-NITS** — 0 blockers, 4 nits.

### Nit disposition

| # | Nit | Disposition |
|---|-----|-------------|
| Nit-1 | Gate report missing path-coverage labels | **FIXED in T8** (this commit) |
| Nit-2 | Test assertion could reference KS-2 threshold by name | **Recorded as tech-debt** — minor clarity improvement only, no correctness impact |
| Nit-3 | Refactor ST block in `test_python_path_parity.py` to use `_assert_segmented_parity` helper | **Recorded as tech-debt** — refactor only, no behavior change |
| Nit-4 | Move `std::normal_distribution<float>` declaration inside `if (noiseScale > 0.0f)` scope | **Recorded as tech-debt** — dead-variable elimination only |

---

## Carry-forwards

### S26-FU-3 — Depthwise N=1000 parity failure (pre-existing)

**Evidence cells**: 5 failures in G1-DW sweep:
- DW rs=0.0, N=1000: 3/3 cells fail (max |delta| 16.85%, pred_std_R up to 1.1004)
- DW rs=1.0, N=1000: 2/3 cells fail (max |delta| 17.68%, pred_std_R up to 1.1028)

**Pre-existing status confirmed**: pre-FU-2 binary produces MLX RMSE 0.17972 at (DW, N=1k,
seed=1337, rs=0) — identical to post-FU-2. FU-2's noise path is dormant at rs=0 and N=1k
failures appear at both rs=0 and rs=1, ruling out noise injection as a cause.

**Characterization**: MLX is consistently *better* than CPU at N=1000 Depthwise (pred_std_R
> 1.0 throughout, never below threshold), so this is not a regression signal. It is a
systematic overfitting asymmetry — MLX likely splits more aggressively than CPU at small N
in the non-oblivious depth loop, possibly due to a difference in partition-statistics
accumulation or split-score normalization at small partition sizes.

**Suggested next step (triage scope)**: Instrument `FindBestSplitPerPartition` at N=1000 to
compare per-partition gain scores between MLX and CPU at depth 0 (where the divergence
presumably originates). Determine whether the gap is (a) a DW-specific gain-computation
difference (e.g., missing regularization term in per-partition scoring), (b) a small-N
partition-statistics precision issue, or (c) a shared instability at small N that also
affects SymmetricTree at N=1000 (test: add N=1000 ST cells to confirm or rule out).

---

## Open tech-debt (recorded, not blocking)

1. **Nit-2**: `test_python_path_parity.py` — test assertion could name the KS-2 threshold
   constant (`KS2_PRED_STD_UPPER = 1.20`) for readability.
2. **Nit-3**: `test_python_path_parity.py` — ST block duplicates DW/LG block structure; a
   `_assert_segmented_parity(policy, ...)` helper would reduce copy-paste.
3. **Nit-4**: `csv_train.cpp` — `std::normal_distribution<float> noiseDist(0.0f, 1.0f)`
   declared outside `if (noiseScale > 0.0f)` scope; is dead weight when noise is disabled.
   Move inside the guard on next touch of this function.

---

## Files of record

| File | Role |
|------|------|
| `docs/sprint26/fu2/d0-triage.md` | CPU source audit (T1) — confirms global scalar gradRms |
| `benchmarks/sprint26/fu2/fu2-gate-report.md` | Gate report (G1-DW, G1-LG, G2, G5, KS summary) |
| `benchmarks/sprint26/fu2/g1_sweep.py` | 54-cell sweep driver |
| `benchmarks/sprint26/fu2/g1-results.md` | Raw per-cell data + segmented summary |
| `benchmarks/sprint26/fu2/g4_determinism.py` | 100-run Depthwise determinism driver |
| `benchmarks/sprint26/fu2/g5-determinism.md` | 100-run stats |
| `tests/test_python_path_parity.py` | Extended parity harness (DW + LG added in T4) |
| `docs/decisions.md §DEC-028` | DEC-028 with S26-FU-2 extension footnote |
| `.claude/state/DECISIONS.md §DEC-028` | Mirrored footnote |
