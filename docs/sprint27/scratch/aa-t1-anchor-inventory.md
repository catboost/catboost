# S27-AA-T1 — Numeric Anchor Inventory

**Produced:** 2026-04-22 (Sprint 27, Track B)
**Agent:** @qa-engineer
**Purpose:** Enumerate every committed numeric anchor for T2 re-run and T3 drift classification.
**Scope:** live assertions + documented canonical values that would drift if generating code changed.

---

## Anchor Table

| ID | Path | Line | Value | Kind | Harness-to-regen | Last-touched-sha | Captured-context |
|----|------|------|-------|------|-----------------|-----------------|-----------------|
| AN-001 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 91 | `0.306348` | RMSE | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_rmse_final_loss_matches_sprint4_anchor` | `634a72134d` | csv_train path, N=100, 20 features, 10 iters, depth 4, seed 0, rs=1, max_bin=32; updated S26-D0-9 after DEC-028 RandomStrength fix (prior value 0.432032) |
| AN-002 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 107 | `0.317253` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_specific_predictions_match_anchor` | `634a72134d` | preds[0] from same Sprint 4 100-row config post-DEC-028; prior value 0.414606 |
| AN-003 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 108 | `-0.568259` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_specific_predictions_match_anchor` | `634a72134d` | preds[1] same config post-DEC-028; prior value -0.545893 |
| AN-004 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 109 | `1.598960` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_specific_predictions_match_anchor` | `634a72134d` | preds[99] same config post-DEC-028; prior value 1.356884 |
| AN-005 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 155 | `[0.37227302, 0.36382151, 0.26390547]` | prediction | `pytest python/tests/test_qa_round9_sprint4_partition_layout.py::TestRegressionAnchor::test_multiclass_k3_proba_anchor` | `634a72134d` | first-row softmax proba from Sprint 4 multiclass K=3 config, post-DEC-028; prior values [0.35687973, 0.36606121, 0.27705906] |
| AN-006 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py` | 66 | `0.11909308` | logloss | `pytest python/tests/test_qa_round10_sprint5_bench_and_scan.py::TestBenchBoostingAnchors::test_binary_100k_anchor` (requires `/tmp/bench_boosting`) | `f804c56742` | bench_boosting binary: 100k × 50 × cls=2 × depth6 × 100iters × bins32 × seed42; updated BUG-002 fix (prior 0.69314516, then 1.07820153 pre-BUG-001) |
| AN-007 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py` | 67 | `0.63507235` | logloss | `pytest python/tests/test_qa_round10_sprint5_bench_and_scan.py::TestBenchBoostingAnchors::test_multiclass_20k_anchor` (requires `/tmp/bench_boosting`) | `f804c56742` | bench_boosting: 20k × 30 × cls=3 × depth5 × 50iters × bins32 × seed42; updated BUG-002 fix (prior 1.09757149 pre-BUG-002) |
| AN-008 | `CHANGELOG.md` | 41 | `1.78561831` | logloss | Build bench_boosting then: `./bench_boosting --rows 20000 --features 30 --classes 10 --depth 5 --iters 50 --seed 42` | `fd04d34684` | K=10 multiclass, 20k × 30 × depth5 × 50iters; corrected Sprint 8 TODO-022 (prior stale value 2.22267818 from mismatched params) |
| AN-009 | `docs/sprint19/results.md` | 41 | `0.48231599` | RMSE | Build bench_boosting from tip then: `./bench_boosting --rows 10000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42` | `b04b1efd2c` | Sprint 19 gate config 10k/RMSE/d6/128b, post-S19-13; DEC-023 parity anchor (Value A); config #8 is bimodal at ~50/50 with 0.48231912 (BUG pre-v5) |
| AN-010 | `docs/sprint19/results.md` | 47 | `0.47740927` | RMSE | Build bench_boosting from tip then: `./bench_boosting --rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42` | `b04b1efd2c` | Sprint 19 gate config 50k/RMSE/d6/128b (primary determinism gate); 100/100 determinism confirmed S19, S22-D3, S24-D0 |
| AN-011 | `docs/sprint19/results.md` | 34–49 | 16 additional config losses (1k–50k × RMSE/Logloss/MultiClass × 32/128, excl. #8 and #14 already listed) | RMSE / logloss / multiclass | `./bench_boosting` with each {N, loss, bins} combination; same fixed params as AN-009/010 | `b04b1efd2c` | S19-04 18-config bit-exact parity sweep; these are the declared DEC-008 reference values post-S19-13. Note: values differ from sprint18 table (different kernel version). |
| AN-012 | `docs/sprint18/parity_results.md` | 31–48 | 18 config losses (e.g. `0.44764100`, `0.46951200`, etc.) | RMSE / logloss / multiclass | Build bench_boosting at commit `19fa5ce6cc` (sprint18 kernel) then run 18-config DEC-008 grid | `7ab4e8e804` | S18-04b parity results — reference for post-BUG-S18-001 fixed kernel; determinism anchor 0.48016092 at 10k/RMSE/128b (used in S18 gate); superseded at kernel level by S19 values (AN-011) |
| AN-013 | `docs/sprint17/parity_results.md` | 22–39 | 18 config losses captured under csv_train sprint16 vs sprint17 (e.g. RMSE 10k/128 = `0.496241`) | RMSE / logloss / multiclass | Run csv_train sprint16 binary vs sprint17 binary on the 18-config CSV grid | `ed0ec8221b` | S17-04 parity sweep; reference binary is `csv_train_sprint16` (serial reduction). All ULP=0. Uses csv_train+CSV path, not bench_boosting — different data synthesis. |
| AN-014 | `benchmarks/results/m3_max_128gb.md` | 13–30 | e.g. MLX RMSE 10k = `3.0407`, MLX Logloss 10k = `0.4099` (9 MLX loss values total) | RMSE / logloss / multiclass | Run `python benchmarks/bench_mlx_vs_cpu.py` with catboost_mlx 0.3.0 on M3 Max | `7b36f60a82` | Sprint 14 benchmark run; catboost_mlx 0.3.0 (pre-DEC-028/DEC-029 fixes). Not asserted in tests. Purely documentary. |
| AN-015 | `python/tests/test_qa_round12_sprint9.py` | 655, 661 | `0.59795737`, `0.95248461` | logloss | Requires `.github/workflows/mlx_test.yaml` to contain these strings (UNKNOWN — target file does not contain them; test skips when `mlx_test.yaml` missing) | `b8a0ab258a` | Sprint 9 Item H: asserts CI workflow YAML contains binary baseline 0.59795737 and multiclass K=3 baseline 0.95248461. The target file (`mlx_test.yaml`) does not currently contain these values — test effectively always skips. See AMBIGUOUS note below. |
| AN-016 | `docs/sprint26/d0/g1-g3-g4-report.md` | 101 | `0.19457837` | RMSE | Run `python benchmarks/sprint26/d0/g4_determinism.py` (S26 G5 determinism, 100 runs at N=10k/seed=1337/rs=0) | `cbbfc29257` | S26 G5 determinism: 100 runs of Python-path SymmetricTree, N=10k, seed=1337, rs=0; mean and median RMSE converge to 0.19457837 / 0.19457836. Not asserted in automated tests — docs-only anchor. |
| AN-017 | `benchmarks/sprint26/fu2/fu2-gate-report.md` | 101 | `0.17222003` | RMSE | Run `python benchmarks/sprint26/d0/g4_determinism.py` or analogous script at FU-2 config (N=10k, seed=1337, rs=0, iterations=50, depth=6, max_bin=128, grow_policy=SymmetricTree, FU-2 binary) | `2d806d0fa4` | FU-2 G5 determinism: 100 runs; mean/median RMSE 0.17222003 / 0.17222002. Not asserted in automated tests — docs-only anchor. Represents current production-tip Python-path SymmetricTree output. |
| AN-018 | `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md` | 31 | `0.48047778` | RMSE | Build bench_boosting at commit `0f992cf863` or nearby S19 tip then: same 50k/RMSE/d6/128b/seed42 config | `0f992cf863` | Parity gate value at S19 intermediate tip (post-EvalAtBoundary removal, pre-T1); used as parity reference during A1 production port. Also appears in sprint21 d0_attribution.md run output. Not asserted in live tests. |

---

## Ambiguous / Needs-T3-Judgment entries

**AN-015 (HIGH AMBIGUITY):** `test_qa_round12_sprint9.py` asserts that `.github/workflows/mlx_test.yaml` contains the strings `0.59795737` and `0.95248461`. The workflow file is named `mlx-test.yaml` (with a hyphen, not underscore), so the fixture always skips. These values appear to be Sprint 9 bench_boosting baselines that were supposed to be embedded in CI but never were — the test guards a dead integration. T3 should decide: (a) these anchors never landed in CI — class-a stale-capture; (b) or they represent a real value that should be compared against current output by resolving the broken fixture path.

**AN-011 (BULK ENTRY):** 16 additional bench_boosting config losses from `docs/sprint19/results.md`. These are the DEC-008 reference values used for ULP-comparison gates (not numeric equality assertions in tests). They are "captured outputs" in the doc sense but the live gate mechanism compares pre vs post kernel ULP, not doc-vs-run. Treating as docs-only; T2 could re-run the 18-config sweep to check drift.

**AN-012 (SUPERSEDED KERNEL):** Sprint 18 parity values (`7ab4e8e804`) are from a different kernel version than v5. They are not asserted anywhere — they document the S18 parity sweep result. Not a live regression risk unless someone references them for a v5 comparison. T3: likely class-a documented-supersession.

**AN-013 (SUPERSEDED PATH):** Sprint 17 parity values use `csv_train` subprocess with CSV files, not the bench_boosting+synthetic-data pipeline. These anchor values are not asserted in any live test and are from a kernel predating DEC-016/T1. T3: class-a documented-supersession.

**AN-014 (CATBOOST_MLX 0.3.0):** `m3_max_128gb.md` losses were captured with catboost_mlx 0.3.0 (Sprint 14), predating DEC-028/DEC-029 by 12+ sprints. Not asserted in tests. T3: class-a stale-capture.

---

## Summary statistics

| Dimension | Count |
|-----------|-------|
| Total anchors | 18 |
| Live test assertions (code throws on mismatch) | 9 (AN-001…007, AN-015 × 2 values, AN-006, AN-007) |
| Docs-only anchors (no automated enforcement) | 9 (AN-008…014, AN-016…018) |
| RMSE kind | 9 |
| Prediction / proba kind | 4 |
| Logloss / loss (binary/multiclass) | 5 |
| Predating DEC-028 merge (`634a72134d`) | 13 (AN-006–018; all captured before S26-D0-9) |
| Touched Sprint 16+ (sha order) | 7 (AN-006 via f804c, AN-009–012 via S19/S18 commits, AN-016–017 via S26) |
| DEC-028 anchor-update group (AN-001–005) | 5 (updated AT 634a72134d, i.e. these ARE the DEC-028-aware values) |

---

## Top 5 highest-risk anchors

1. **AN-015** — `test_qa_round12_sprint9.py:655,661` (`0.59795737`, `0.95248461`). Fixture always skips due to `mlx_test.yaml` vs `mlx-test.yaml` naming. Zero enforcement for 15+ sprints. Unknown which pipeline configuration generated these values or whether current code produces them.

2. **AN-008** — `CHANGELOG.md:41` (`1.78561831`, K=10). Documented-only, not asserted in any test. K=10 multiclass exercises `ComputeLeafValues` fused path (DEC-019) and the S18 `kHistMultiByte` tile. Any regression in the multiclass fused leaf path would be invisible. Last touched SHA `fd04d34684` predates S19/S22/S24 kernel changes.

3. **AN-009** — `docs/sprint19/results.md:41` (`0.48231599`, config #8 bimodal). This is the DEC-023 Value A anchor. v5 kernel fixed the bimodality; this value should be deterministic on current master. However, it is not asserted in any live test — only appears in sprint docs and `.claude/state/`. A regression in the SIMD-shuffle path (e.g. from a Metal compiler version change) would go undetected.

4. **AN-006** — `python/tests/test_qa_round10_sprint5_bench_and_scan.py:66` (`0.11909308`). Tolerance `1e-6` (very tight). Requires `/tmp/bench_boosting` binary — test skips if not built. This is the only active bench_boosting loss assertion; captured post-BUG-002 (S5). Does not exercise the Python/nanobind path. If bench_boosting is not rebuilt from current source before running, a stale binary can produce a false pass.

5. **AN-017** — `benchmarks/sprint26/fu2/fu2-gate-report.md:101` (`0.17222003`). Most recent production-tip Python-path SymmetricTree loss anchor (S26-FU-2, `2d806d0fa4`). Not enforced by any automated test. This is exactly the kind of value needed for G2-AA gate validation — but it's only in a markdown table. It should become an AN that T2 re-runs.
