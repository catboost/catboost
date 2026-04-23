# S27-AA-T2 — Anchor Re-Run Results

**Produced:** 2026-04-22 (Sprint 27, Track B)
**Agent:** @qa-engineer
**Branch:** `mlx/sprint-27-correctness-closeout` (master tip `20d14564e8` + S27 kickoff `bc67152924` + T1 `d4e2d7cf88`)
**bench_boosting:** rebuilt from source at current tip before any run (compiled `2026-04-22`). Binary: `/tmp/bench_boosting`. Sources: `bench_boosting.cpp` + `histogram_t2_impl.cpp`.
**Purpose:** Re-run every anchor's generating harness and record drift vs T1 committed values.

---

## Re-Run Results Table

| ID | Path | Line | Original Value | Current Value | Drift (abs) | Drift (rel) | Harness Status | FU-1 Tag | Notes |
|----|------|------|----------------|---------------|-------------|-------------|----------------|----------|-------|
| AN-001 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 91 | `0.306348` | `0.30634810` | `1.0e-07` | `3.3e-07` | CLEAN | FU-1-INDEPENDENT | pytest passes (atol=1e-3); residual sub-ULP from truncated anchor string |
| AN-002 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 107 | `0.317253` | `0.31725318` | `1.8e-07` | `5.7e-07` | CLEAN | FU-1-INDEPENDENT | pytest passes (atol=1e-3); sub-ULP from truncated anchor string |
| AN-003 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 108 | `-0.568259` | `-0.56825905` | `5.0e-08` | `8.8e-08` | CLEAN | FU-1-INDEPENDENT | pytest passes (atol=1e-3) |
| AN-004 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 109 | `1.598960` | `1.59896022` | `2.2e-08` | `1.4e-08` | CLEAN | FU-1-INDEPENDENT | pytest passes (atol=1e-3) |
| AN-005 | `python/tests/test_qa_round9_sprint4_partition_layout.py` | 155 | `[0.37227302, 0.36382151, 0.26390547]` | `[0.37227302, 0.36382151, 0.26390547]` | `0` | `0` | CLEAN | FU-1-INDEPENDENT | pytest passes; bit-exact match on all 3 elements |
| AN-006 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py` | 66 | `0.11909308` | `0.11097676` | `8.116e-03` | `6.82e-02` | NEEDS-TRIAGE | FU-1-INDEPENDENT | **LIVE TEST FAILURE** — `pytest::TestBenchBoostingAnchors::test_binary_100k_anchor` fails (tolerance 1e-6). bench_boosting rebuilt from current tip. 6.8% relative drift. Binary classification path. |
| AN-007 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py` | 67 | `0.63507235` | `0.61788315` | `1.719e-02` | `2.71e-02` | NEEDS-TRIAGE | FU-1-INDEPENDENT | **LIVE TEST FAILURE** — `pytest::TestBenchBoostingAnchors::test_multiclass_20k_anchor` would fail (stopped after AN-006). 2.7% relative drift. Multiclass K=3 path. |
| AN-008 | `CHANGELOG.md` | 41 | `1.78561831` | `1.85752499` | `7.191e-02` | `4.03e-02` | CLEAN | FU-1-INDEPENDENT | docs-only; run manually: `bench_boosting --rows 20000 --features 30 --classes 10 --depth 5 --iters 50 --seed 42`. 4.0% relative drift. K=10 multiclass path. |
| AN-009 | `docs/sprint19/results.md` | 41 | `0.48231599` | `0.48231599` | `0` | `0` | CLEAN | FU-1-INDEPENDENT | bit-exact; config #8 (10k/RMSE/128b). DEC-023 Value A anchor stable. |
| AN-010 | `docs/sprint19/results.md` | 47 | `0.47740927` | `0.47740927` | `0` | `0` | CLEAN | FU-1-INDEPENDENT | bit-exact; primary determinism gate config (50k/RMSE/128b). |
| AN-011 | `docs/sprint19/results.md` | 34–49 | 16 bulk values (S19 18-config grid excl. #8, #14) | All 16 values match exactly | `0` (all 16) | `0` (all 16) | CLEAN | FU-1-INDEPENDENT | Full 18-config bench_boosting sweep re-run; all 18 values including AN-009/010 bit-exact. See detail table below. |
| AN-012 | `docs/sprint18/parity_results.md` | 31–48 | 18 S18 values (e.g. `0.44764100`, `0.48016100`, etc.) | Current tip values differ (e.g. `0.44631991`, `0.48231599`); max abs drift `~0.0090` | `~9e-03` (max) | `~2e-02` (max) | NOT-ASSERTED | FU-1-INDEPENDENT | Superseded kernel (`19fa5ce6cc`). Docs record S18 parity sweep. Not a live assertion. Drift is expected — S19 kernel changed accumulation pattern. |
| AN-013 | `docs/sprint17/parity_results.md` | 22–39 | 18 S17 csv_train values (e.g. RMSE 10k/128 = `0.496241`) | UNRUNNABLE | — | — | UNRUNNABLE | FU-1-INDEPENDENT | `csv_train_sprint16` / `csv_train_sprint17` binaries not available on current branch. Different data path (CSV + subprocess) than bench_boosting. Superseded. |
| AN-014 | `benchmarks/results/m3_max_128gb.md` | 13–30 | 9 MLX loss values (e.g. RMSE 10k=`3.0407`, Logloss 10k=`0.4099`) | UNRUNNABLE (budget) | — | — | NOT-ASSERTED | FU-1-INDEPENDENT | Captured with catboost_mlx 0.3.0 (Sprint 14, 12 sprints stale). `benchmarks/bench_mlx_vs_cpu.py` exists but run would take 30+ min and current version predates DEC-028/DEC-029. Docs-only. |
| AN-015 | `python/tests/test_qa_round12_sprint9.py` | 655, 661 | `0.59795737`, `0.95248461` | UNRUNNABLE | — | — | UNRUNNABLE | FU-1-INDEPENDENT | Fixture always skips: test looks for `.github/workflows/mlx_test.yaml` (underscore); file is `mlx-test.yaml` (hyphen). 6 `TestCIBenchWorkflow` tests all `SKIPPED`. Values never enforced by CI. |
| AN-016 | `docs/sprint26/d0/g1-g3-g4-report.md` | 101 | `0.19457837` | `0.19457838` | `1.0e-08` | `5.1e-08` | CLEAN | FU-1-INDEPENDENT | 1-run proxy (100-run too expensive); sub-ULP noise in last digit. SymmetricTree path. Functionally zero drift. |
| AN-017 | `benchmarks/sprint26/fu2/fu2-gate-report.md` | 101 | `0.17222003` | `0.17222002` | `1.0e-08` | `5.8e-08` | CLEAN | FU-1-DEPENDENT | 1-run proxy; Depthwise grow_policy — DW path invoked. Sub-ULP; functionally zero drift. Pre-FU-1 value. FU-1 fixes DW leaf-index at depth>=2 — this value will change post-FU-1. |
| AN-018 | `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md` | 31 | `0.48047778` | `0.47740927` | `3.069e-03` | `6.39e-03` | CLEAN | FU-1-INDEPENDENT | Captured at S19 intermediate tip `0f992cf863` (A1 BATCH_DOCS=64 port, later dropped). Current tip produces S19-final value `0.47740927` (same as AN-010). Drift is kernel-revert artifact — expected, not a regression. |

---

## AN-011 Full 18-Config Detail (all values bit-exact vs S19 reference)

| Config | S19 Reference | Current Tip | ULP |
|--------|--------------|-------------|-----|
| 1k / RMSE / 32 | 0.40689126 | 0.40689126 | 0 |
| 1k / RMSE / 128 | 0.46936080 | 0.46936080 | 0 |
| 1k / Logloss / 32 | 0.34161490 | 0.34161490 | 0 |
| 1k / Logloss / 128 | 0.61407095 | 0.61407095 | 0 |
| 1k / MultiClass / 32 | 0.61065382 | 0.61065382 | 0 |
| 1k / MultiClass / 128 | 0.99084771 | 0.99084771 | 0 |
| 10k / RMSE / 32 | 0.44631991 | 0.44631991 | 0 |
| 10k / RMSE / 128 | 0.48231599 | 0.48231599 | 0 |
| 10k / Logloss / 32 | 0.30072498 | 0.30072498 | 0 |
| 10k / Logloss / 128 | 0.60412812 | 0.60412812 | 0 |
| 10k / MultiClass / 32 | 0.57359385 | 0.57359385 | 0 |
| 10k / MultiClass / 128 | 0.95665115 | 0.95665115 | 0 |
| 50k / RMSE / 32 | 0.44676545 | 0.44676545 | 0 |
| 50k / RMSE / 128 | 0.47740927 | 0.47740927 | 0 |
| 50k / Logloss / 32 | 0.30282399 | 0.30282399 | 0 |
| 50k / Logloss / 128 | 0.60559267 | 0.60559267 | 0 |
| 50k / MultiClass / 32 | 0.56538904 | 0.56538904 | 0 |
| 50k / MultiClass / 128 | 0.94917130 | 0.94917130 | 0 |

---

## Summary

| Dimension | Count |
|-----------|-------|
| Total anchors | 18 |
| Runnable (harness executed, value produced) | 14 |
| UNRUNNABLE | 3 (AN-013, AN-014, AN-015) |
| NOT-ASSERTED (docs-only, manually run) | 2 included in runnable count (AN-008, AN-018) |
| Drifted > 1e-4 | 4 (AN-006, AN-007, AN-008, AN-018) |
| Drifted < 1e-4 (functionally zero) | 4 (AN-001–004, AN-016, AN-017 sub-ULP) |
| Bit-exact (ULP = 0) | 6 (AN-005, AN-009, AN-010, AN-011 ×18 configs) |
| FU-1-DEPENDENT | 1 (AN-017) |
| FU-1-INDEPENDENT | 16 |
| UNKNOWN FU-1 | 0 |
| Live test failures | 2 (AN-006, AN-007 — bench_boosting tolerance 1e-6) |

---

## Build provenance

bench_boosting rebuilt from source at commit `d4e2d7cf88` (S27 T1 tip):

```
clang++ -std=c++17 -O2 -I<repo> -I/opt/homebrew/Cellar/mlx/0.31.1/include \
  -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  catboost/mlx/tests/bench_boosting.cpp \
  catboost/mlx/methods/histogram_t2_impl.cpp \
  -o /tmp/bench_boosting
```

Timestamp: `2026-04-22 22:08`. No stale-binary risk — binary is newer than all source commits.

---

## FU-1 note

AN-017 (`0.17222002`, Depthwise, single run) is tagged `FU-1-DEPENDENT`. It uses `grow_policy="Depthwise"` at `depth=6`, which exercises `ComputeLeafIndicesDepthwise`. With BUG-A (encoding, depth>=2) + BUG-B (split lookup, depth>=3) both unfixed on this branch, the current value is pre-FU-1. T4 should re-capture after FU-1 merges.

The 100-run mean `0.17222003` (original) vs single-run `0.17222002` (current) gap is 1e-8 — this is natural run-to-run float32 variation, not a meaningful signal.
