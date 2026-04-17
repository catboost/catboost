# Handoff — CatBoost-MLX

> Last updated: 2026-04-16 by orchestrator (Sprint 16 sync-storm validation complete)

## Current state

- **Branch**: `mlx/sprint-16-perf-diagnosis`
- **Last commit**: `da66ce5447` (Sprint 16 state files + docs + bench harness + sync-storm fix)
- **Campaign**: Operation Verstappen — multi-sprint performance domination push (Sprints 16–24)

## What just happened

Sync-storm fix (S16-07) fully validated:
- All 18 `EvalNow` calls removed from `pointwise_target.h`
- 3 per-depth `EvalNow` removed from `structure_searcher.cpp`
- Remaining EvalNow renamed to `EvalAtBoundary` with `[[deprecated]]` alias
- One `EvalAtBoundary({trainData.GetCursor()})` added at iteration boundary in `mlx_boosting.cpp`
- **Numerical parity**: bit-exact loss across RMSE/Logloss/MultiClass × 1k/10k/50k (verified Sprint 15 vs Sprint 16 binaries)
- **Performance**: zero regression (within 1% noise across all 9 test cases)

Baseline regenerated with accurate numbers (old phase_a data was stale from early sprints):
- MLX is **100–300x slower** than CPU CatBoost (not 10–24x as previously reported)
- Per-iteration cost barely scales with N: ~300ms at 1k, ~323ms at 10k, ~487ms at 50k (50 features)
- Tree search (histogram kernel) accounts for 99%+ of per-iteration time
- Confirms `maxBlocksPerPart=1` histogram occupancy as the #1 bottleneck

## Active Sprint 16 work

| Agent | Task | Status |
|-------|------|--------|
| @mlops-engineer | S16-05/06: bench harness extension + CI perf gate | DONE |
| @ml-engineer | S16-07: sync-storm elimination | DONE — validated, committed |
| @qa-engineer | S16-08: numerical parity validation | DONE — bit-exact across all 9 combos |
| @performance-engineer | S16-01/02/04: stage profiler + baseline + sync inventory | Code drafted (in agent output), needs to be written to disk |
| @devops-engineer | CMake `CATBOOST_MLX_STAGE_PROFILE` option | DONE — committed |
| @technical-writer | S16-11: docs/sprint16/ skeleton + ARCHITECTURE.md | DONE — committed |
| @research-scientist | S16-03: Metal System Trace overnight capture | Pending dispatch |

## Bug found

`bench_mlx_vs_cpu.py` used `n_bins=` parameter but API accepts `bins=`. Fixed in working tree.

## Blockers

None.

## Next action

1. Write stage profiler code to disk (from @performance-engineer output)
2. Capture baseline stage profiles
3. Dispatch @research-scientist for Metal System Trace
4. Commit bench_mlx_vs_cpu.py fix + updated baseline
5. Final Sprint 16 PR
