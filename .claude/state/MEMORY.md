# Project Memory

<!-- Agents append learnings here. Do not delete entries — only correct errors. -->
<!-- Format: - [YYYY-MM-DD] <concise learning> -->
<!-- Max 200 lines. Consolidate older entries when approaching the limit. -->

## Codebase Patterns

- [2026-04-09] `train_lib/train.cpp` is the C++ loss dispatch entry point. Loss names are matched by string in `ParseLossType`. Case folding must happen before this call — the function has no tolerance for uppercase variants.
- [2026-04-09] Python `_validate_params` and `_build_train_args` are the two places that must stay in sync when adding a new loss. `_validate_params` does named-param extraction and range checks; `_build_train_args` builds the binary CLI string. BUG-001/BUG-002 both originated from these two sites being out of sync.
- [2026-04-09] `csv_train` binary has its own loss dispatch, separate from `train_lib/train.cpp`. When wiring a new loss, update both. They can diverge silently (e.g., `csv_train --loss huber` defaults delta=1.0 while `train.cpp` CB_ENSUREs it is specified).
- [2026-04-09] `kLeafAccumSource` Metal kernel has a compile-time `MAX_LEAVES=64` constant. This means `max_depth > 6` (which yields 128 leaves) is not supported. A runtime `CB_ENSURE` was added in `928c7ff` to guard this — do not remove it without replacing the kernel.

## Gotchas & Pitfalls

- [2026-04-09] IDE clang-language-server reports hundreds of false positive diagnostics for CatBoost includes. This is an IDE path configuration issue — the actual build works. Do not react to these diagnostics.
- [2026-04-09] CatBoost canonical loss param syntax `LossName:param=value` (e.g., `Quantile:alpha=0.7`) causes Python `float()` to fail if the `param=` prefix is not stripped first. Always strip before parsing numeric params.
- [2026-04-09] `ComputePartitionLayout` is still on CPU as of Sprint 3 close. This is the #1 identified sync bottleneck (MLOps audit). Do not attempt GPU tree search optimizations without addressing this first — it dominates the per-iteration CPU-GPU round trip cost.

## Dependencies & Tools

- [2026-04-09] Python test suite entry point: `python/tests/`. QA test naming convention: `test_qa_round<N>_<topic>.py`. Sprint 3 tests are in `test_qa_round8_sprint3_losses.py`.

## Performance Notes

- [2026-04-09] Baseline before Sprint 3 OPTs: multiclass K=3 micro-benchmark = 1.07s.
- [2026-04-09] After OPT-1 (fuse leaf sum dispatch) + OPT-2 (bin-to-feature lookup precompute): 0.98s (~8% speedup).
- [2026-04-09] EvalNow call count in `SearchTreeStructure` (static audit): binary = 6 per depth level, multiclass K = 4 + 2K per depth level. Total across all methods/ files: 13.
- [2026-04-09] Parallel scan for `suffix_sum_histogram` (currently 1-thread serial) estimated at 5–32x potential speedup. Deferred to Sprint 4 (TODO-008).

## Testing Notes

- [2026-04-09] Test suite count at Sprint 3 close: 604 passing. Pre-Sprint-3 baseline: 558. Net gain from Sprint 3: +46 tests.
- [2026-04-09] csv_train regression smoke test runs at 0.30s and produces final loss 0.596. Use this as the regression baseline when touching train_lib or csv_train.
