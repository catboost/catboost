# Sync-Point Inventory — CatBoost-MLX (S16-04)

Every `EvalAtBoundary`, `.data<T>()`, and `.item<T>()` call in the MLX training
path, with frequency classification and sprint assignment.

**Legend**
- **Frequency**: `per-iter` = once per boosting iteration; `per-depth` = once per tree depth level; `per-leaf` = once per lossguide leaf expansion; `one-time` = outside the training loop
- **Cost class**: `A` = high (blocks GPU drain, forces full graph materialize); `B` = medium (CPU readback, but small array); `C` = low (boundary sync, unavoidable until restructuring)
- **Sprint**: which sprint is planned to remove or move this sync

---

## `catboost/mlx/methods/mlx_boosting.cpp`

| File:Line | Call | Frequency | Arrays | Cost | Notes | Sprint |
|-----------|------|-----------|--------|------|-------|--------|
| `mlx_boosting.cpp:88` | `EvalAtBoundary({trainData.GetCursor()})` | per-iter | cursor [approxDim × numDocs] | C | Iteration boundary sync. Bounds lazy graph depth to O(1 iter). Must stay — prevents unbounded memory accumulation. | Keep |

---

## `catboost/mlx/methods/structure_searcher.cpp`

### Depthwise path

| File:Line | Call | Frequency | Arrays | Cost | Notes | Sprint |
|-----------|------|-----------|--------|------|-------|--------|
| `structure_searcher.cpp:275` | `EvalAtBoundary({allGradSums, allHessSums})` | per-depth | [approxDim × numPartitions] float32 | A | Required: `.data<float>()` readback follows immediately for per-partition split. Sprint 19 will eliminate this via GPU-resident split selection kernel. | Sprint 19 |

### Lossguide path (inside `evalLeaf` lambda)

| File:Line | Call | Frequency | Arrays | Cost | Notes | Sprint |
|-----------|------|-----------|--------|------|-------|--------|
| `structure_searcher.cpp:593` | `EvalAtBoundary({allGradSums, allHessSums})` | per-leaf | [approxDim × 2] float32 | A | Required: `.data<float>()` readback follows for partition-0 sum extraction. Sprint 19 candidate. | Sprint 19 |
| `structure_searcher.cpp:653` | `EvalAtBoundary(compressedData)` | per-leaf | [numDocs × numUi32PerDoc] uint32 | A | Required: CPU walks `dataPtr[]` per-doc in the lossguide split-apply loop. Largest single-call cost. Sprint 20 (CPU-loop elimination). | Sprint 20 |
| `structure_searcher.cpp:686` | `EvalAtBoundary(result.LeafDocIds)` | per-iter (lossguide only) | [numDocs] uint32 | B | Lossguide exit boundary. Ensures `ApplyLossguideTree` receives a materialized GPU array for `mx::take`. Sprint 20 candidate. | Sprint 20 |

---

## `catboost/mlx/methods/score_calcer.cpp`

| File:Line | Call | Frequency | Arrays | Cost | Notes | Sprint |
|-----------|------|-----------|--------|------|-------|--------|
| `score_calcer.cpp` (FindBestSplitGPU) | `mx::eval({scoreResult[0], scoreResult[1], scoreResult[2]})` | per-depth | [numBlocks] float32 × 3 | B | Required: CPU reduction over block candidates follows immediately (lines after eval). Sprint 17 will shrink numBlocks by improving occupancy. The readback itself is unavoidable until a full GPU argmax is implemented. | Sprint 21 |

---

## `catboost/mlx/tests/csv_train.cpp` (standalone — not part of library path)

| Location | Call | Frequency | Notes |
|----------|------|-----------|-------|
| Multiple inline usages | `mx::eval(...)` | per-depth / per-iter | csv_train is a fully standalone reimplementation. It is not instrumented by the stage profiler. Production training uses the library path (mlx_boosting.cpp + structure_searcher.cpp). |

---

## Summary counts (library path, `RunBoosting` + helpers)

| Scope | EvalAtBoundary count | Avg array size | Cost class |
|-------|---------------------|----------------|------------|
| Per-iter boundary (oblivious/depthwise) | 1 | large | C (keep) |
| Per-depth depthwise (lines 275) | 1 | small | A → Sprint 19 |
| Per-leaf lossguide (lines 593, 653, 686) | up to 3 × (maxLeaves-1) | medium–large | A/B → Sprint 19-20 |
| Per-depth scoring readback (score_calcer) | 1 | tiny | B → Sprint 21 |

**Sprint 16 result**: Removed 18 `EvalNow` calls from `pointwise_target.h` (per-iter).
Remaining syncs are either (a) loop-boundary (unavoidable) or (b) CPU-readback-driven
(require structural changes in Sprints 19-21 to eliminate).

---

## Removed syncs (S16-07)

The following 18 `EvalNow` calls were removed in Sprint 16 (sync-storm fix, S16-07).
All were in `catboost/mlx/targets/pointwise_target.h`, one pair (grad+hess) per target class.
They are listed here for historical attribution only.

| Target class | Former line | Former call |
|--------------|-------------|-------------|
| RMSE | ~35, ~48 | `EvalNow({gradients, hessians})`, `EvalNow(loss)` |
| Logloss | ~84, ~105 | same pattern |
| MultiClass | ~167, ~212 | same pattern |
| MAE | ~261, ~271 | same pattern |
| Quantile | ~300, ~314 | same pattern |
| Huber | ~347, ~364 | same pattern |
| Poisson | ~399, ~413 | same pattern |
| Tweedie | ~456, ~476 | same pattern |
| MAPE | ~513, ~526 | same pattern |

**Replacement**: one `EvalAtBoundary({trainData.GetCursor()})` at `mlx_boosting.cpp:88`
(top of per-iteration loop). This bounds graph depth with a single sync instead of 18.
