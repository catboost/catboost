# Persistent QueryCrossEntropy metric workspaces

Checkpoint `20260913T093443Z` is installed: 6052 combined tests +16 subtests,
846 alternate native acceptance, 36 installed GPU paths and eight CLI variants.
Wheel SHA256 `acb876f7fcfbdbf28ba0bb9a71abeceab25e04ae45e07df756d3266001856f5e`.

The previous progress tracker rebuilt the QueryCrossEntropy target, query
metadata and GPU solver buffers on every metric evaluation. Metrics now use
only target/weight/point/statistic/query/reduction buffers, with no tree-search
or matrix-solver allocations. An opaque metric handle copies immutable
targets, weights, offsets and scales once, then evaluates new predictions and
alpha values using the same CUDA-derived kernels and arithmetic.

Each workspace serializes calls, reports exact allocation and successful-call
counts, and resets GPU error status before evaluation. Closing a handle removes
it from the registry; already-started calls retain ownership until completion.
Failed creation, invalid inputs, overflowed points, reused/closed handles and
parallel workspaces are covered by actual GPU tests.

Native progress keeps at most 256 MiB of metric GPU buffers across its learn
and validation datasets. Larger/overflowing entries are evaluated once and
released. Standalone QCE retains the applicable learn or validation workspace
under the same cap and closes it through ExitStack on success, callback errors
and recovery. This retention cap is separate from training's 1 GiB workspace
budget; individual uncached metrics retain their existing 1 GiB bound.

The target's original weights and scale table remain fixed. Metric alpha can
change per call; CUDA target fallback semantics still override metric-specific
use_weights/scale requests. The default standalone validation loss and identical
selection metric are evaluated once. Workspaces are ephemeral, so snapshots
continue storing optimizer/data/metric history rather than GPU allocations.

40 new API cases and ten cache/lifecycle cases pass, including native multiple
validation datasets and alpha variants, original target semantics, exact
snapshot replay, allocation counts and cleanup after exceptions. The focused
run has 121 passing cases, followed by the complete 6052 +16 regression and
846 alternate acceptance. All 36 installed smoke paths and eight native CLI
variants pass (six YetiRankPairwise plus QCE Simple/Newton with custom metrics).

Three alternating warmed native M3 Pro runs used 4096 learn rows, two 1024-row
validation sets, four features, 20 depth-3 Simple trees and four metric alphas:

| Query size | Previous median seconds | Reused median seconds | Speedup |
|---|---:|---:|---:|
| 16 | 0.2527 | 0.2160 | 1.17x |
| 64 | 0.1576 | 0.1284 | 1.23x |

Every prediction, leaf, weight and metric-history byte is identical to the
previous native build. Arrays, timings, scripts, both extensions, CLI, source
hashes and logs are preserved. These are Metal-to-Metal workload measurements;
CUDA/NVIDIA equivalence is not established. Prior wider-ID checkpoint
20260913T091701Z remains intact, as do earlier subgroup and sparse checkpoints.
Full CUDA RNG, categorical/P4 ranking, grouped Ordered, estimated features and
other documented parity work remain open. No CPU CatBoost fitting occurred.
