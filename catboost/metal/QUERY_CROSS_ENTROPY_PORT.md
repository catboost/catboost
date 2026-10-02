# QueryCrossEntropy translation

The GPU statistics and full leaf Hessian are implemented in
[`native/metal_query_cross_entropy_kernels.h`](native/metal_query_cross_entropy_kernels.h).
The target component has 33 passing M3 tests, including an independent constrained optimum,
an explicit pair-matrix reference, finite-difference curvature, empty leaves,
single-class queries, zero weights, extreme logits, and native input boundaries.
Numeric Plain/DocParallel P1 training is now connected through the native
CatBoost GPU API and standalone CatBoostMetalRanker. Full CUDA parity remains
incomplete; categorical/P4 histories remain pending. The installed build
also supports Simple leaves; see SIMPLE_LEAVES_PORT.md.

## CUDA correspondence

- `cuda/targets/kernel/query_cross_entropy.cu`: query bias optimization, point
  derivatives, shifted derivatives, group sums, and pair curvature.
- `cuda/targets/query_cross_entropy.h`: maximum query size is 256.
- `private/libs/options/loss_description.cpp`: default alpha is 0.95; raw-value
  scale is selected by query size and count of targets greater than 0.5.

Each group uses CUDA's eight bisection iterations and five safeguarded Newton
iterations to optimize a shift within [-20, 20]. Queries whose targets differ
from the first target by at most 1e-5 use only the point-loss component. GPU
probabilities retain CUDA's [1e-7, 1-1e-7] clipping. The scalar loss uses stable
softplus, retaining small positive losses that a direct `log(1+exp(x))` loses.

The 16-byte `QCEParams` contains rows, groups, leaves, and alpha. The statistics
kernel takes feature-independent targets, effective object/group weights, raw
predictions, contiguous group offsets, and one already-selected scale per group.
It writes:

| Buffer | Shape | Fields |
|---|---|---|
| Row statistics | float32[N,4] | negative gradient, point curvature, shifted curvature, weighted loss |
| Group statistics | float32[Q,4] | fitted shift, shifted-curvature sum, weighted loss sum, weight sum |
| Single-class flags | uint32[Q] | CUDA single-class decision |

QueryCrossEntropy has a **full query Hessian**. The leaf projection adds the
point diagonal and the query Laplacian. For shifted curvature sums A_l within
leaf l and S across the query, its diagonal contribution is
A_l*(S-A_l)/(S+1e-20); an off-diagonal entry is -A_l*A_m/(S+1e-20).
The query term is zero when S <= 1e-20. Computing the complement separately
preserves exact cancellation when all query rows occupy one leaf.

This matches CUDA's explicit edge weights d_i*d_j/(S+1e-20) while avoiding an
O(query_size squared) edge allocation. `CacheQueryCrossEntropyLeafSums` caches
gradient, point curvature, shifted curvature, and its separately accumulated
complement for each query/leaf. `SumQueryCrossEntropyLeafMatrix` then reduces
these caches into the full matrix with compensated sums across queries. Work
is O(leaves * rows + leaves squared * queries), and the caller can tile the
query cache. All 33 tests also pass through this cached path. The earlier
direct projection remains an independent implementation for diagnostics.
## Connected full trainer

`native/metal_query_cross_entropy_runtime.h` implements the shared
`CBMFullMatrixRuntime` contract alongside PairLogitPairwise. It freezes a whole-query
No/Bernoulli weak target for split search, recomputes original query statistics
for every Newton/trial point, and projects complete point-diagonal plus query
Laplacian matrices. CUDA rejects Gradient leaves and L2 stochastic search for
this target; the other six score enums use curvature. Depth is limited to eight.

Candidate search uses independently bounded candidate/query tiles and persistent
compensated accumulators. Tile changes preserve results bit for bit. The resident
workspace accounts for all its buffers, shrinks both tiles before allocation,
and rejects a target exceeding its 1 GiB budget. Whole-trainer accounting includes
this workspace and reserves shared core buffers before choosing target tiles.
Unused scalar histograms and derivatives are reduced to tiny binding buffers.

The full leaf system retains all coordinates and its absolute mean. Leaf
regularization preserves CUDA's empty-diagonal correction, L2 and non-diagonal
ridge. Updates mask empty leaves by original document weights. AnyImprovement
and Armijo retry rejected steps using CUDA's attempt budget. Invalid trial state
is isolated from the accepted target and can recover without poisoning training.
Shader pipelines are shared across targets and metric evaluations.

Native and Python scale selection preserve first-default/last-entry semantics
without allocating a square table based on an arbitrary user-supplied group size.
Entries beyond 256 are validated but need no storage. Object and query weights
are combined exactly once before entering the runtime. Snapshot identities bind
targets, original weights, boundaries, alpha, scale configuration and cursors.

### Metric correspondence

CUDA's QueryCrossEntropy metric uses the target, including its scale table and
single-class threshold. The shared CPU metric ignores raw_values_scale, uses a
different single-class threshold and a different shift solver. The Metal native
progress path and standalone controller therefore evaluate this metric on GPU.
As in CUDA TTargetFallbackMetric, metric alpha may differ, but metric
use_weights/raw_values_scale parameters do not replace the target's weights or
scale table. The metric controls stopping and best-model selection normally.

### Validation

- 33 statistics/projection component cases and 29 candidate matrix cases.
- 19 runtime cases: actual target-to-candidate projection, original full leaf
  matrices, candidate/query tile invariance, workspace shrinking and invalid-trial recovery.
- 57 complete resident training cases: six score enums, soft labels, scales,
  Newton/backtracking, forced step halving, repeated depth-eight splits and exact resume.
- 32 scale/metric cases, 30 standalone lifecycle cases and 41 native API cases.
- Native tests include prequantized forest comparisons, object/group weights,
  baselines/initial models, snapshots, metric parameters, CBM/JSON and GPU prediction.

Installed checkpoint `20260913T062333Z` passes **5149 tests plus 16 subtests**
and **648 alternate-extension acceptance cases**. Fourteen installed native/
standalone GPU fit/prediction paths pass, plus native CLI QCE training, CBM
loading and installed GPU prediction. The prior PairLogitPairwise checkpoint
remains available for recovery.

One warmed M3 Pro synthetic run trained 20 depth-four trees on 16384 weighted
rows, six features and query size 16 in 2.777 seconds.
On 4096 held-out rows, NDCG@10 improved from 0.144
(constant-score shared tie convention) to 0.980; scaled
QCE fell from 0.659 to 0.184.
GPU and standard model-reader predictions matched exactly. The script and report
are preserved with the wheel. This is local synthetic evidence, not NVIDIA parity.

## Remaining work

- Categorical/P4 query histories; Simple leaf mode is connected in the installed
  build described in SIMPLE_LEAVES_PORT.md.
- Exact CUDA random draw protocol for whole-query sampling (the current stream
  is query-index/absolute-iteration deterministic and snapshot exact).
- Persist evaluation workspaces rather than allocate a temporary target per metric call.
- Remove unused scalar histogram allocations for full-matrix targets and validate
  larger datasets, more query shapes, and performance across M-series devices.
- NVIDIA numerical/quality/performance comparison; none has been run.

The older projection probe's 64-leaf cap is diagnostic only; complete training
and candidate tests cover 256 leaves. No CPU CatBoost fitting was used.

## Short-query performance and shared memory

The optimized checkpoint chooses 32, 64, 128 or 256 threads for query
statistics and per-query candidate caches based on maximum query size. Shared
reduction scratch shrinks with the threadgroup. All 18 adaptive-vs-256 checks
are bit-identical across query-size boundaries and No/Bernoulli sampling.

Three alternating warmed native M3 Pro runs per build used 16384 learn and
4096 validation rows, six features, 20 depth-four trees and three Newton
iterations. Median wall time changed as follows:

| Query size | Previous build | Optimized build | Speed ratio |
|---|---:|---:|---:|
| 16 | 2.718 s | 1.329 s | 2.05x |
| 64 | 0.804 s | 0.500 s | 1.61x |
| 256 | 0.334 s | 0.337 s | 0.99x |

Predictions, leaves, weights and metric histories match bit for bit across
all runs. No CPU training or live NVIDIA comparison was performed. Scripts,
per-run JSON and arrays are in `.build/qce-optimized-benchmark/` and the release.

Full-matrix QCE/PairLogitPairwise no longer allocate scalar histograms or
row derivative arrays. Tiny unused binding buffers remain. The target tile
budget now reserves shared core buffers first. Twelve focused wide-bank cases
train depth-eight forests with 5000 stored features using under 16 MiB and
match narrow-bank outputs exactly. Scalar histograms previously required
2.62 GB for that bank. Checkpoint `20260913T065034Z` is installed: 5179 combined tests plus
16 subtests, 648 alternate acceptance cases, fourteen installed GPU paths
and the rebuilt native CLI fit/CBM/GPU-read check pass.

## Persistent progress metrics

Native and standalone trackers now reuse metric-only GPU workspaces with a
256 MiB retention cap and preserve exact histories. See
[QCE_METRIC_WORKSPACES.md](QCE_METRIC_WORKSPACES.md) for validation and measured
1.17x/1.23x native gains in the four-metric/two-validation workload.

## Native one-hot categories

The native trainer now supports one-hot categorical features through ordinary
Pools and the CLI. CUDA counts learn plus validation categories; CTR history
is still gated for these objectives. See RANKING_ONE_HOT_PORT.md for validation.
