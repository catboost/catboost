# PairLogit pointwise GPU foundation

[`metal_pairwise_kernels.h`](native/metal_pairwise_kernels.h) implements the
given-edge PairLogit objective, per-edge derivatives, and deterministic
original-row aggregation. The persistent target helper in
[`metal_pairwise_runtime.h`](native/metal_pairwise_runtime.h) adds validated
pair metadata, GPU status checks, and final leaf centering. The combined
90 kernel/runtime tests and 64 connected persistent-training tests passed on
the M3 on September 12, 2026. The supplied-pair scalar session, public ranker and standard native CatBoost
adapter are connected, including stored Pool pairs, weighting, snapshots and
GPU inference. NDCG/MAP/PFound selection is documented in
[RANKING_METRICS.md](RANKING_METRICS.md). This does not
implement the separate PairLogitPairwise tree-search/leaf-matrix algorithm.

## CUDA equations and weights

The source is [`pair_logit.cu`](../cuda/targets/kernel/pair_logit.cu), ordinary
`PairLogitPointwiseTargetImpl`. For a supplied winner `u`, loser `v`, edge
weight `w`, difference `d=a[u]-a[v]`, and float32 probability `p=sigmoid(d)`:

```text
positive loss = w * softplus(-d)
winner ascent gradient = +w*(1-p)
loser ascent gradient = -w*(1-p)
both endpoint diagonal curvatures += w*p*(1-p)
```

Supplied edge weights are used **literally**. They are not multiplied by
object or group weights in the CUDA target. Host preparation stores the group
weight separately and copies each supplied pair weight unchanged into its
competitor list; CUDA flattening copies that weight unchanged:
[`MakeGroupInfos`](../private/libs/target/data_providers.cpp),
[`samples_grouping.h`](../cuda/gpu_data/samples_grouping.h), and
[`samples_grouping_gpu.cpp`](../cuda/gpu_data/samples_grouping_gpu.cpp).

Generated pairs are a distinct preparation path. They omit target ties, make
the larger target the winner, and use the processed document weight at the
group's first row as their common weight, unless `ForceUnitAutoPairWeights`
requests one. This is not the product of the two endpoint weights. `max_pairs`
limits generated pairs per group; its brute-force/random-unique paths have
different ordering and no inverse-sampling correction. Supplied pairs bypass
generation and `max_pairs`. See
[`pairs/util.cpp`](../private/libs/pairs/util.cpp) and
[`data_providers.cpp`](../private/libs/target/data_providers.cpp).

Supplied endpoints must differ, lie within the dataset, and belong to the same
group when grouping is present. Edge weights are finite and nonnegative.
Supplied pairs are not deduplicated, reordered by target values, or discarded
because labels tie. Duplicates and reversed pairs remain independent
contributions. A real training target requires positive total edge weight;
diagnostic empty/zero-weight inputs return zeros.

Keep these quantities distinct:

| Quantity | Ordinary PairLogit rule |
| --- | --- |
| Metric denominator | Sum of edge weights |
| Training row weight | Sum of weights of every edge incident on the row |
| Total training row weight | Twice the sum of edge weights |
| Exported leaf weight | Sum of incident row weights inside that leaf |
| Gradient leaf denominator | Sum of incident row weights plus leaf regularization |
| Newton leaf denominator | Sum of projected diagonal curvatures plus leaf regularization |

[`InitPairLogit`](../cuda/targets/querywise_targets_impl.h) explicitly replaces
target row weights with incident pair mass and marks `StorePairWeights`.
Isolated rows therefore receive zero training weight even if their original
object weight was positive. The pointwise oracle exports these replacement
weights, not original object weights or Hessian sums.

## Metal ABI and deterministic incidence layout

The independent MSL string is `CBMMetalPairwiseSource`. Its 32-byte parameters
are `uint rows,pairs,objective,apply_leaf_values,leaves,reserved0,reserved1,reserved2`.
Objective ID `14` is reserved for PairLogit. The scalar `KernelParams` layout
is unchanged.

Build an incidence CSR once from the supplied edge arrays. `row_offsets` has
`rows+1` entries; `incident_edges` and `incident_signs` each have `2*pairs`
entries. Each edge occurs once at its winner with sign `+1` and once at its
loser with sign `-1`, preserving edge order within each row. All point and
derivative buffers retain original row order. Validate dimensions, endpoints,
CSR bounds, weights, and group membership before exposing device pointers.

| Kernel | Buffer bindings | Dispatch |
| --- | --- | --- |
| `PreparePairwisePoint` | 0 cursor; 1 raw unshrunk leaf values; 2 original-row leaf IDs; 3 output point; 4 parameters | `rows` threads |
| `PairLogitEdgeDerivatives` | 0 point; 1 winners; 2 losers; 3 supplied edge weights; 4 output `float4[pairs]`; 5 parameters | `pairs` threads |
| `ReducePairwiseRows` | 0 CSR offsets; 1 incident edge indices; 2 signed incidence; 3 edge values; 4 output gradients; 5 output diagonal curvature; 6 output incident row weights; 7 parameters | `rows` full 256-thread groups |
| `ReducePairwiseObjective` | 0 edge values; 1 output `float2[groups]`; 2 parameters | Any positive count of full 256-thread groups |

The edge tuple is `(winner gradient, edge curvature, positive loss, edge
weight)`. The objective reduction outputs partial loss divided by edge count
and partial edge weight divided by edge count. Sum the small result in host
double. Their ratio gives the metric; multiply by edge count to obtain the
unnormalized oracle loss. Empty edges yield zero partials.

The row kernel uses high/low compensated sums and 8 KiB of threadgroup scratch.
This avoids CUDA float atomic accumulation order and preserves a small
remainder in cancelling incident gradients. The fixed CSR is metadata, not
host model fitting. A later performance pass can optimize low-degree row
dispatch without changing its deterministic contract.

## Persistent training integration

`CBMPairwiseRuntime` receives the trainer's Metal device and encodes into its
uncommitted command buffer. It never commits or waits. It caches its own
pipelines and keeps the CSR, point, edge, objective-partial, mean, and status
buffers alive across tree steps. `AllocatedBytes()` reports the exact GPU
allocation, while `IncidentWeights()` exposes the validated initial row mass
for the shared trainer's preflight and initial buffer upload.

`EncodePointDerivatives` emits point, edge, row, and validation kernels;
`EncodeLossReduction` adds the bounded loss reduction. After the caller waits,
`ReadLossPartials()` returns unnormalized positive loss and edge mass.
`EncodeCenterLeafValues` computes a compensated unweighted mean and subtracts
it on the GPU before shared tree shrinkage. Active leaves may vary up to the
constructor's maximum of 65,536. The extra GPU allocation is
`44*pairs + 8*rows + 4 + 8*loss_groups + 8*ceil(max_leaves/256) + 8` bytes.

GPU validation is sticky until `ClearStatus()` is called with no command in
flight. Nonfinite current points or derivatives are errors. The optional
`allowNonfiniteTrial` / `allowNonfiniteLoss` flags permit a nonfinite candidate
objective to reach backtracking's rejection and step-halving logic; invalid
leaf indices remain hard errors, and the edge-mass denominator must remain
finite and positive.

At initialization, derive incident row weights from original edge weights and
use them throughout ordinary PairLogit training. For each structure-search
target, compute original edge derivatives, aggregate to rows, then apply
object bootstrap to those completed row gradients and structure weights.
Do not sample edges in the ordinary PairLogit path.

Every leaf update and backtracking trial must rebuild the complete original
row point and recompute the complete edge list before projecting derivatives
to current leaves. Edges crossing leaves couple their trial values. Edges
inside one leaf have cancelling projected gradients but still contribute
both diagonal curvatures in this **pointwise** approximation.

Doc-parallel CUDA enables `NeedZeroAverage` for PairLogit and subtracts the
unweighted arithmetic mean of **all** solved leaf values, including empty
leaves, before tree shrinkage/export. See
[`train_template.h`](../cuda/train_lib/train_template.h) and
[`doc_parallel_leaves_estimator.cpp`](../cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.cpp).
The separate feature-parallel source appears to calculate its bias before
copying solved values into the leaf buffer; this foundation targets the
doc-parallel rule and does not claim identical behavior across those paths.

Default PairLogit leaf estimation is Newton with ten iterations; its Gradient
default is forty. Pair metadata and weights must enter dataset/snapshot
fingerprints. Shared Pool preparation should supply generated pairs and group
validation; this kernel does not synthesize rankings or pairs from labels.

## PairLogitPairwise remains separate

The pairwise variant uses `EOracleType::Pairwise` and retains one curvature
per edge. It clips probabilities to `[1e-7,1-1e-7]`, unlike the pointwise
kernel's near-zero lower bound. It bootstraps **edge weights**, filters zero
edges, and then computes derivatives. It does not replace exported document
weights with incident pair mass.

Its support-pair builder removes same-leaf edges and groups the remaining
edges by ordered leaf pair. Each inter-leaf curvature contributes `+h` to both
diagonals and `-h` to both off-diagonals of a leaf Laplacian. Gradient estimation
uses pair weights in place of curvatures. The pure pairwise oracle fixes the
last leaf at zero, solves `leaves-1` coordinates, and later centers leaves.

Leaf-matrix regularization adds off-diagonal `-NonDiagLambda/leaves` and
diagonal `NonDiagLambda*(1-1/leaves)+Lambda`; a zero unregularized diagonal
also receives `10`. Split scoring uses additional, different diagonal
stabilization and Cholesky rules. These must be translated separately rather
than substituting the pointwise diagonal solver. Sources:
[`pair_logit_pairwise.cpp`](../cuda/targets/pair_logit_pairwise.cpp),
[`pairwise_oracle.h`](../cuda/methods/leaves_estimation/pairwise_oracle.h),
[`leaves_estimation_helper.cpp`](../cuda/methods/leaves_estimation/leaves_estimation_helper.cpp),
[`matrix_per_tree_oracle_base.h`](../cuda/methods/leaves_estimation/matrix_per_tree_oracle_base.h),
and [`linear_solver.cu`](../cuda/methods/kernel/linear_solver.cu).

## Numerical adaptations and evidence

Metal uses stable sigmoid and softplus algebra. The float32 probability still
saturates at one, so confident winners can have zero pointwise gradient and
curvature; stable softplus retains their small positive loss. This differs
numerically from CUDA's subtractive `d-log(1+exp(d))` loss. Subnormal handling
and deterministic compensated reduction also preclude bitwise CUDA claims.
The connected leaf oracle evaluates probability and edge derivatives in
float32 before accumulating rows in double. A float64 sigmoid rounded only
at its output is insufficient near saturation: a one-ULP probability change
can materially change a tiny Hessian and a weakly regularized Newton step.

[`test_pairwise_kernels.py`](tests/test_pairwise_kernels.py) compiles
[`pairwise_probe.mm`](tests/pairwise_probe.mm) and checks independent objective
equations, derivative finite differences, duplicate/reversed edges, 65,539
high-degree edges, edge-order and launch changes, all permutations of a
`+2^24,+1,-2^24` cancellation fixture, logits through ±10,000, disconnected
component shifts, cross-leaf trial values, same-leaf diagonal behavior,
incident weight totals, empty/zero edges, and invalid inputs. No CPU model is fit.

[`test_pairwise_runtime.py`](tests/test_pairwise_runtime.py) adds 63 actual
Metal cases for reusable command transactions, group validation, literal
supplied weights, compensated centering through 65,536 leaves, sticky error
recovery, permissive backtracking candidates, and exact dispatch/allocation
accounting. The original 27 kernel cases also pass with the expanded shader.

[`test_pairwise_training.py`](tests/test_pairwise_training.py) adds 64 connected
cases: independent split and iterative leaf equations, Newton/Gradient, four
standard/Newton split scores, all three backtracking modes, all five bootstrap
types, original incident leaf mass, and exact resumed sampling/noise/cursor
state including automatic MVS. Its extreme-logit fixture requires seven
rejected trials before acceptance at `1/128`, despite a two-attempt nominal
budget. The standard native model adapter and public lifecycle are tested
separately when that integration lands.
