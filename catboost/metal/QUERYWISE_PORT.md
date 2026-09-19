# QueryRMSE and QuerySoftMax translation

This is an implementation note for the grouped Metal derivative kernels. The
grouped kernels, leaf-projection helpers and curvature validator passed 90 numerical tests on the
local Apple GPU on September 12, 2026. QueryRMSE and QuerySoftMax are now connected
to standalone and native Metal training, including weighted metrics, sampling,
snapshots and validation. NDCG/MAP/PFound selection is documented in
[RANKING_METRICS.md](RANKING_METRICS.md). This does not claim complete ranking
support or CUDA/M-series performance parity.

## CUDA source and equations

The translation follows the CUDA target implementation, rather than fitting a
CPU model as a reference:

- [`query_rmse.cu`](../cuda/targets/kernel/query_rmse.cu),
  [`query_helper.cu`](../cuda/gpu_data/kernel/query_helper.cu), and
  [`kernel.h`](../cuda/targets/kernel.h) construct weighted residual means and
  QueryRMSE derivatives.
- [`query_softmax.cu`](../cuda/targets/kernel/query_softmax.cu) computes weighted
  target totals, weighted softmax probabilities, derivatives, and objective.
- [`querywise_targets_impl.h`](../cuda/targets/querywise_targets_impl.h) selects
  query objectives, places bootstrap after derivatives, and defines their
  pointwise leaf oracle.
- [`pointwise_oracle.cpp`](../cuda/methods/leaves_estimation/pointwise_oracle.cpp)
  projects derivatives to leaves and selects Newton/Gradient denominators.
- [`data_providers.cpp`](../private/libs/target/data_providers.cpp) constructs
  effective weights as object weight times group weight before GPU upload.

For each query, let `w` denote those effective original weights, `y` targets,
and `a` raw predictions including the current unshrunk leaf trial.

| Quantity | QueryRMSE | QuerySoftMax |
| --- | --- | --- |
| Query statistic | `mu = sum(w*(y-a))/sum(w)`, zero when weight sum is zero | `A = sum(w*y)`; `p_i = w_i*exp(beta*a_i)/sum(w*exp(beta*a))` |
| Ascent derivative | `w_i*(y_i-a_i-mu)` | `beta*(w_i*y_i-A*p_i)` |
| CUDA curvature approximation | `w_i` | `beta*A*(beta*p_i*(1-p_i)+lambda)` on positive-weight rows with `A>0`, otherwise zero |
| Positive objective sum | `sum(w*(y-a-mu)^2)` | `-sum(w*y*log(p))`, including only positive-weight, positive-target rows |
| Metric denominator | `sum(w)`; take the square root after division | `sum(w*y)` |

QueryRMSE intentionally preserves CUDA's diagonal approximation `h=w`, rather
than replacing it with the diagonal of the centered Hessian. Its derivative
corresponds to half the positive squared objective sum, as in CUDA. SoftMax
`lambda` changes curvature only; it is distinct from the leaf solver's
`l2_leaf_reg` and does not enter its loss or gradient. A singleton positive
target query therefore has zero gradient but nonzero SoftMax curvature when
`beta*lambda` is nonzero.

[`loss_description.cpp`](../private/libs/options/loss_description.cpp) sets
SoftMax defaults `beta=1` and `lambda=0.01`. The inspected CUDA/options paths do
not impose a sign range on either parameter. The kernels implement finite
signed values algebraically. Gradient ignores unused curvature. Newton leaf
estimation retains signed curvature and requires a finite positive
regularized leaf diagonal; Newton split scores require finite nonnegative
row curvature. SoftMax requires nonnegative targets and
a positive **global** sum of weighted targets. Individual zero-target or
zero-weight queries are valid and produce zero derivatives and loss. QueryRMSE
targets can have either sign. The target provider also enforces nonnegative
weights, finite input requirements, and group-weight consistency.

## Runtime contract

[`metal_querywise_kernels.h`](native/metal_querywise_kernels.h) provides the
standalone MSL string `CBMMetalQuerywiseSource`. Concatenating it does not change
the scalar `KernelParams` layout. The 32-byte `QuerywiseParams` structure is:

```cpp
uint32_t rows, groups, objective, apply_leaf_values;
float beta, lambda;
uint32_t leaves, reserved;
```

Objective IDs are reserved as `12=QueryRMSE`, `13=QuerySoftMax`. Offsets are
`uint32_t[groups+1]`, begin at zero, increase strictly, and end at `rows`.
Each query is contiguous in original training order. Offsets refer to the
original rows even when histogram/leaf partitions use another ordering.
The caller must validate offsets and buffer lengths before dispatch.

`PrepareQuerywisePoint` dispatches `rows` threads:

| Buffer | Type and meaning |
| --- | --- |
| 0 | `float[rows]` current original-order cursor |
| 1 | `float[leaves]` unshrunk raw leaf values |
| 2 | `uint[rows]` original-order leaf IDs |
| 3 | `float[rows]` output point |
| 4 | `QuerywiseParams` |

It copies the cursor when `apply_leaf_values=0`, otherwise computes
`point[row] = cursor[row] + raw_leaf_values[leaf_ids[row]]`. Bind valid buffers
even when the optional shift is disabled.

Both `QueryRmseDerivatives` and `QuerySoftMaxDerivatives` dispatch exactly
`groups` threadgroups of 256 threads, with these bindings:

| Buffer | Type and meaning |
| --- | --- |
| 0 | `float[rows]` targets |
| 1 | `float[rows]` effective original weights |
| 2 | `float[rows]` complete-query point |
| 3 | `uint[groups+1]` query offsets |
| 4 | `float[rows]` output weighted ascent derivatives |
| 5 | `float[rows]` output CUDA curvature approximation |
| 6 | `float2[groups]` output positive objective sum and denominator |
| 7 | `QuerywiseParams` |

Each thread strides across long queries, so the mathematical kernel has no
256-row query limit. Compensated high/low scratch storage is 8 KiB per query
threadgroup. Outputs
remain in original row order. No derivative must be multiplied by original
weights a second time. Aggregate the bounded per-query statistics separately
for metrics and the leaf oracle; the CUDA oracle uses negative objective sums
and adds any configured leaf regularization to its trial value.

`QuerywiseParams.reserved` bit zero optionally marks a row gradient as NaN
when its curvature is negative or nonfinite, allowing the runtime to reject
invalid Newton structure targets before histogram construction. Ordinary
Gradient mode leaves this bit clear and ignores unused curvature. The
curvature output itself retains its literal value.

The runtime must also run `ValidateQuerywiseStructureCurvature` for Newton
split scores. It binds curvature at buffer 0, a cleared four-byte atomic
status at buffer 1, and `QuerywiseParams` at buffer 2, dispatching `rows`
threads. It flags negative or nonfinite values explicitly; a NaN gradient
alone is insufficient when a score implementation skips invalid children.

Three additional helpers connect complete-query derivatives to the existing
leaf solver without recomputing query normalization inside a leaf:

| Kernel | Bindings and output | Dispatch |
| --- | --- | --- |
| `ResetQuerywiseLeafIds` | 0 leaf IDs; 1 `QuerywiseParams`; writes zero | `rows` threads |
| `ReduceQuerywiseLeafPartials` | 0 gradients; 1 curvature; 2 original weights; 3 partition row indices; 4 partition offsets; 5 output partials; 6 projection parameters | `(tiles,leaves)` full 256-thread groups |
| `ReduceQuerywiseObjective` | 0 per-query `float2(loss,denominator)`; 1 output `float2` partials; 2 `QuerywiseParams` | At most 4096 full 256-thread groups |

The 16-byte `QuerywiseProjectionParams` contains
`uint rows,leaves,tiles,leaf_method`. `leaf_method=1` means Gradient and writes
zero curvature even if that unused input is negative or nonfinite; Newton
preserves curvature. Each leaf/tile emits consecutive `float4` high and low
parts for `(gradient,curvature,original_weight,0)`, compatible with the existing
`EstimateNewtonLeafValues` layout. There are `2*leaves*tiles` such vectors.
The objective reducer emits **unnormalized** loss/denominator sums. Aggregate
those bounded partials in host double; unlike the PairLogit edge helper, no
edge-count rescaling is needed.

## Required training hooks

The CUDA query target reports `EOracleType::Pointwise` and scalar dimension.
Matching that path does not require a dense query-by-query Hessian solver. It
does require computing each complete query before projection to leaves:

1. Compute grouped derivatives using original effective weights, then apply
   bootstrap to the already-computed derivatives and structure weights.
   Sampled weights must never change query means or softmax normalization.
2. For each leaf estimation step and every backtracking trial, create the
   full original-order point and recompute grouped derivatives/objective.
3. Project those gradients and curvatures by current leaf ID. Newton uses
   projected curvature plus leaf regularization; Gradient uses the sum of
   original effective weights plus leaf regularization.
4. Preserve the existing CUDA walker rules, normalization options, empty-leaf
   behavior, and shrinkage placement. QueryRMSE and QuerySoftMax do not enable
   the separate `MakeZeroAverage` final-leaf option; see
   [`train_template.h`](../cuda/train_lib/train_template.h).
5. Carry query metadata through train/evaluation preparation, validate it, and
   bind it into snapshot fingerprints. Evaluate QueryRMSE and QuerySoftMax
   metrics with their distinct denominators. Prediction remains scalar.

[`catboost_options.cpp`](../private/libs/options/catboost_options.cpp) defaults
QueryRMSE to Newton with one leaf iteration, and QuerySoftMax to Gradient with
100 leaf iterations; its Newton default is ten. Exact leaf estimation is not
defined by this translation.

## Numerical adaptations and validation

Metal uses compensated high/low sums through the fixed 256-thread reduction;
CUDA uses its own warp/block reductions and wider leaf projection accumulators.
Bitwise CUDA agreement is not claimed. Positive-beta softmax uses CUDA's
`beta*(raw-maxraw)` stabilization. Negative beta uses the minimum raw value to
keep exponentials bounded. The loss is evaluated in the log domain to remain
finite when a float32 probability underflows; CUDA's `log(p)` can return
negative infinity in that case. Float32 overflow in an input product or total
remains subject to the runtime's finite-value checks.

[`test_querywise_kernels.py`](tests/test_querywise_kernels.py) compiles
[`querywise_probe.mm`](tests/querywise_probe.mm) and executes these kernels on
the actual Apple GPU. Independent double-precision equations cover mixed query
lengths through 65,539 rows, nonunit/zero weights, singleton and zero-signal
queries, signed beta/lambda values, large common offsets, softmax underflow,
cross-leaf trial points, derivative finite differences, and invalid offsets.
The projection tests additionally cover production leaf-solver compatibility,
cross-tile cancellation, zero-weight overflowing residuals, and the distinct
Gradient/negative-Newton-curvature policies.
The probe is diagnostic and does not expose a public querywise training API.

```sh
/tmp/catbooster-metal-venv/bin/python -m pytest -q \
  catboost/metal/tests/test_querywise_kernels.py
```

QueryRMSE/QuerySoftMax now also support Depthwise, Lossguide and Region in
native and standalone APIs. [GREEDY_QUERY_PORT.md](GREEDY_QUERY_PORT.md)
records whole-query projection, CTR dataset cursors and exact recovery.
