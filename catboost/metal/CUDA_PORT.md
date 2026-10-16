# CUDA-to-Metal translation map

This source map describes the evolving single-device Plain/DocParallel/
SymmetricTree port. [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) records
which entry points and options have passed validation. Both standalone and
native training call persistent C session APIs in `native/metal_trainer.h`
and `native/metal_multiclass.h`.
Shared CatBoost code supplies data preparation, metrics, and model conversion.

## Source correspondence

| Phase | CUDA source | Metal destination |
| --- | --- | --- |
| Boosting loop and learning-rate scaling | [`doc_parallel_boosting.h`](../cuda/methods/doc_parallel_boosting.h) | `native/metal_trainer.mm`; native `train_lib/train.cpp` |
| Symmetric search and repeated-winner stopping | [`oblivious_tree_doc_parallel_structure_searcher.cpp`](../cuda/methods/oblivious_tree_doc_parallel_structure_searcher.cpp) | Native session tree step |
| Twelve scalar objective gradients/Hessians | [`pointwise_targets.cu`](../cuda/targets/kernel/pointwise_targets.cu) | `native/metal_objective_kernels.h` |
| Stable row partitions, offsets, gather/scatter | [`pointwise_optimization_subsets.cpp`](../cuda/methods/pointwise_optimization_subsets.cpp) | `native/metal_histogram_kernels.h`, `native/metal_incremental_partition_kernels.h` |
| Histograms and bin scans | [`pointwise_hist2_one_byte_templ.cuh`](../cuda/methods/kernel/pointwise_hist2_one_byte_templ.cuh), [`split_properties_helpers.cuh`](../cuda/methods/kernel/split_properties_helpers.cuh) | `native/metal_histogram_kernels.h` |
| Smaller-child computation and sibling subtraction | [`pointwise_hist2.cu`](../cuda/methods/kernel/pointwise_hist2.cu) | `native/metal_histogram_reuse_kernels.h` |
| L2/Cosine/NewtonL2/NewtonCosine/SolarL2/LOOL2 scores | [`score_calcers.cuh`](../cuda/methods/kernel/score_calcers.cuh), [`pointwise_scores.cu`](../cuda/methods/kernel/pointwise_scores.cu) | `FindSplitWinners`, `ReduceSplitWinners` in `native/metal_kernels.h` |
| Bayesian/Bernoulli/Poisson/MVS object sampling | [`bootstrap.cu`](../cuda/cuda_util/kernel/bootstrap.cu), [`mvs.cu`](../cuda/cuda_util/kernel/mvs.cu), [`random_gen.cuh`](../cuda/cuda_util/kernel/random_gen.cuh) | `native/metal_bootstrap_kernels.h` |
| Randomized Cosine feature scores | [`random_score_helper.h`](../cuda/methods/random_score_helper.h), `TCosineScoreCalcer` | `native/metal_score_noise_kernels.h`; runtime scheduling |
| Objective projection and iterative leaf estimation | [`update_part_props.cu`](../cuda/cuda_util/kernel/update_part_props.cu), [`pointwise_oracle.cpp`](../cuda/methods/leaves_estimation/pointwise_oracle.cpp), [`descent_helpers.cpp`](../cuda/methods/leaves_estimation/descent_helpers.cpp) | `ReduceLeafObjectivePartials`, `EstimateNewtonLeafValues` |
| AnyImprovement/Armijo leaf backtracking | `TNewtonLikeWalker` in [`descent_helpers.cpp`](../cuda/methods/leaves_estimation/descent_helpers.cpp) | `native/metal_backtracking_kernels.h`; native session walker |
| Leaf bits and prediction updates | [`split.cu`](../cuda/gpu_data/kernel/split.cu), [`add_model_value.cu`](../cuda/models/kernel/add_model_value.cu) | `UpdateLeafBins`, `AddObjectiveBinModelValue` |
| Categorical statistics | [`ctr_calcers.cu`](../cuda/ctrs/kernel/ctr_calcers.cu) and shared categorical preparation | `native/metal_ctr_kernels.h`, `native/metal_ctrs.mm`, `train_lib/categorical.cpp` |
| Tree evaluation and output conversion | [`evaluator.cu`](../libs/model/cuda/evaluator.cu), [`evaluator.cpp`](../libs/model/cuda/evaluator.cpp) | `native/metal_inference_kernels.h`, `native/metal_inference.mm`, `../libs/model/metal/evaluator.cpp` |
| Metrics, stopping, snapshots | [`boosting_progress_tracker.cpp`](../cuda/methods/boosting_progress_tracker.cpp), shared metrics/error tracker | `train_lib/progress.cpp`, `train_lib/snapshot.h`; standalone `_training.py` |
| Multiclass objectives, coupled leaf estimation, sampling and gain scores | [`multiclass_targets.cpp`](../cuda/targets/multiclass_targets.cpp), [`greedy_search_helper.cpp`](../cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp) | `native/metal_multiclass.mm`; [multiclass sampling contract](MULTICLASS_SAMPLING_PORT.md) |
| QueryRMSE/QuerySoftMax grouped foundation | [`query_rmse.cu`](../cuda/targets/kernel/query_rmse.cu), [`query_softmax.cu`](../cuda/targets/kernel/query_softmax.cu) | `native/metal_querywise_kernels.h`; [query integration contract](QUERYWISE_PORT.md) |
| Given-edge PairLogit pointwise target/runtime | [`pair_logit.cu`](../cuda/targets/kernel/pair_logit.cu) | `native/metal_pairwise_kernels.h`, `native/metal_pairwise_runtime.h`; [pairwise integration contract](PAIRWISE_PORT.md); persistent scalar-session training connected, public/native adapter pending |

## Weighted objectives and leaf estimation

Each tree evaluates the objective at the ensemble cursor, searches its
structure, estimates leaves on original observations, scales completed leaf
values by the learning rate, and updates the cursor. Persistent sessions allow
one tree at a time, validation/stop decisions, and restoring GPU state.

RMSE uses `g = w * (target - prediction)` and `h = w`, with no factor of two.
Binary objectives use `g = w * (target - sigmoid(raw))` and
`h = w * sigmoid(raw) * (1 - sigmoid(raw))`. Stable probability/loss expressions
handle large finite logits. Poisson, Huber, and Expectile retain their CUDA
objective-specific target and parameter restrictions.

L2/Cosine structure scoring uses observation weights; NewtonL2/NewtonCosine
use the objective curvature. SolarL2 and LOOL2 follow their separate CUDA
score equations and restrictions.
Bootstrap multiplies the structure gradients and weights. Final leaf
estimation recomputes derivatives using the original observation weights.
This distinction matters for classification and subsampling.

At each leaf-estimation iteration, let `G`, `H`, and `W` be sums at the current
unshrunk leaf point. Newton uses `G / (H + lambda + 1e-20)`; Gradient substitutes
`W` for `H`. Weight-empty leaves are zeroed; nonpositive diagonals do not take a
step. The supported configuration adds `l2_leaf_reg` directly, disables loss
normalization and the extra ridge-objective gradient, and scales leaves only
after estimation. Standalone No/AnyImprovement/Armijo backtracking now has
separate CUDA-walker reference tests, including accepted-step budgets,
nonfinite candidate objectives, and exact cursor resumption. Its GPU kernels
are in `native/metal_backtracking_kernels.h`; the host controls acceptance.
The current preserved native wheel includes all three backtracking modes and
the supported Exact scalar objectives.

RMSE may start at the weighted target mean. Binary `boost_from_average` uses
the weighted positive fraction's logit. The standalone adapter rejects that
option for Poisson/Huber/Expectile under the shared option rules.

## Partitions, histograms, and split semantics

Rows keep stable ordering inside a leaf. The root uses an identity partition;
later partitions split the newly appended bit instead of sorting complete leaf
keys again. Banked threadgroup histograms accumulate weighted gradients and
structure weights, then flush with device float atomics. Numeric features
receive parallel inclusive scans; one-hot features retain equality-bin sums.

After the first depth, parent histograms are retained, the smaller child is
computed, and subtraction produces its sibling. Equal sizes choose the right
child, matching CUDA. Subtraction retains float roundoff rather than clamping
the stored sibling. Packed binary/half-byte specializations and compact
per-feature storage remain separate optimizations.

A numeric candidate derives its right side from the parent:

```text
right_weight = max(parent_weight - left_weight, 0)
right_gradient = parent_gradient - left_gradient
```

One-hot candidates swap sides so equality routes right. L2 contributes
`-G*G / (W+lambda)` when `W > 1e-20`. Cosine uses:

```text
mu = W > 0 ? G / (W + lambda) : 0
score = -sum(G * mu) / sqrt(1e-10 + sum(W * mu * mu))
```

Both minimize scores. GPU reductions preserve the lowest candidate index on a
tie. The host reads a small winner record instead of scanning every candidate.
CUDA also has a host winner boundary; eliminating every depth synchronization
would be an additional orchestration change.

Numeric bins route right when `bin > zero_based_border_index`; equality with
the float border goes left. One-hot bins route right on equality. The first
split is leaf bit zero. The search evaluates all candidates and stops if the
winner already occurs in the tree, so actual depth can be smaller than the
configured maximum. Export uses actual depths and corresponding leaf counts.

## Bootstrap and score randomness

[BOOTSTRAP_PORT.md](BOOTSTRAP_PORT.md) records distribution formulas, MVS
regularization state, and buffer contracts. The multiply-with-carry step is
translated from CUDA; seed expansion uses absolute iteration, stream, and
object/feature so resumed Metal training reproduces its sequence. This does
not reproduce CUDA's launch-dependent seed initialization.

Scalar Cosine randomness uses original weighted gradients before bootstrap. CUDA's
zero-aware divide returns zero for `-1e-15 < g < 1e-15`, otherwise
`g / (w + 1e-15)`. The weighted squared norm divides by **row count**:

```text
stddev = sqrt(sum(w * zero_aware_divide(g, w)^2) / rows)
scale = random_strength * stddev / (1 + exp(iteration * learning_rate - log(rows)))
```

The scale stays fixed within a tree. Each depth draws one normal per feature,
shared across its thresholds. L2 ignores the perturbation. Standalone tests
cover the weighted formula, actual winning splits, decay, and resumed
training; native acceptance also covers these options. Multiclass instead
uses post-bootstrap full-class energy divided by sampled weight and selects
rounded gain before applying a noisy raw-score growth check. Its different
contract is documented in [MULTICLASS_SAMPLING_PORT.md](MULTICLASS_SAMPLING_PORT.md).

## Categorical statistics and standard models

Shared/native preparation retains category hashes, metadata, target borders,
and perfect-hash conversion. Metal segmented scans calculate category history
statistics. Standalone public training exposes Borders and FeatureFreq CTRs;
lower-level kernels also implement Buckets and FloatTargetMeanValue. Native
acceptance covers all four simple CTRs and their standard model tables.
Native scalar and multiclass training now exercise multiple independent CTR
permutation cursors, including P4, with shared structure search and per-cursor
leaf estimation. Native scalar symmetric Plain/Ordered FeatureParallel training
also generates compound CTRs after selected splits, retaining per-history grids,
dynamic feature activity and exact snapshot state. Its Sample/Group histories and
four CTR types use the same GPU projection/CTR primitives and standard tables;
see [COMPOUND_CTR_PORT.md](COMPOUND_CTR_PORT.md). FeatureParallel size penalties
distinguish dynamic, active, registered and used CTR configurations.

Plain boosting can use permutation-dependent CTRs. Ordered boosting separately
requires prefix folds and fold prediction cursors; numeric Ordered does not
depend on categorical support. [ORDERED_PORT.md](ORDERED_PORT.md) records its
separate cursor/fold contract. Native Ordered numeric, one-hot, simple and compound
CTR integration is recorded in [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

The standalone adapter constructs standard JSON feature/tree/CTR data and
loads it with the installed CatBoost reader. Native training uses shared
`TObliviousTreeBuilder` and full-model conversion. Both produce ordinary CBM
and JSON files. Native Darwin ARM64 registers under the existing GPU task type;
no third task enum or new model format is needed.

Metal inference evaluates symmetric trees in tiles with compensated
accumulation and scale/bias processing. Native inference reuses shared Pool
preparation, one-hot handling, and CTR lookup, then evaluates trees on Metal.
CPU inference remains available for exported models.

## Precision and validation limits

CUDA histograms/cursors already use float, but some reductions, scores, and
leaf-update intermediates use double. Metal's staged compensated float
expansions improve partition/leaf and prediction sums. Score intermediates and
final leaf updates still differ from CUDA's mixed-precision path. A float64
return array alone does not establish equivalent arithmetic. Large cancelling
sums and near-tied splits remain important parity tests.

Runtime indexing/allocation checks precede native pointer calls. Current
training caps are software limits documented in the status page; inference
supports deeper existing models independently of training depth support.

No NVIDIA execution or other M-series hardware comparison has been performed.
Tests establish translated formulas, actual M3 execution, and interoperability
with CatBoost's existing model reader. Complete CUDA feature, quality, and
speed parity is not yet established. The historical [PORT_REVIEW.md](PORT_REVIEW.md)
is preserved; current remaining work is in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
