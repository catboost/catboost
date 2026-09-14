# Metal API and training options

**Card 4 is complete and its coherent release acceptance passed.**
Installed checkpoint: `20260914T025201Z`. Final accepted counts:
14,376 tests plus 16 subtests in the full matrix; 6,118 tests alternate; 2,431 tests installed; 348 CLI; 203 preinstall smoke; 203 installed smoke; 350 exact preceding snapshot recoveries. The preceding published release is card 3 commit
`27c9fad9cf`, checkpoint `20260914T000929Z`; its acceptance does not certify the
new source. This document records implemented contracts and deliberate
compatibility differences. See [the card](../../Kanban/04-api-and-options.md)
and the [preceding training-mode report](TRAINING_MODES_PORT.md).

The current native contracts are in
[`train.cpp`](train_lib/train.cpp) and [`categorical.cpp`](train_lib/categorical.cpp).
Shared GPU validation must continue to reject CPU-only or unregistered
combinations. A removed guard is insufficient when feature preparation, scores,
model publication or saved state still lack the corresponding behavior.

## Audited scope

| Workstream | CUDA contract | Card 4 implementation/status |
|---|---|---|
| Native cross-validation | Metrics-only and optional returned fold models | Finalizes histories/cursors for both modes; metrics-only calls return before model publication |
| Text and embedding training | Shared offline/online estimators feed registered GPU trainers | Estimator banks, histories, estimated splits, final processing collections and shared readers are connected |
| Simple leaves | Source-specific scalar, query, vector and greedy semantics | Registered native paths are connected, including six symmetric vector families; FeatureParallel retains its one-step estimator |
| Fixed binary splits | Plain DocParallel non-symmetric float features with one border | Resolves manager IDs and preserves forced-prefix/policy termination rules |
| RSM | PairLogitPairwise, QueryCrossEntropy and YetiRankPairwise feature packs | Per-tree masks preserve ordering, retries and shared host draw accounting |
| Automatic CTR priors | Simple Borders, one target border, shared host Beta estimator | Resolves per-feature priors before grids and snapshot identity |
| Full frequency counters | Simple training counters include first eval; dynamic/final tables remain learn-only | Separate training/evaluation and publication tables, with evaluation identity in snapshots |
| Feature weights | GPU gain multipliers; other feature-use penalties are CPU-only | Maps original/estimated/bundle/simple-CTR IDs; documents the dynamic local-index correction |
| One-hot boundary | 256 known categorical values | Native bins use all 256 values; shared hashed readers distinguish unseen inference |
| Normalization, ridge, meta-L2 | Distinct score/leaf consumers and source no-ops | Native setters, score equations, shared host seeds and snapshot accounting are connected |
| Langevin | Explicit GPU flag and source-specific weak/leaf events | Scalar, Ordered, greedy and vector consumers share the actual host event stream; full-matrix rejection remains explicit |

Native scalar/query Plain FeatureParallel and native compound CTRs were connected
in cards 2–3. Their standalone frontend now routes through the native adapter,
including Pool preparation, snapshots and online learn cursors. Full-matrix,
vector and greedy trainers retain their registered partition boundaries.
These new standalone routes require this fork's native Metal package; an
unmodified upstream wheel cannot supply them. Combination/custom objectives and
text/embedding calcer configuration remain native CatBoost interfaces.
Explicit symmetric Simple also activates native routing for standalone scalar,
vector and registered diagonal-query objectives; existing implicit defaults
remain unchanged. Existing full-matrix Simple, whether implicit or explicit,
retains its direct runtime unless another newly exposed option, such as RSM,
selects native routing. Existing full-matrix configurations and snapshots keep
their original path.

## Cross-validation

The shared [`cross_validation.cpp`](../libs/train_lib/cross_validation.cpp)
already creates the trainer by `ETaskType::GPU`; Metal registers both its trainer
and trainer environment. Ordinary CV sets `CalcMetricsOnly=true` when
`return_models=false`, and passes no destination model. Metal now finishes
progress and populates histories/evaluation cursors, then returns before model
conversion/publication for these metrics-only calls. Requested fold models use
ordinary finalization. Shared CV already
forces evaluation metrics each iteration and runs its own fold stopping
callbacks; the Metal progress path receives those controls.

Retain shared restrictions: no CV snapshots, no Pool marked
`EObjectsOrder::Ordered`, nonempty data with enough objects for the folds, and no
stratification for querywise metrics or absent/multidimensional targets. The Pool
order restriction is distinct from `boosting_type="Ordered"`. Native acceptance
covers returned-model modes, independent fold aggregation, early stopping,
custom folds and rejection paths without CPU training; final totals appear below.

The standalone adapter exposes the retained model's online learn cursor through
transient history. The shared entry point composes timestamp, shuffle and
generated-pair permutations and scatters only this returned cursor back to the
caller Pool's order. Native/snapshot cursors retain prepared order. An optional
v6 tail stores the best cursor and its iteration. Legacy snapshots still restore
training/models; when no compatible best cursor was saved, its accessor reports
it unavailable. Final CTR tables cannot substitute for online learn values.
These transient fields are omitted from model metadata and ordinary history
serialization.

The shared native copy path preserves stored evaluation predictions through
`CatBoost.copy()`, the standalone `to_catboost()` result and pickle round trips.
Returned evaluation values are independent copies. This fixes lost evaluation
state discovered by Full-counter and YetiRank frontend checks; stored online
values are retained instead of being recomputed from final model tables.

## Text and embedding features

CUDA's [`TEstimatorsExecutor`](../cuda/gpu_data/estimated_features_calcer.cpp)
calls shared host `ComputeFeatures`/`ComputeOnlineFeatures`; those calculations
can be reused without a CPU tree-training fallback. The prepared
`TTrainingDataProviders::FeatureEstimators` already reaches the native adapter.
The adapter now appends their feature banks and original feature metadata to
both symmetric and non-symmetric model builders.

The source estimator scope is:

- Text Bag of Words is offline and available across the applicable targets.
  NaiveBayes and BM25 are online classification estimators.
- Embedding LDA and KNN are online estimators for classification and regression.
- No additional estimator-specific tree-policy registration was found; the
  selected objective must still have the requested trainer registration.

Sources: [`data.cpp`](../private/libs/algo/data.cpp),
[`text_feature_estimators.cpp`](../private/libs/feature_estimator/text_feature_estimators.cpp)
and [`embedding_feature_estimators.cpp`](../private/libs/feature_estimator/embedding_feature_estimators.cpp).
Online learn statistics update strictly after the current row, test statistics
use the complete learn data, and each learning permutation needs its own bank.
CUDA retains borders from the first learn/permutation calculation. BoW uses a
0.5 split; estimated floats use global float binarization with forbidden NaNs
and at most 255 borders, including the source constant-feature fallback.

The integrated model path includes original text/embedding metadata,
`TEstimatedFeatureSplit` identity (source feature, estimator GUID, local feature
ID and source type), model processing collections and feature-estimator
finalization. See CUDA's
[`model_converter.cpp`](../cuda/cpu_compatibility_helpers/model_converter.cpp)
and shared [`full_model_saver.cpp`](../private/libs/algo/full_model_saver.cpp).
The existing CTR provider remains attached during feature-calcer finalization.
Native progress evaluates intermediate trees with shared estimators and
apply-compatible data, so validation and early stopping use finalized evaluation
features during training. See [estimated_features.md](docs/estimated_features.md).

CUDA prohibits estimated features inside CTR tensors in
[`binarizations_manager.cpp`](../cuda/data/binarizations_manager.cpp); that is not
a required extension to the compound scheduler. CUDA's
[`GPU model evaluator`](../libs/model/cuda/evaluator.cpp) also rejects text and
embedding models. Raw text/embedding GPU inference therefore remains an explicit
unsupported boundary; shared CPU model application can evaluate the processing
collections without performing CPU training.

GPU leaf-index output is also unsupported by both the [CUDA evaluator](../libs/model/cuda/evaluator.cpp)
and the [Metal evaluator](../libs/model/metal/evaluator.cpp). Routing checks select
the shared CPU model reader explicitly after GPU prediction.

## Fixed splits and greedy Simple

The functional fixed-split implementation belongs to
[`train_template_pointwise_greedy_subsets_searcher.h`](../cuda/train_lib/train_template_pointwise_greedy_subsets_searcher.h).
It requires Plain non-symmetric training and validates each requested feature as
a float feature with exactly one registered border. Shared options force those
policies to DocParallel. The list does not itself switch the default grow policy.

The values are feature-manager IDs, not Metal candidate indices. They normally
match flat feature IDs, including ignored numeric/categorical slots, but float
slicing and omitted text/embedding slots shift later IDs. Metal's compressed
candidate grid now resolves those identities explicitly. The shared CLI name converter
supplies flat IDs, leaving an upstream mismatch for unusual layouts; that source
quirk must be distinguished from a deliberate Metal compatibility rule.

[`greedy_search_helper.cpp`](../cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp)
indexes the list by global maximum depth and installs bin zero with negative
infinite score/gain in eligible uncached leaves. Repeated IDs are accepted.
Cached winners, minimum leaf size, depth and policy capacity still govern
termination; the complete requested prefix is not guaranteed. The final forced
level resumes policy-specific leaf selection. Region has stronger structural
assumptions that long forced prefixes may violate; this is a source inference
covered by bounded prefix validation, not a verified NVIDIA result.

Symmetric fixed-split behavior is trainer-dependent: the scalar symmetric
pointwise template has no consumer and silently ignores the option, while
symmetric multiclass through the greedy template rejects it. Metal preserves
that distinction; accepting the ignored scalar option is not a symmetric
forced-split capability.

For Simple, [`TGreedySubsetsSearcher::NeedEstimation`](../cuda/methods/greedy_subsets_searcher.h)
returns false. Metal exports the searched weak tree and sampled weights,
including source empty-partition behavior, instead of iterative leaves. It
retains one estimation iteration and objective-specific restrictions,
including top-level YetiRank's required Newton method. Targeted acceptance covers
categorical permutations, vector leaf geometry, readers and exact continuation.
The coherent release gate reran these supported paths.
CUDA uses this template for symmetric MultiClass, MultiClassOneVsAll, MultiRMSE,
RMSEWithUncertainty, MultiLogloss and MultiCrossEntropy as well. Explicit Simple
reuses searched values across permutations; all six native registrations are
connected. For newly enabled vector Simple/Langevin search, repeated negative
winning splits remain eligible as in CUDA's generic symmetric template. The
previously released default vector paths retain their established stopping rule.

Symmetric scalar/query DocParallel has a different Simple implementation. It
uses the normal score dispatch: second-order scores receive target curvature,
and other scores receive target weights. QueryRMSE and QuerySoftMax use original
object weights for that latter branch; PairLogit uses incident pair mass. Each
sampled leaf exports `sum / (weight + l2_leaf_reg)` when its row count is positive,
even if its weak weight is zero or negative, and exports that signed weight.
The greedy query target's inverted dispatch is separate. Negative QuerySoftMax
`lambda` is permitted by shared options and can produce negative curvature;
admission of signed model/snapshot weights is limited to the relevant Simple
configuration. These checks do not change ordinary iterative query training.

Plain/Ordered FeatureParallel always uses the batched leaf estimator. Its Simple
method performs one Gradient step using original task weights, with complete
permutation/prefix cursor handling. The existing Metal correction to the CUDA
PairLogit/YetiRank centering defect remains in place. See the source explanation
in [TRAINING_MODES_PORT.md](TRAINING_MODES_PORT.md).

## RSM sampling

[`IsPairwiseScoring`](../private/libs/options/enum_helpers.cpp) names exactly
PairLogitPairwise, QueryCrossEntropy and YetiRankPairwise. Shared GPU validation
rejects nondefault RSM elsewhere. These registered trainers are symmetric Plain
DocParallel and limited to depth eight.

[`TComputePairwiseScoresHelper`](../cuda/methods/pairwise_oblivious_trees/pairwise_score_calcer_for_policy.cpp)
samples once at construction, after the stochastic weak target. It leaves binary
features untouched. For nonbinary grids it draws one uniform per packed group:
eight features for half-byte storage or four for one-byte storage, retaining all
candidate borders of the selected features. It requires `rsm > 0.01` when that
sampling path is invoked. An empty selection retries on the same RNG with
`min(2*rsm, 1)`; a fully selected grid uses the original representation.

Static features are processed before permutation-dependent features, and policy
order is Binary, HalfByte, OneByte. CUDA first shuffles the one-byte grid using
local `TRandom(0)`, then sorts by its bin-count/CTR grouping level. One-hot fold
counts use OnAll cardinality, so categories cannot be classified using only the
observed learn candidate count. Relevant sources are
[`grid_policy.h`](../cuda/gpu_data/grid_policy.h),
[`feature_layout.cpp`](../cuda/gpu_data/feature_layout.cpp),
[`feature_layout_doc_parallel.h`](../cuda/gpu_data/feature_layout_doc_parallel.h)
and [`pairwise_structure_searcher.cpp`](../cuda/methods/pairwise_oblivious_trees/pairwise_structure_searcher.cpp).

Metal now builds source-compatible groups and per-tree masks, retaining existing
candidates for model routing. Per-feature or
per-depth Bernoulli masks would change the contract. RSM draws share the target
host stream, so weak-target/bootstrap/leaf consumption and saved-state replay
are accounted for alongside the existing documented Metal device streams.
New `rsm < 1` simple-CTR grids retain CUDA's 0.5 fallback for constant columns.
This prevents selected packs from becoming empty without introducing another
RSM draw. Existing `rsm=1` CTR grids retain their prior behavior.

## Full frequency counters

`counter_calc_method="Full"` changes precomputed FeatureFreq values to
`(learn_count + eval_count + prior_numerator) / (learn_rows + eval_rows + prior_denominator)`.
Counts include every row independently of labels, object weights and query
weights, including categories seen only in evaluation. CUDA supports one
evaluation dataset; Metal uses the first evaluation dataset for this option.
With no evaluation dataset, Full and SkipTest have the same values. Borders
are computed from the learn slice of these values. Borders, Buckets and
FloatTargetMeanValue histories do not change.

This option has different consumers at different phases. The precomputed
[`batch_binarized_ctr_calcer.cpp`](../cuda/gpu_data/batch_binarized_ctr_calcer.cpp)
sets `UseFullDataForCatFeatureStats` on its CTR helper. Dynamic tree tensors
instead use the learn-only tracker from
[`oblivious_tree_structure_searcher.cpp`](../cuda/methods/oblivious_tree_structure_searcher.cpp),
and their reprojection helper keeps its default learn-only mode in
[`oblivious_tree_bin_builder.cpp`](../cuda/gpu_data/oblivious_tree_bin_builder.cpp).
Those dynamic frequencies ignore Full.

Final model conversion preserves the FeatureFreq type, and shared
[`online_ctr.cpp`](../private/libs/algo/online_ctr.cpp) includes evaluation rows
only for the distinct Counter type. Exported FeatureFreq tables therefore
always contain learn-only counts and use the learn row count as denominator.
Metal preserves this source behavior: temporary Full tables serve training and
evaluation cursors; final publication restores learn-only FeatureFreq tables.
Consequently `get_test_eval()` can differ from `predict(eval_pool)` under Full.
Independent acceptance must check both formulas and exact snapshot restoration,
including rejection when the first evaluation dataset changes.

## Priors, weights and remaining defaults

[`EstimatePriors`](../cuda/train_lib/train.cpp) uses the host-only
[`TBetaPriorEstimator`](../cuda/ctrs/prior_estimator.cpp), raw quantized category
values, OnAll unique cardinality and the binarized target. It does not use object
weights or query/permutation history weights. It writes float priors
`{Alpha, Alpha + Beta}` into per-feature CTR descriptions before grid creation.

Only simple Borders priors are supported. Shared GPU validation rejects automatic
priors for combinations and other CTR types. The source returns without
estimating if actual target borders exceed one; with at most one actual border,
requested estimation requires configured target-border count one. When any
description requests automatic estimation, the source also replaces sibling
Borders descriptions for that feature even if their individual flag is No.
Global automatic descriptions populate only features without an override.
The implementation retains these source details and resolves the options before
building grids, model CTR identity and snapshot fingerprints. Independent tests
compare the learned prior with a Beta-binomial likelihood oracle and cover
weights, overrides, permutations and saved-state continuation.

There is also a source initialization defect: both CUDA training branches call
`EstimatePriors` before `SetTargetBorders` on a fresh feature manager whose target
border vector is empty. Literal reproduction would binarize every target to zero.
The Metal implementation establishes the shared target borders first, then
applies the host estimator to those intended inputs. This is an explicit correction
to source ordering, not evidence of observed NVIDIA behavior.

GPU `feature_weights` is distinct from `first_feature_use_penalties`,
`per_object_feature_penalties` and `penalties_coefficient`, which are declared
CPU-only in [`feature_penalties_options.h`](../private/libs/options/feature_penalties_options.h).
The GPU score kernels weight the improvement relative to the preceding score;
they retain the underlying winning score separately. Thus a weight cannot be
implemented by scaling raw histograms or changing the saved score. Metal now
passes mapped weights to the applicable scalar, Ordered, vector, greedy and
full-matrix scorers while preserving leaf estimation and model application.

CUDA's `ExpandFeatureWeights` indexes the raw option-map keys directly; it does
not remap flat user indices into feature-manager IDs. The manager registers float
and categorical slots, including ignored ones, skips text/embedding slots, slices
wide float features, and then appends estimated features, bundles and CTRs. Derived
CTRs do not inherit their source categorical weight. The dynamic CTR visitor
passes the same global weight vector to a scorer using local pack feature IDs,
so it aliases those local indices into the global vector. These unusual source
semantics have explicit compatibility tests; multiplying constituent weights or
propagating a category weight to all its CTRs would implement a different rule.
Prequantized exclusive bundles reserve manager IDs before CTRs, and their member
columns use the bundle's weight. The source loss-specific omission of bundles is
preserved for full-matrix pairwise and multiclass training.
The card 4 implementation deliberately corrects the dynamic local-index alias:
derived CTR user weights stay one, while model-size penalties still apply. See
the explicit contract in [feature_weights.md](docs/feature_weights.md).
See [`feature_penalties_options.cpp`](../private/libs/options/feature_penalties_options.cpp),
[`binarizations_manager.h`](../cuda/data/binarizations_manager.h) and
[`tree_ctr_datasets_visitor.cpp`](../cuda/methods/tree_ctr_datasets_visitor.cpp).

Fold normalization, ridge in the objective and meta-L2 have distinct consumers.
Their native wiring and host seed handling passed coherent release validation:

- Fold normalization changes symmetric scalar Cosine/NewtonCosine score regularization to
  `lambda * leafWeight`. The batched leaf estimator divides each task's value,
  gradient and Newton/Gradient denominator by that task's original total weight
  before adding L2. Both FeatureParallel and symmetric scalar DocParallel use
  that estimator; the latter creates all permutation tasks in one walker.
  Greedy, vector and full-matrix DocParallel use different leaf oracles that do
  not consume the leaf-normalization flag.
- Ridge subtracts `lambda * point` from gradients and `lambda/2 * sum(point²)`
  from the maximized value. FeatureParallel, symmetric/greedy scalar DocParallel and full-matrix
  leaf oracles apply it; the DocParallel vector oracle explicitly skips it.
  DocParallel/greedy Simple and Exact bypass iterative estimation; FeatureParallel
  Simple's sole step starts at zero, where the ridge gradient is zero. The
  ordinary denominator L2 term is present independently of this flag.
- Meta-L2 is consumed only by Plain symmetric scalar L2/NewtonL2 scoring. The device score seed
  selects the configured exponent with the configured frequency; leaf score
  `s=-sum²/(weight+lambda)` becomes
  `copysign((abs(s)/weight)^exponent, s) * weight`. Ordered, greedy and the other
  Plain score families, vector and full-matrix trainers have no corresponding
  consumer. Finite zero or negative exponents and frequencies outside `[0,1]`
  follow source validation: a frequency at most zero disables the transform,
  above one always selects the configured exponent, and `(0,1]` samples it.

The estimator distinction follows the actual factories in
[`doc_parallel_pointwise_oblivious_tree.h`](../cuda/methods/doc_parallel_pointwise_oblivious_tree.h)
and [`doc_parallel_boosting.h`](../cuda/methods/doc_parallel_boosting.h), rather
than the partition name alone. Meta-L2 makes no additional shared host RNG draw:
each existing dataset score seed expands through local `TRandom` into one seed
per present Binary/HalfByte/OneByte policy, and each policy chooses its exponent
from that seed. Dynamic CTR scoring also includes the device seed and base-tensor
hash. A single exponent for every feature bank would lose that distinction.
The same runtime CTR can belong to multiple active base-tensor packs. The Meta-L2
callback preserves both exponent choices and Metal compares their two complete
candidate scores; choosing each leaf's exponent independently would produce a
score that CUDA never evaluates. Without Langevin, FeatureParallel and stochastic
DocParallel accessors inspect their source streams without advancing them, and
completed-tree state accounts for the actual dataset draws. The deterministic
DocParallel Meta stream advances with scorer calls and reconstructs its state
from saved tree depths. With Langevin, the callback consumes those
same score seeds directly from the shared event controller; later search
accounting subtracts the consumed count. Target, cache and leaf events therefore
remain interleaved correctly. Prior default configurations keep their existing
RNG path.

Langevin's registered source consumers are connected. The GPU defaults are
`langevin=false` and `diffusion_temperature=0`; a positive temperature alone does
not enable Langevin. CUDA does not apply the CPU automatic activation,
temperature 10000 or model-shrinkage defaults. Registered ordinary scalar,
querywise and vector trainers accept the explicit flag. The full-matrix
PairLogitPairwise, QueryCrossEntropy and YetiRankPairwise leaf oracles reject it,
including an explicit flag with zero temperature.

Weak-target noise is applied by symmetric scalar DocParallel after bootstrap
and Bernoulli/Poisson filtering, and by Ordered FeatureParallel before statistics
and bootstrap. Plain FeatureParallel, greedy and vector trainers have no weak
noise consumer. Iterative leaf estimation has its own source event stream;
DocParallel Simple/Exact and greedy Simple have no leaf-noise events, whereas
FeatureParallel Simple retains its two initial events. Acceptance includes an
effective configuration and the greedy Simple no-op with positive temperature.
Host event seeds follow one shared source RNG, including actual rejected leaf
trials and stochastic Combination component calls; no unused leaf packet is
reserved. Snapshots store the controller's actual draw count and cache state,
with bounds derived from validated task, search and walker dimensions. Host
leaf noise uses the shared 128-element block helper. Device weak-noise draws
retain the documented Metal MWC adaptation, so this is not a NVIDIA bitwise RNG
claim. Zero temperature retains source host events without adding noise.

Posterior sampling, model shrinkage, first/per-object feature penalties and
full-history approximations remain CPU-only options.

Defaults also require explicit compatibility decisions: this port defaults to
Plain and complexity one, while generic CUDA options default applicable scalar
boosting to Ordered and CTR complexity four before data-dependent adjustments.
CUDA collapses Plain permutations only when no permutation-dependent CTR or
online estimated feature exists; online estimators must participate in that
decision. Native Metal now supports CUDA's 256-value one-hot boundary: all
`uint8` values represent known categories and model application matches their
hashes, keeping unseen values distinct. The older standalone DocParallel encoder
still reserves byte 255 for unseen values and retains its explicit 255-value
boundary. Existing accepted configurations and snapshots retain their defaults.

One further published Metal default is preserved explicitly. CUDA's
`UpdateGpuSpecificDefaults` changes an **implicit** Simple leaf method to Newton
when categorical or online estimated features retain multiple histories. This
affects the default YetiRankPairwise method; an explicit Simple setting stays
Simple in CUDA. Metal retains its published implicit Simple behavior to avoid
changing existing models and snapshot algorithms during this release. Callers
can request Newton explicitly. This is a documented default difference, like
Plain boosting and complexity one, not an unsupported explicit training option.

Sources: [`catboost_options.cpp`](../private/libs/options/catboost_options.cpp),
[`oblivious_tree_options.cpp`](../private/libs/options/oblivious_tree_options.cpp),
[`boosting_options.h`](../private/libs/options/boosting_options.h),
[`pointwise_scores.cu`](../cuda/methods/kernel/pointwise_scores.cu), and
[`leaves_estimation_config.h`](../cuda/methods/leaves_estimation/leaves_estimation_config.h).

## Accepted release

Checkpoint `20260914T025201Z` passed all required gates and is installed
in `catboost/metal/.venv`. Failures, errors and skips are zero.

| Acceptance gate | Passed |
|---|---:|
| Full native/standalone matrix | 14,376 tests plus 16 subtests |
| Alternate extension | 6,118 tests |
| Installed package | 2,431 tests |
| CLI configurations | 348 |
| Preinstall GPU smoke configurations | 203 |
| Installed GPU smoke configurations | 203 |
| Exact preceding snapshot recoveries | 350 |
| Metal C++ helper checks | 118 |
| Estimated/CTR metadata C++ checks | 3 |
| Combination metric C++ checks | 2 |
| Quantized categorical apply C++ checks | 3 |

These selections overlap and must not be summed. The preinstall and
installed smoke runs are separate gates over the same inventory; host
subsets can overlap the C++ total. Earlier diagnostic counts are not added
to this final acceptance. No CPU CatBoost fit or NVIDIA execution was used.

The snapshot gate replays 244 preserved earlier fixtures plus
106 original snapshots written by checkpoint `20260914T000929Z`.
Expected arrays, histories and callback sequences come from those original
builds; current-build expectations do not replace them. The preceding wheel
and immutable fixture hashes were checked again before finalization.

The gate records identify 743 unchanged source files and source base
`27c9fad9cfb9f415ba63483ab72368a7cbefbc30`. Package identity:

- Checkpoint directory: `catboost/metal/.build/releases/20260914T025201Z/`
- Wheel: `catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl`
- Wheel SHA256: `afc6061556df133351dfc14232faea1a257be328838f2e0522fff1d91af66117`
- Standard extension SHA256: `037c62099bb935b003601fd7c686421786ad15f5820e8f819a315aa1d7e5066e`
- Alternate extension SHA256: `1dae6900a58034df2e0562b3b24a0f2af1251f02b186645a52d1f4f36f6651b5`
- CLI SHA256: `0b54dfcdd6c0f0ed1440ba2f3ef9e658f623bdc26725267519ec4ca9eea68b66`

One numerical follow-up remains explicit: CUDA's querywise greedy stochastic
target swaps the Newton-score and Gradient-score denominator dispatch in
[`querywise_targets_impl.h`](../cuda/targets/querywise_targets_impl.h). Newly
connected greedy Simple leaves preserve that source behavior; symmetric Simple
retains the normal dispatch described above. Existing Metal
greedy Gradient/Newton checkpoints retain their prior mapping pending the
numerical parity card. Simple exports also preserve CUDA's signed weak weights
when query curvature is negative. Model and snapshot admission of signed weights
is restricted to the expected Simple configuration, with separate rejection
tests for iterative configurations.

The source audit and host runner checks do not establish NVIDIA execution
equivalence, full CUDA feature parity, numerical parity, identical device RNG or
performance parity. Numerical agreement, scale/performance and the separate
model-based analysis workflow remain the subsequent cards.

## Separate model-based feature evaluation workflow

The separate `model-based-eval` CLI workflow remains an identified API omission:
CUDA implements `TGPUModelTrainer::ModelBasedEval`, while Metal's override rejects
it. It runs feature-ablation experiments rather than ordinary training or CV.
The separate [card 7](../../Kanban/07-model-based-feature-analysis.md) covers supported scalar registrations,
the required single evaluation pool, experiment feature sets and baseline
snapshot behavior, plus output files and rejection cases. It is not implemented
as part of the ordinary training-option changes above.
