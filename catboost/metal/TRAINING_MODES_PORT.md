# Native Metal training modes

**Card 3 is complete and checkpoint `20260914T000929Z` is installed.**
[The training modes card](../../Kanban/03-training-modes.md) adds greedy YetiRank,
Ordered and FeatureParallel query/ranking training, Combination losses and native
custom Metal objectives. Full, alternate and installed acceptance passed against
the same frozen sources. The preceding checkpoint `20260913T220233Z` remains
intact. The source recovery archive starts from card 2 commit `f0a029e742`
on `metal-m3`.

## Registered scope and interfaces

The local CUDA trainer registrations define this matrix. Native entry points are
`CatBoostRegressor`, `CatBoostClassifier` and `CatBoost` with `task_type="GPU"`.
The [native adapter](train_lib/train.cpp) validates the objective, tree policy,
partition and estimator together before creating a Metal session.

| Objective family | Plain DocParallel | Plain FeatureParallel | Ordered FeatureParallel |
|---|---|---|---|
| QueryRMSE, QuerySoftMax, PairLogit | Symmetric; existing Depthwise, Lossguide and Region routes | Symmetric, including native compound CTRs | Symmetric, including native compound CTRs |
| Classic YetiRank | Symmetric; this card adds Depthwise, Lossguide and Region | Symmetric, including native compound CTRs | Symmetric, including native compound CTRs |
| Combination of supported scalar/query components | Symmetric | Symmetric, including native compound CTRs | Symmetric, including native compound CTRs |
| Scalar custom per-object Metal objective | Symmetric | Symmetric, including native compound CTRs | Symmetric, including native compound CTRs |
| PairLogitPairwise, QueryCrossEntropy, YetiRankPairwise | Existing symmetric full-matrix trainers | Unregistered | Unregistered |

CUDA registration sources are
[`querywise.cpp`](../cuda/train_lib/querywise.cpp),
[`querywise_non_symmetric.cpp`](../cuda/train_lib/querywise_non_symmetric.cpp),
[`querywise_region.cpp`](../cuda/train_lib/querywise_region.cpp), and
[`train_template_pointwise.h`](../cuda/train_lib/train_template_pointwise.h),
with Combination in [`combination.cpp`](../cuda/train_lib/combination.cpp) and
custom per-object in [`pointwise.cpp`](../cuda/train_lib/pointwise.cpp).
Combination and custom objectives have no greedy registration. Compound CTRs
require symmetric FeatureParallel; vector, full-matrix and greedy trainers do
not use that scheduler.

The standalone Python estimators expose the applicable greedy YetiRank and
Ordered query/ranking routes through their existing numeric, one-hot and simple
CTR preparation. Their fixed candidate banks remain separate from native dynamic
categorical search. This report does **not** claim a standalone Plain
FeatureParallel frontend, standalone compound CTR generation, or a standalone
Combination/custom-objective frontend. The latter two objectives use native
CatBoost and the additive runtime interfaces.

Defaults remain Plain boosting, `max_ctr_complexity=1` and four permutations.
Ordered or explicit complexity greater than one defaults to FeatureParallel;
other Plain training defaults to DocParallel. Ordered supports Cosine and
NewtonCosine scores. The new scalar Plain routes retain L2, Cosine, NewtonL2,
NewtonCosine, SolarL2, LOOL2 and SatL2 subject to the shared option validator.
Greedy training retains its policy limits and rejects MVS. Object bootstrap
sampling remains the supported unit for these newly connected objectives;
group-preserving CTR histories are a separate setting.

The new symmetric QueryRMSE, QuerySoftMax and PairLogit routes accept Newton,
Gradient and one-iteration Simple leaves. Combination and custom accept the
same methods. Existing greedy query routes retain Newton/Gradient. Exact is rejected
for those objectives and for native GPU Ordered. Classic YetiRank retains
Newton leaves, No backtracking, classic mode, query size at most 1023 and the
existing PFound target-domain checks. Other new objectives allow No,
AnyImprovement and Armijo backtracking. Fold-size normalization and adding ridge
to the objective default to false; the adapter retains its existing restriction
to Ordered fold normalization and rejects ridge-objective/meta-L2 options.

## Leaf and target contracts

Ordered keeps an independent cursor for each prefix/quality task and the full
estimation task. Query boundaries remain intact under permutation and slicing.
QueryRMSE centers residuals within the complete query; QuerySoftMax computes its
query normalizer at the task's current prediction. PairLogit uses the prepared
pair endpoints and incident pair mass for Gradient weights. Its reporting
denominator is pair mass, while QuerySoftMax reports against weighted target
mass. Those metric denominators do not replace the leaf task's weight sum.

`Simple` has different source semantics in the two partitions:

| Partition | One-iteration Simple behavior |
|---|---|
| Symmetric DocParallel | Skip independent leaf estimation. Export the sampled weak target's sum divided by its sampled weak weight plus L2, with the source empty-partition rule. Copy the chosen weak tree across dataset permutations. The score family determines original-weight versus curvature denominators. |
| Plain or Ordered FeatureParallel | Run the leaf estimator's non-Newton branch for one Gradient-style step. Use original parent-target weights, including the outer Combination target's weights, and applicable task normalization. |

DocParallel Simple exports the weak-statistic weights themselves. Combination's
negative Yeti contribution can make these signed; finite signed values are
preserved for this specific route. Regular leaf estimation continues to export
the original target weights. Top-level YetiRank cannot select Simple because the
shared options require its Newton method.

Sources:
[`doc_parallel_pointwise_oblivious_tree.h`](../cuda/methods/doc_parallel_pointwise_oblivious_tree.h),
[`oblivious_tree_doc_parallel_structure_searcher.cpp`](../cuda/methods/oblivious_tree_doc_parallel_structure_searcher.cpp),
and [`oblivious_tree_leaves_estimator.cpp`](../cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.cpp).

## Combination and custom objectives

[Combination preparation](train_lib/combination.h) accepts the twelve scalar
components RMSE, Logloss, CrossEntropy, Poisson, Huber, Expectile, Lq, Tweedie,
LogLinQuantile, Quantile, MAE and MAPE, plus QueryRMSE, QuerySoftMax, PairLogit and
classic YetiRank. The shared syntax requires at least two indexed loss/weight
entries and at least one nonzero coefficient. Zero coefficients are skipped;
Metal accepts up to 128 active components with finite positive input weights.
Nested, vector, full-matrix and custom components are rejected.

The [GPU target](native/metal_combination_runtime.h) accumulates query components
before pointwise components, preserving declaration order inside each group.
It forms unnormalized weighted sums of derivatives and weak-target denominators.
As in CUDA, each YetiRank coefficient is negated exactly once. A Combination
containing PairLogit or YetiRank does not inherit the outer objective's
zero-average leaf flag. This sign and ordering contract also applies to multiple
Yeti components.

Reported Combination values sum each component's **finalized metric**, including
the square root for RMSE, while leaf optimization sums raw objective values.
Yeti's private optimization value is zero; its public component metric is
negative weighted PFound. Default PFound decay is 0.85, independently of the
Yeti stochastic target's decay. An explicit Combination evaluation description
uses its own components. This reporting behavior is a coherent extension around
the pinned CUDA metric defect described below.

The pinned shared defaults contain a preserved quirk: the Combination
`haveDefaults` flag is never set, so the **last nonzero component** supplies the
leaf method and iteration defaults, and its L2 default is multiplied by its
coefficient. This report does not reinterpret those defaults as an aggregate.
See [`catboost_options.cpp`](../private/libs/options/catboost_options.cpp) and
[`combination_targets_impl.h`](../cuda/targets/combination_targets_impl.h).

FeatureParallel uses **one batched leaf walker** over all estimation tasks. Each
trial evaluates every task and component, sums the task objectives, and accepts
or rejects one shared step. DocParallel finishes an independent walker for each
dataset permutation. These different orders affect both leaf values and Yeti
seeds. Combination backtracking requests stochastic seeds on demand so rejected
trials consume exactly the calls they execute; unused trial capacity consumes
none.

Custom per-object training compiles a dedicated Metal function body from
`calc_ders_range_metal()`. The body returns weighted maximized value,
negative-loss derivative and nonnegative curvature; it does not call Python for
row derivatives. The [custom shader contract and example](docs/custom_objectives.md)
define source limits, output validation, the separate reporting metric and error
behavior. CPU callbacks and Numba CUDA kernels retain their own interfaces and
cannot substitute for Metal source. Models contain ordinary fitted trees and
can be read without the custom objective object.

## Categorical search and seeded continuation

Native query, ranking, Combination and custom routes reuse card 2's
[compound CTR machinery](COMPOUND_CTR_PORT.md): begin/grow/finish preserves
resident target statistics and cursors while appending permutation banks and
changing candidate activity. Selected grids and registered feature identities
persist as required; inactive columns remain available for existing splits and
cursor routing. Group histories exclude the complete current query. FeatureFreq
uses full-learn counts, and exported final CTR tables use complete learn data.
The existing CTR type, prior, counter mode and software memory restrictions remain
in force. Greedy Yeti uses its existing simple categorical bank path.

Host seed accounting follows the local CUDA call order:

| Route | Shared host draw order |
|---|---|
| Symmetric DocParallel Yeti | Constructor base draw; weak target; first bootstrap cache when enabled; attempted search draws; permutation-major leaf evaluations |
| Greedy DocParallel Yeti | Constructor base draw; first bootstrap cache when enabled; weak target; actual uncached leaf-search batches; permutation-major leaf evaluations |
| Plain/Ordered FeatureParallel Yeti | History chooser; weak target calls; first mirror bootstrap cache, including bootstrap No; static/dependent/dynamic search calls; evaluation-major leaf tasks |

Ordered weak target seeds include both learn and quality calls, even an empty
quality slice. Within each call, Yeti components follow their target order.
FeatureParallel bootstrap initialization consumes 65,537 host draws. A search
attempt consumes the static dataset draw, an additional dependent simple-CTR
dataset draw when present, and a tree-visitor draw when dynamic packs are active.
Greedy counts uncached eligible leaf-search batches rather than installed depth;
cached winners consume no new scorer seed.

Pure Yeti consumes one leaf evaluation for one iteration, otherwise I+1,
including the final unused derivative evaluation. With Combination backtracking,
I>1 executes between I+1 and max(I,100)+1 evaluations per walker. The
[FeatureParallel helper](train_lib/feature_parallel_yeti_random.h) requires full
task/component groups; the [DocParallel helper](train_lib/yeti_random.h) permits
different trial counts for separate permutation walkers. Both retain actual host
draw counts and validate phase, component counts and bounded snapshot state.

Snapshots retain prediction banks, Ordered prefix descriptors/cursors, compound
registry state and host RNG state. Greedy Yeti adds actual search-count metadata;
DocParallel Combination with Yeti adds its tagged target RNG payload. Custom
identity includes the exact shader bytes. A resume checks data/options identity
and saved counts, reconstructs the host RNG by advancing its stream, and resumes
without replaying completed target evaluations. Existing modes keep their
untagged payload behavior. `init_model` starts a new training segment and fresh
random/compound state from initial predictions; it is not snapshot continuation.

## Explicit source differences

The implementation preserves CUDA registrations and audited host scheduling,
while retaining or applying these documented corrections to the pinned source:

- CUDA's Yeti query-centering caller passes query count where row count is
  required. Metal uses complete-query centering under its established policy.
- CUDA FeatureParallel's final zero-average block reads newly allocated zero
  leaves before filling them, so its computed bias is zero. Metal's direct
  PairLogit/Yeti routes use the established arithmetic leaf-centering behavior;
  outer Combination remains uncentered.
- CUDA Combination's non-Newton FeatureParallel path can pass an empty curvature
  scratch buffer to Yeti despite the kernel requiring a full buffer. Metal
  provides valid component scratch storage for those registered combinations.
- CUDA's Yeti target selects PFound, but its `ComputeStats` switch omits PFound;
  Combination can reach that failure. Metal reports the finalized signed PFound
  component and honors an explicit Combination evaluation description.

Source locations are
[`kernel.h`](../cuda/targets/kernel.h),
[`querywise_targets_impl.h`](../cuda/targets/querywise_targets_impl.h),
[`combination_targets_impl.h`](../cuda/targets/combination_targets_impl.h), and
[`oblivious_tree_leaves_estimator.cpp`](../cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.cpp).
The negative Yeti coefficient and Combination defaults quirk are preserved
source behavior, not corrections. CUDA device RNG, bootstrap/noise streams,
close-score ties and device-pack order remain separate parity work. **No NVIDIA
execution or live Metal-versus-CUDA comparison has been performed for this card.**

Signed Combination statistics follow the source reduction order: score noise
sums signed weighted contributions before taking the final square root. Ordered
scoring clamps the complement bucket's aggregate mass, with the bucket selected
according to numeric versus one-hot split orientation. A finite nonpositive
regularized Newton diagonal produces a zero direction. The custom-objective
contract separately requires nonnegative curvature on every observation.

Two shared compatibility fixes support these modes. Combination metric names now
use the standard formatter, retaining parseable component parameters and distinct
weighted/unweighted evaluation configurations. Quantized categorical model
readers widen compressed 8/16-bit bins while retaining the existing 32-bit path,
offsets and perfect-hash mapping. Dedicated C++ regressions cover both fixes.
The native scalar API also uses a separate C++ exception boundary so CatBoost's
vendored runtime preserves specific validation messages; an inner Objective-C
handler retains Metal exception reasons.

## Acceptance and release

Final acceptance used the standard and alternate native extensions, CLI and
standalone runtime built from 655 frozen source files. All pytest selections
finished with zero failures, errors or skips. Counts overlap; do not sum them.

| Check | Final result |
|---|---|
| Full native/standalone regression | **12,745 passed + 16 subtests**, 643.77 s |
| Alternate extension regression | **4,901 passed**, 457.62 s |
| Installed-package acceptance | **1,267 passed** |
| Installed GPU smoke configurations | **134 passed**: 34 preceding + 100 new |
| CLI configurations | **287 passed**: 187 preceding + 100 new |
| Preserved prior snapshots | **244 exact recoveries**: 214 preceding + 30 from `20260913T220233Z` |
| Host Metal helper/RNG units | **54 passed** |
| Shared quantized categorical reader units | **3 passed**, covering 8/16/32-bit storage |
| Shared Combination metric description units | **2 passed** |
| Native exception boundary proof | **Passed**; original source, commands, binary and result preserved |

The four new native acceptance files contribute 882 cases included in the full
and alternate matrices: [training modes](tests/test_native_training_modes.py)
(359), [greedy Yeti](tests/test_native_greedy_yeti.py) (174),
[Combination](tests/test_native_combination.py) (246) and
[custom objectives](tests/test_native_custom_objective.py) (103). They cover
baselines, initial models, callbacks, early stopping, best-model trimming, exact
snapshots, raw/quantized Pools, CBM/JSON readers and independent compound final
tables. Independent runtime/public tests cover query/pair arithmetic, per-task
histories, global versus independent trial acceptance, categorical activity and
actual seed continuation. Training acceptance requires Metal dispatch. No CPU
CatBoost fit or live NVIDIA execution was used.

The initial complete regression exposed two test-fixture issues: a private
ctypes wrapper changed signatures shared with public runtime tests, and an older
guard still expected newly supported Ordered PairLogit to fail. The private
wrapper now owns separate function bindings, with an isolation regression; the
guard checks unsupported Ordered PairLogitPairwise. The final source freeze and
all acceptance above include those corrections. Earlier diagnostic failures are
preserved separately and are not counted as final acceptance.

Checkpoint: `20260914T000929Z`, installed in `catboost/metal/.venv`.
Wheel: `catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl`.

| Artifact | SHA256 |
|---|---|
| Wheel | `0514c3b4414b5b9dd197fd40d9b84fae95e362f0e9bd0e201c3d7bcc3cf436a5` |
| Standard and installed extension | `431e44e746ee8418a5b81560658728af40d187bebcd767c63cb2b1f04bf8170c` |
| Alternate extension | `48f10b00994deb2158a52d724cd735b0dbb49cece8d459011d6e9704dd74dbde` |
| CLI | `a49a5689cbb50c58f66976f8a7eaf13cb874ada9c95b43d6c5bb8c697b92c59c` |

The local release archive is
`catboost/metal/.build/releases/20260914T000929Z/`. It retains both extensions,
the CLI, wheel, manifest, source patch/overlay, original old snapshots and raw
acceptance evidence. Reconstruct source from
`f0a029e742179a65b84b77e9af1d7b2a4844eded`, apply `source-changes.patch`, then
copy `source-overlay/` onto the checkout. Reconstruction is verified against all
655 frozen source hashes. `acceptance/` records commands, XML, logs, fixture
identities and installed binary verification; `RECOVERY.md` gives restoration
steps. [Reproduction helpers](examples/training_modes_acceptance/README.md)
replay the CLI, smoke and snapshot matrices. Build artifacts remain local.

The remaining work is tracked in [API/options card 4](../../Kanban/04-api-and-options.md),
[numerical agreement card 5](../../Kanban/05-numerical-agreement.md) and
[scale/performance card 6](../../Kanban/06-scale-and-performance.md).
