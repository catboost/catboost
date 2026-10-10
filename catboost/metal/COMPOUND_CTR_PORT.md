# Compound categorical CTRs on Metal

Native CatBoost now connects tree-dependent categorical combinations to scalar
symmetric training with Plain or Ordered FeatureParallel. Explicit
`max_ctr_complexity=2` or `3` can introduce category combinations and category
projections conditioned on numeric or one-hot splits during the current tree.
The option is a maximum; training selects only combinations that win a split.

**Checkpoint `20260913T220233Z` is installed and accepted.** The full matrix
passes **11,323 tests plus 16 subtests**, the alternate extension passes 3,892
tests, and all 41 named C++ checks pass. The release also passes 214 exact
prior snapshot recoveries, 187 CLI configurations, 385 installed-package tests
and 34 installed GPU smoke configurations.

## Native support

| Capability | Supported surface |
|---|---|
| Entry point | Native `CatBoostRegressor` / `CatBoostClassifier` / `CatBoost` with `task_type="GPU"` and `grow_policy="SymmetricTree"` |
| Boosting | Plain or Ordered with `data_partition="FeatureParallel"`; explicit `max_ctr_complexity > 1` selects this partition by default |
| Objectives | RMSE, Logloss, CrossEntropy, Poisson, Huber, Expectile, Lq, Tweedie, LogLinQuantile, Quantile, MAE and MAPE |
| CTR types | Borders, Buckets, FloatTargetMeanValue and FeatureFreq, including supported `combinations_ctr` priors and binarization settings |
| Histories | Sample and Group; FeatureParallel permutations preserve whole groups and use original source-row identities |
| Leaves and scores | Existing scalar Plain methods/scores, including Exact for Quantile/MAE/MAPE; Ordered retains Newton/Gradient and Cosine/NewtonCosine |
| Data and lifecycle | Raw or quantized Pools, numeric and one-hot predicates, validation, baselines, initial models, callbacks, early stopping, best-model trimming and snapshots |
| Model readers | Native GPU prediction and standard CBM/JSON exports, including unseen constituents and unseen joint tuples |

Learn-only final frequency tables require `counter_calc_method="SkipTest"`.
Automatic prior estimation and unsupported CTR types retain explicit errors.
Compound DocParallel, non-symmetric, multiclass, multioutput and ranking
configurations are rejected. The standalone Python estimators keep their
existing categorical surface; this integration enables the native interface.

## Tree search and retained feature grids

[tree_ctr_tensors.h](train_lib/tree_ctr_tensors.h) follows CUDA's
`cuda/methods/tree_ctrs.cpp` scheduler. A selected CTR contributes its underlying
projection. Numeric and one-hot splits accumulate a pure-tree predicate tensor.
The scheduler crosses eligible base tensors with categories, avoiding repeated
categories. Complexity counts categorical constituents plus one for any binary
predicate component. New numeric/one-hot splits replace the preceding pure-tree
packs; already admitted CTR-base packs remain available within the tree.

[tree_ctrs.cpp](train_lib/tree_ctrs.cpp) hashes those projections and computes
exclusive histories on Metal. Every generated column has an append-only global
feature ID and a bank for each permutation. A new border grid comes from the
selected search history; all banks for that feature use the same grid. A
selected configuration, or an eligible eagerly cached category-only tensor,
retains its registered grid in subsequent trees. Other configurations resolve
to the current history's grid variant. Previous columns remain available for
existing splits and cursor routing even when their candidates become inactive.
Configuration identity includes the CTR binarization ID, which the standard
exported `TModelCtr` does not contain.

[tree_ctr_session.h](train_lib/tree_ctr_session.h) connects that registry to
the scalar [native runtime](native/metal_trainer.mm) and the additive
[Ordered runtime](native/metal_ordered_trainer.mm). At each tree boundary it
deactivates old dynamic candidates while retaining their banks. Between split
searches it appends new banks and applies the scheduler's current activity mask.
The final selected CTR is registered without generating an unused next-depth
pack. Derivatives, bootstrap draws, selected histories, leaf partitions and
prediction cursors stay resident through begin/grow/finish. Ordered updates each
prefix and the independent full-estimation task through the appropriate bank.

FeatureParallel distinguishes dynamic, active, registered and used columns.
Only installed dynamic CTR splits enter the used-CTR set; simple CTR penalties
persist after selection. Dynamic CTRs also retain their size penalty after use.
The dynamic denominator depends on currently active unused tree CTRs; the static
denominator additionally considers registered unused CTRs. Expired transient
grids do not enter the latter set. Plain applies categorical penalties after
Cosine noise; Ordered applies its categorical penalty before noise. The exact
formulas and upstream call sites are in
[feature_parallel_ctr_scores.md](docs/feature_parallel_ctr_scores.md).

[ordered_random.h](train_lib/ordered_random.h) preserves the shared host chooser
stream for both compound boosting modes. Each attempted depth consumes the
static dataset draw, an additional draw for a permutation-dependent simple CTR
dataset when present, and the tree visitor draw when dynamic packs are active.
The failed or duplicate final attempt also counts. Bootstrap initialization and
chooser draws retain their CUDA host ordering. This accounting does not establish
agreement with CUDA's device bootstrap/noise streams or device-pack tie order.

## Group histories, export and continuation

Group histories exclude the complete current group from target statistics.
The helper derives dense group ordinals from the Pool and invokes the existing
grouped GPU CTR kernels. Histories must keep each query contiguous. FeatureFreq
uses full-learn counts and therefore does not change with Sample versus Group
history policy. Final model statistics come from the complete learn data;
projection tables contain standard 64-bit model hashes, including applicable
numeric/one-hot predicates. The shared CTR provider is attached to each progress
model and the final forest, so validation metrics, GPU evaluation and exported
readers use the same final tables.

The native snapshot keeps its existing v6 common payload and adds a tagged
compound-CTR payload only when that mode is enabled. It preserves registry
descriptors, border grids, registered/used/activity metadata, all prediction or
Ordered prefix cursors, and the FeatureParallel chooser state. Helper format v3
adds history-unit and group-boundary identity; it can read the preceding helper
v2 format for Sample histories. A restore checks the full data/options identity,
regenerates banks in their original order using the saved grids, and validates
metadata against the reconstructed registry. Completed snapshots are checked
even when no further iterations are requested.

Snapshot extension resumes the saved training state exactly. `init_model`
starts a new training segment with an initial prediction cursor, a fresh
compound registry and a fresh chooser stream, following the adapter's existing
initial-model behavior. It is not equivalent to resuming a snapshot. The
`max_ctr_complexity=1` routes, feature layouts and untagged native v6 snapshot
bytes remain unchanged; preserved fixtures test that compatibility separately.

## Acceptance evidence

| Check | Final checkpoint result |
|---|---|
| Both native extensions, CLI and `model_ut` build | Passed; the final incremental build reports no work remaining |
| C++ tensor scheduler, history/permutation, host-RNG, completed-snapshot metadata and GPU helper suites | **41 passed**: 8 tensor, 5 permutation, 5 host-RNG, 9 snapshot-metadata and 14 helper cases |
| [Scalar dynamic runtime](tests/test_dynamic_session.py), [dynamic scores](tests/test_dynamic_scores.py), [Ordered dynamic runtime](tests/test_ordered_dynamic_runtime.py) | **238 passed**: 142 scalar runtime, 36 scoring and 60 Ordered runtime cases; included in the full matrix |
| [Native compound acceptance](tests/test_native_compound_ctrs.py) | **148 passed**; included in the full matrix |
| Full native/standalone regression | **11,323 passed plus 16 subtests**; tested source hashes remain unchanged |
| Alternate extension matrix | **3,892 passed** |
| Preceding native/standalone snapshots | **214 exact recoveries**: 12 newly preserved old-build cases plus the preceding 202 original fixtures |
| CLI acceptance | **187 configurations passed**: 171 previous plus 16 compound CTR configurations |
| Installed-package acceptance | **385 tests plus 34 GPU smoke configurations passed** |
| Installed checkpoint | `20260913T220233Z`; installed extension matches the frozen tested binary |

The native suite requires models to select joint categorical and mixed-predicate
projections. It independently reconstructs final CTR sufficient statistics from
original rows and traverses exported leaves, rather than trusting the helper's
own tables. It also covers complexity 2/3, P1/P4, all four CTR types, both history
units, raw/quantized Pools, initial models, baselines, model readers, retained-grid
snapshot extension, best-model state and explicit rejection boundaries. Runtime
tests cover interleaved append/activity changes, distinct permutation banks,
metadata recovery and rejection of incomplete-tree snapshot operations.

Test selections overlap and must not be summed as a release total. The
[acceptance helpers](examples/compound_ctr_acceptance/README.md) preserve package
identity, original snapshots, outputs and checksums. GPU fits are required;
standard CPU prediction is used only to check model-reader interoperability.

Checkpoint `20260913T220233Z` is installed in `catboost/metal/.venv`:

- Wheel: `catboost-1.2.10-cp312-cp312-macosx_11_0_arm64.whl`
- Wheel SHA256: `f1d79ee43aeebfbb65341ac70d10caf83150636f0663c00b11537bf24ba41ffe`
- Standard extension SHA256: `6b47372ba5ae7f37dd9987490436580a4b7525c01d8fdef59651ad804d537a87`

The release preserves both native extensions, CLI, source reconstruction patch
and added files, build/test commands, raw reports and original snapshot fixtures.
The preceding installed checkpoint `20260913T195124Z` remains intact.

## Reproduction

Run from the repository root on an Apple Silicon Mac. Follow the
[native build prerequisites and wheel packaging instructions](README.md#build-native-catboost-with-metal)
using a dedicated environment. The existing configured native build can be
updated with:

```sh
ninja -C /tmp/catbooster-native-build -j 4 \
  _catboost _catboost_no_cuda catboost model_ut
```

The build directory above is the local acceptance build; a separately configured
build should use its own directory. Run the named C++ suites from that build:

```sh
/tmp/catbooster-native-build/catboost/libs/model/ut/model_ut \
  +TMetalTreeCtrTensors \
  +TMetalFeatureParallelPermutations \
  +TMetalFeatureParallelTreeCtrRandom \
  +TMetalTreeCtrCompletedSnapshotMetadata \
  +TMetalTreeCtrFeatures
```

With the rebuilt native wheel installed in the selected environment:

```sh
PYTHONPATH=catboost/metal/python \
  catboost/metal/.venv/bin/python -m pytest -q \
  catboost/metal/tests/test_dynamic_session.py \
  catboost/metal/tests/test_dynamic_scores.py \
  catboost/metal/tests/test_ordered_dynamic_runtime.py

CATBOOST_NATIVE_METAL_TESTS=1 PYTHONPATH=catboost/metal/python \
  catboost/metal/.venv/bin/python -m pytest -q \
  catboost/metal/tests/test_native_compound_ctrs.py
```

For an isolated standard or alternate extension, put its package root before
`catboost/metal/python` on `PYTHONPATH`; verify it resolves to the intended
`_catboost.so`. The [helper README](examples/compound_ctr_acceptance/README.md)
provides fresh-directory commands for old-baseline capture/replay, combined
compatibility replay, CLI training and installed-wheel smoke checks. Release
evidence records the exact package roots and source hashes used.

## Remaining parity work

CUDA device RNG agreement, close-score/tie behavior, device-pack visitation
order, wider workloads and other M-series or NVIDIA hardware remain separate
validation work. Retained feature banks and runtime workspaces keep their
existing software memory limits; cache eviction, bank compaction, complete
aggregate memory accounting and large-scale performance remain open.
Query/ranking FeatureParallel training and remaining Ordered objectives are
tracked in [training modes card 3](../../Kanban/03-training-modes.md).
CUDA's DocParallel, vector and non-symmetric trainers do not use this dynamic
compound scheduler. Estimated text/embedding predicates and remaining training
options require separate integration. Tests fit models through Metal; live
NVIDIA comparison remains unperformed.
