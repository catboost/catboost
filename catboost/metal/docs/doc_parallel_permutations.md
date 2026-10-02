# CUDA Plain DocParallel permutation contract

Source audit for the Metal multiple-permutation port, 2026-09-12. This describes
the checked-out CUDA implementation; it is not Ordered boosting. CUDA rejects
Ordered with DocParallel in `private/libs/options/boosting_options.cpp:72`.

## Dataset layout and permutation generation

The effective permutation count defaults to four
([boosting_options.cpp](../../private/libs/options/boosting_options.cpp):14).
GPU defaults reduce it to one for Plain training with no CTR-eligible or online
estimated features ([train.cpp](../../cuda/train_lib/train.cpp):87-114).
`has_time` also forces one
([catboost_options.cpp](../../private/libs/options/catboost_options.cpp):1042).

Three different orders must remain distinct:

1. **Incoming training-provider order.** Shared CatBoost preprocessing already
   applies timestamp ordering and, when appropriate, a user-seeded learn-data
   shuffle before entering the GPU trainer. The latter uses
   `TRestorableFastRng64(random_seed)` and preserves groups. This happens for
   categorical data unless `has_time` is set. An internal identity permutation
   means identity relative to this provider, not necessarily the original Pool.
   See [train_model.cpp](../../libs/train_lib/train_model.cpp):1064-1076 and
   [preprocess.cpp](../../private/libs/algo/preprocess.cpp):160-203.
2. **Load-balancing order.** Targets, sample weights, baselines, all learn
   cursors, and every dataset's feature columns share this one row order. It is
   normally identity. With groups and shuffling enabled, CUDA chooses permutation
   ID 42, overrideable by `CB_LOAD_BALANCE_PERMUTATION`.
   See [doc_parallel_dataset.h](../../cuda/gpu_data/doc_parallel_dataset.h):149-160
   and [dataset builder](../../cuda/gpu_data/doc_parallel_dataset_builder.cpp):32-53.
3. **CTR history order `order[p]`.** Each permutation changes which prior rows
   contribute to target-dependent CTRs. It does not reorder that permutation's
   target or cursor arrays. Permutation zero is identity. The dataset builder
   calls `GetPermutation(provider, p)` with its default **block size 1**. The
   GPU option normalization to `fold_permutation_block=64` is not passed to
   this DocParallel constructor.

Exact nonidentity CTR-order generator:

```text
seed = uint32(1664525 * p + 1013904223 + block_size)
rng = CatBoost TRandom(seed)  # TMersenne<uint64>, MT19937-64
rng.Advance(10)
order = [0, 1, ..., n - 1]
for i in 1 .. n - 1:
    swap(order[i], order[rng.Uniform(i + 1)])
```

The expression in `GetSeed()` wraps as uint32 before conversion to its ui64
return type. `Uniform` uses the upstream rejection algorithm, not simply
`NextUniformL() % bound`. For block sizes other than one, shuffle block IDs and
preserve order within each block. For grouped data, shuffle group IDs and append
each group's rows in their existing order. These internal seeds depend on
permutation ID and block size, not directly on `random_seed`; the user seed
already affected incoming provider order.

Sources: [permutation.h](../../cuda/data/permutation.h):93-103,
[permutation.cpp](../../cuda/data/permutation.cpp):7-22,
[data_utils.h](../../cuda/data/data_utils.h):22-63,
[cpu_random.h](../../libs/helpers/cpu_random.h):6-32,
[shuffle.h](../../../util/random/shuffle.h):23-33,
[common_ops.h](../../../util/random/common_ops.h):49-61.

## CTR buffers, shared grids, and test data

Numeric, one-hot, FeatureFreq, and other permutation-independent columns can
share storage. Borders, Buckets, and FloatTargetMeanValue columns require one
history-based column per permutation. A tensor containing a permutation-dependent
split is also dependent. For `P=1`, all columns use the independent storage
shortcut but target CTRs still use exclusive identity-order histories.

For each feature configuration, construct its grid once from permutation zero's
**learn-only history values**, then reuse the same borders, feature IDs, and
candidate border IDs for every other permutation and test dataset. The builder
visits permutations in ascending order and `TGpuBordersBuilder` caches borders
by feature ID. CUDA uses the configured grid builder and inserts border `0.5`
when it returns an empty grid. Independently quantizing each permutation would
make a shared tree split mean different thresholds.

Test CTRs are computed only during permutation zero's pass, after all learn rows
in each category; they use full learn statistics, never test targets. Default
FeatureFreq uses learn counts. The explicit `counter_calc_method=Full` option
includes test category frequencies and is a separate supported-data policy.

The CTR calcer gathers original-provider targets into stable category/history
order, computes exclusive segmented histories, then scatters results back to
provider row IDs. Compressed-column writing applies the common load-balancing
gather afterward. CUDA CTR weights are one for learn and zero for test,
independently of training sample weights; group-history mode additionally
excludes the entire current group. Multi-output CUDA's CTR target builder
currently selects target dimension zero.

Sources: [dataset_helpers.cpp](../../cuda/gpu_data/dataset_helpers.cpp):6-46,154-173,
[ctr_type.cpp](../../private/libs/ctr_description/ctr_type.cpp):44-58,
[binarizations_manager.cpp](../../cuda/data/binarizations_manager.cpp):134-146,
[dataset builder](../../cuda/gpu_data/doc_parallel_dataset_builder.cpp):200-261,
[CTR calcer](../../cuda/ctrs/ctr_calcers.h):127-148,174-193,
[grid cache](../../cuda/gpu_data/gpu_binarization_helpers.cpp):6-54,
[CTR writer](../../cuda/gpu_data/dataset_helpers.h):206-230.

## One structure, separate leaf estimates and cursors

Keep `P` ensembles and `P` learn cursors. Initialize each cursor from the same
baseline or optimum constant. The final estimation permutation is `P-1`.

At iteration `t`:

```text
base_iteration_seed = first fresh TGpuAwareRandom(random_seed).NextUniformL()
rng = TRandom(uint64(t + base_iteration_seed))
rng.Advance(10)
L = P - 1 if P > 1 else 1
search_p = rng.NextUniformL() % (L - 1) if L > 1 else 0
structure = search(features[search_p], target, cursor[search_p])
for p in 0 .. P - 1:
    leaves[p] = estimate(structure, features[p], target, cursor[p])
    leaves[p] *= learning_rate
    cursor[p] += apply(structure, leaves[p], features[p])
    ensemble[p].append(structure, leaves[p])
test_cursor += apply(structure, leaves[P - 1], test_features)
```

**Preserve or explicitly change the surprising chooser expression.** The source
uses modulo `L-1`, not `L`. Thus `P=1,2,3` always searches permutation zero;
`P=4` searches zero or one; `P=5` searches zero, one, or two. The penultimate and
final permutations are never used for structure search when `P>=3`. This audit
does not establish why that expression was chosen. Silently replacing it with
modulo `P-1` would diverge from the current CUDA implementation.

Leaf estimation above is conditional on the weak learner's `NeedEstimation()`;
when false, CUDA retains the structure searcher's model values in all copies.
The standard separate-estimation path evaluates each task's own current cursor
and feature bins. Reusing one permutation's gradients or leaves for all datasets
would change the training algorithm.

Learn metrics use cursor `P-1`; test metrics use the single test cursor. Export
only ensemble `P-1`, adding the common starting bias. Consequently the exported
model's prediction on learn rows, which uses full CTR tables, need not equal
the learn cursor, which used exclusive training histories. Test predictions
should agree with the exported model's full-table inference.

Sources: [boosting](../../cuda/methods/doc_parallel_boosting.h):101-103,137-185,
268-291,315-414,453,526-528; [fresh trainer RNG](../../cuda/train_lib/train.cpp):238;
[trainer construction](../../cuda/train_lib/train_template.h):19-25,94.

## Snapshot and Metal integration contract

CUDA snapshots save **all `P` ensembles**, a feature map containing CTR configs
and grids, plus the best test cursor when requested. The outer tracker saves
options, metric history, profiling history, and data checksums. Normal learn and
test cursors are rebuilt by replaying each saved ensemble on its corresponding
dataset. The starting bias is regenerated. These snapshot structures do not
serialize `BaseIterationSeed`, the permutation orders, or complete mutable RNG
state; the base seed and orders are regenerated from deterministic construction.
This observation alone does not establish restart parity for every stateful
bootstrap implementation.

For Metal, a reviewable dataset/session interface should provide:

- One shared candidate map/grid and common target, weight, and row ordering.
- `P` feature-bin views sharing independent columns, with explicit history
  orders and dataset fingerprints for restore validation.
- Separate operations to search a structure and estimate/apply a supplied
  structure, permitting the per-permutation loop above.
- `P` cursors and leaf ensembles; a declared final estimation index `P-1`.
- Deterministic seed/iteration selection plus persisted sampler state needed by
  Metal's existing exact-resume contract; do not save only the exported ensemble.

Acceptance checks should cover effective `P=1,2,3,4`, exact native permutation
vectors and chooser traces, one shared structure with distinct per-permutation
leaf estimates, fixed CTR borders across permutations, unchanged histories when
future targets change, test/export agreement, and uninterrupted versus resumed
training. Numeric-only training retains the one-permutation shortcut.

Sources: [progress payload](../../cuda/methods/doc_parallel_boosting_progress.h):10-35,
[feature map](../../cuda/methods/serialization_helper.h):23-76,
[snapshot tracker](../../cuda/methods/boosting_progress_tracker.cpp):169-198,241-255,
[cursor replay](../../cuda/methods/doc_parallel_boosting.h):226-263,329-335,483-528.
