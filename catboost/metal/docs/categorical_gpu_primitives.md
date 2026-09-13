# Categorical Metal primitives

The categorical training path retains original CatBoost category hashes and
standard model CTR tables. Shared CatBoost code prepares Pool dictionaries and
feature borders. Metal performs stable row grouping, exclusive category-history
scans, target-statistic reductions, prior application, and scatter back into the
original row order.

## Stable sorting

[`metal_sort.h`](../native/metal_sort.h) exposes `cbm_sort_u32` for host arrays
and `CBMEncodeSortU32` for resident Metal buffers. The resident form appends to
the caller's uncommitted command buffer, retaining private scratch buffers, and
performs no intermediate readback. Eight stable four-bit radix passes carry
arbitrary uint32 payloads; equal keys preserve their incoming order.

The Python entry point is `catboost_metal._sort.stable_sort`. Both the standalone
CTR wrapper and native categorical adapter use this GPU sort to preserve each
category's supplied history permutation.

## Compound projections

[`metal_projection.h`](../native/metal_projection.h) exposes
`cbm_projection_group`; the Python entry point is
`catboost_metal._projection.group_projection`. Feature matrices are
feature-major. Each projection component references either an original
categorical hash or a predicate on a quantized bin:

| Component type | Value mixed into the hash |
| --- | --- |
| 0 | Original category hash, sign-extended from int32 to uint64 |
| 1 | `bin > threshold`, represented as zero or one |
| 2 | `bin == threshold`, represented as zero or one |

Components follow the standard model order: categories, numeric predicates,
then one-hot predicates. Within each list, caller order is preserved. Starting
at zero, each component applies the exact model operation
`M * (hash + M * value)` modulo 2^64, where `M = 0x4906ba494954cb65`.

The GPU hashes original rows, stably sorts low 32-bit keys, gathers high keys,
then stably sorts high keys in the same command buffer. Full 64-bit equality
defines each category segment. The wrapper derives compact segment IDs with a
linear boundary scan and returns original-order hashes/bins plus grouped
hashes/row IDs. Distinct projections with equal low 32 bits remain distinct;
identical full hashes preserve the supplied history order.

This primitive provides identities and groups for compound CTR computation.
The native tree-CTR helper described below applies the legal CUDA tensor rules
and `max_ctr_complexity` separately from hashing.

## CTR histories and inference

[`metal_ctrs.h`](../native/metal_ctrs.h) accepts compact grouped category IDs
and original row IDs. Borders and Buckets consume binarized scalar targets;
FloatTargetMeanValue consumes float targets; FeatureFreq uses learn-only
category frequencies. Row weights for CTR statistics are one, following CUDA's
`BuildCtrTarget`, independently of tree-training sample weights.

Target-based training values use only earlier rows of the same category in the
supplied permutation. The current row and future targets are excluded. Final
category sums/counts are learned from all training rows and used for held-out
inference; unseen categories receive the configured prior. The native adapter
builds a standard `TStaticCtrProvider`, including fractional mean sums.

Multiple-permutation training uses one common feature grid, separate CTR
history columns, and a separate prediction cursor for each permutation. The
last permutation supplies exported leaf values. See the source audit in
[`doc_parallel_permutations.md`](doc_parallel_permutations.md).

## Tree-dependent CTR batches

[`tree_ctr_tensors.h`](../train_lib/tree_ctr_tensors.h) implements CUDA's
separate accumulated CTR projection and pure-tree predicate tensor. A selected
CTR contributes its underlying projection. A numeric or one-hot split replaces
the previous pure-tree candidate packs; admitted CTR-base packs persist within
the tree. Eligible base tensors cross with each non-one-hot category, excluding
duplicates and estimated predicates. Complexity is the number of categories
plus one when any binary predicates are present.

[`tree_ctrs.h`](../train_lib/tree_ctrs.h) exposes `TMetalTreeCtrFeatures`:

1. Construct it with the Pool, options, first unused runtime feature ID, shared
   provider, and explicit source-row history orders. Empty orders mean P1.
2. Call `BeginTree()` and deactivate all previously registered dynamic features
   in scoring, retaining their feature banks and model tables.
3. After each selected split, call `AddSplit(split, searchPermutation)`. Append
   its feature-major banks for every permutation. Candidate feature IDs are
   local to the batch; `FirstFeature` supplies their global offset.
4. Activate the returned `ActiveFeatures` global IDs along with static features.
   Expired pure-tree projections remain available to existing model splits.
5. Call `MarkSelected(globalFeature)` for every selected feature, including the
   final depth. `RegisteredCtrFlags` marks eagerly registered appended columns;
   `GetRegisteredFeatures()` gives the current registered dynamic feature IDs.
6. Use `GetSplit(globalFeature, bin)` and the shared provider when exporting the
   tree. Standard tables contain complete 64-bit projection keys directly.

Combination configurations use the shared CUDA `TCtrConfig` ordering and
deduplication. A newly computed grid comes from the selected search history's
learn values and is shared across that candidate's history banks. Selected
configurations and eligible eagerly cached category-only configurations retain
their registered grids in later trees. Unregistered configurations resolve to
the current search history's grid variant; previous variants retain their IDs
and banks but become inactive. Full configuration identity includes the
binarization configuration, which is absent from the exported `TModelCtr`.
Each history uses its own exclusive category statistics; the final model
table comes from permutation zero. Feature IDs, grids and tables persist across
tree boundaries. `Save`/`Restore` stores the registry and exact grids, verifies
history fingerprints, and regenerates banks in their original feature order
before the caller restores optimizer state.
The caller must first verify the snapshot's full training/evaluation data and
option fingerprints; helper history/configuration checks do not replace that
validation of targets, original categories, numeric predicates and metadata.

The helper has actual Metal tests for all four CUDA CTR types, compound category
and predicate projections, four histories, full/unseen inference, standard
JSON/CBM models, and snapshot restoration. Its trainer integration requires the
runtime's begin-tree/grow-depth/append-features/finish-tree state machine. The
helper alone does not enable `max_ctr_complexity > 1` in the public trainer.
The initial integration scope is Plain FeatureParallel P1; Ordered folds and
FeatureParallel's shared RNG consumers remain caller responsibilities.
See [`feature_parallel_ctr_scores.md`](feature_parallel_ctr_scores.md) for the
distinct dynamic/static penalty rules and Plain/Ordered noise ordering.

[`tree_ctr_permutations.h`](../train_lib/tree_ctr_permutations.h) provides the
FeatureParallel source-row orders needed for subsequent P4 integration. Unlike
DocParallel, blocks below 50,000 rows have size one; larger inputs round the
configured size up to a power of two and halve it until `block * 128 <= rows`.
The chooser consumes the shared mutable RNG, including a draw when modulo one
returns zero. Grouped histories shuffle blocks of groups and preserve members.
Native tests compare these orders directly with CUDA's original host helpers.

Generated feature banks currently retain all registered columns with an
explicit 1 GiB bound. Dynamic GPU bank compaction, CUDA's cache eviction policy,
and device-pack candidate visitation order remain separate performance work.

## Verification

Run the primitive checks on Apple Silicon with NumPy, pytest, and CatBoost
available in the selected Python environment:

```sh
PYTHONPATH=catboost/metal/python python -m pytest -q \
  catboost/metal/tests/test_sort.py \
  catboost/metal/tests/test_ctrs.py \
  catboost/metal/tests/test_projection.py
```

The tests cover unsigned extremes, signed category extension, deliberate
low-32-bit collisions, arbitrary history orders, duplicate keys, multi-level
GPU scans, and integer-reference hashes. Compound GPU CTR tables are loaded
into standard CatBoost and evaluated on original known/unseen categorical
rows. CPU training is forbidden in these compatibility checks.

Native Pool, multiclass, model round-trip, and snapshot acceptance is in
[`test_native_categorical_pool.py`](../tests/test_native_categorical_pool.py)
and requires the rebuilt extension with `CATBOOST_NATIVE_METAL_TESTS=1`.

The native `model_ut` target includes `TMetalTreeCtrTensors` (eight rules tests),
`TMetalFeatureParallelPermutations` (five source-comparison tests), and
`TMetalTreeCtrFeatures` (eight GPU helper tests). Run these named suites on an
Apple Silicon native build; none fits a CPU training model.
