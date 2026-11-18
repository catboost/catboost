# Grouped scalar Ordered on Metal

The installed checkpoint supports numeric/one-hot scalar Ordered training on grouped
data. Native CatBoost accepts raw/prequantized Pools with group IDs and group
weights. Standalone scalar estimators accept `group_id`, `group_weight`,
`eval_group_id` and `eval_group_weight`; raw numeric Pools are also supported.
These paths preserve the existing scalar objective equations. Ordered query
objectives remain open. Simple CTR histories are connected in [ORDERED_CTR_PORT.md](ORDERED_CTR_PORT.md); upstream GPU group-unit bootstrap applies only to YetiRankPairwise.

## CUDA translation

- `cuda/data/permutation.cpp` shuffles whole groups while retaining each
  group's internal row order. `dynamic_boosting.h::GetPermutationBlockSize`
  chooses the block policy from document count, then applies it to groups.
- `dynamic_boosting.h::CreateFolds` rounds prefix/quality boundaries to the end
  of the containing group. A line at a group boundary belongs to the next
  group. Each permutation can therefore have different fold and cursor counts.
- The Metal session retains task offsets, fold counts and packed cursor counts
  per learning permutation. Histogram, sampling and score buffers reserve the
  maximum capacity, while each selected permutation binds its actual layout.
  Leaf estimation and publication still update every learning prefix and the
  separate full-estimation task.
- The private grouped fold helper accepts double growth, like CUDA's helper.
  Public CatBoost options are normalized through `TOption<float>` before that
  promotion, and the standalone `_number` helper does the same. Native/public
  numeric and grouped training therefore retain CUDA's float32 option policy.
  Tests accepting raw doubles just above one exercise the private ABI only;
  public frontends correctly reject values that round to one.
- A large final group can make an initial prefix cover every row. Its empty
  quality interval remains a valid full-prefix descriptor. Score noise follows
  CUDA `random_score_helper.h`'s `count + 1e-100` guard and returns a finite
  constant model when no quality rows can distinguish candidates.

An additive grouped C ABI accepts validated original group offsets and
a double growth argument after public option normalization. Every map must cover every complete group exactly
once. At least four groups are required, matching one-device CUDA validation.
Snapshot identity includes group layout; the standalone controller reconstructs
the final group permutation before checking its published cursor. Object,
group and class weights are combined in the existing target-weight order.

## Validation and limits

- 345 private checks cover independent group permutations/folds, scalar forest
  equations, every prefix cursor, mixed equality candidates, all five samplers,
  backtracking, P1/P4/P7, double-growth boundaries and malformed group maps.
- 67 native and 86 standalone checks cover weights, labels, fixed runtime
  comparisons, raw/quantized Pools, categories, CBM/JSON/GPU readers, validation,
  original native model/baseline continuation, best models and exact snapshots.
- 9479 combined tests plus 16 subtests; 2890 alternate acceptance cases.
- 433 installed GPU fit/prediction/recovery paths and 111 CLI variants pass.
- Forty-eight snapshots written by the preceding native/standalone sources
  recover exactly against their saved reference outputs: numeric/one-hot,
  RMSE/Logloss/Huber, P1/P4 and No/MVS. All leaves, weights, GPU predictions,
  histories and standalone prefix/selection cursors agree. Original snapshots
  are preserved separately from resumed files.
- Installed checkpoint `20260913T142818Z`, wheel SHA256 `c9b3000194469d31fb0093aeb5e35313072c7d5c6ba3e1ac03e8f3eecdb8cecb`.
  Both extensions, CLI, source hashes, original snapshots and raw evidence are
  archived; preceding checkpoint `20260913T135844Z` remains intact.

Group-unit sampling remains rejected; these checks use the supported object
sampling unit. Standalone raw Pool baselines are explicitly rejected; native
grouped baselines are tested. Native categorical Pools and standalone raw
category arrays/DataFrames retain the established frontend boundaries.

No CPU CatBoost fitting or NVIDIA/other M-series comparison was performed.
Complete CUDA numerical and performance parity remains unproven.

A subsequent precision-migration attempt was rejected after checking
`private/libs/options/boosting_options.h`: the existing float32 normalization
already matches CUDA. Production sources were restored to this checkpoint;
the attempted binaries were never installed. No numeric snapshot migration is
required for fold growth.
