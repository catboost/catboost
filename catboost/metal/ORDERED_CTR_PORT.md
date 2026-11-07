# Ordered simple CTRs on Metal

The installed checkpoint connects simple categorical CTRs to scalar Ordered training
through the native CatBoost API and standalone estimators. The preceding
installed checkpoint remains preserved.

## CUDA translation

- `cuda/methods/dynamic_boosting.h` builds a feature dataset for every
  permutation. The selected learning permutation supplies split statistics;
  every prefix task and the separate full-estimation task use their own
  dataset for leaf routing, estimation and cursor updates.
- The additive counted bank C ABI accepts one shared feature matrix or one
  matrix per permutation. Every bank shares candidate columns and borders.
  Histogram search binds the selected bank; all-bank routing supplies the
  correct leaf IDs to Newton/Gradient, private Exact, common backtracking and
  publication. Memory checks include all feature banks and leaf-ID buffers.
- Native CTR preprocessing uses the same FeatureParallel block/group history
  maps as the prefix session. Native Borders, Buckets, FloatTargetMeanValue
  and FeatureFreq keep the existing final CatBoost CTR tables and model hashes.
- Standalone Borders/FeatureFreq preparation accepts explicit Ordered history
  maps, including whole-group blocks above 50000 rows. `ctr_history_unit` is
  Sample by default; Group excludes the entire current group from target
  histories. Object/group/class weights continue through scalar loss training.
- `update_feature_weights.cpp` supplies the model-size multiplier. Crucially,
  FeatureParallel's searcher calls `AddUsedCtr` only for dynamic tree CTRs.
  With single-feature projections, penalties therefore persist after selection;
  importing DocParallel's used-feature exemption would change this algorithm.
  Simple Ordered snapshots need immutable penalty metadata, not mutable flags.
- The lifecycle keeps every prefix cursor and the persistent chooser. Snapshot
  identity includes all banks, category hashes, grouping, history policy and
  penalty configuration. Best-model trimming evaluates the final estimation
  bank; model prediction and validation use full final CTR tables.
- Shared feature banks retain prior numeric/one-hot private snapshot identity.
  Native/standalone existing grouped and ungrouped snapshots are checked against
  outputs saved by the preceding installed build before any installation.

## Validation

- 224 private tests: per-bank independent scalar equations, all prefix cursors,
  grouped layouts, backtracking, all samplers, P1–P64 recovery, static penalties,
  counted C ABI validation and malformed-bank/snapshot identity rejection.
- 156 native cases: all twelve scalar losses, four CTR types, both history
  units, all samplers, raw/quantized Pools, initial models, baselines, model
  readers and exact recovery. Twenty-four forest comparisons use independently
  calculated CTR histories with the resident Ordered session.
- 122 standalone cases: all twelve losses, both CTR types/history units,
  exact snapshots, class weights/labels, best-model extension, readers and
  independent group/block histories on 50200 rows.
- 9981 cases validated: 9974 passed the full run; seven test-only failures
  were repaired and pass in the 290-case recheck (including 16 subtests).
  No production changes were needed after that full run. 3168 alternate
  acceptance cases pass.
- 577 installed GPU paths and 135 CLI variants pass.
- Ninety-six preceding native/standalone snapshots recover exactly against
  saved old-build outputs: numeric/one-hot, grouped/ungrouped, three losses,
  P1/P4 and No/MVS. Leaves, weights, GPU predictions, metrics and all prefix/
  chooser state agree. Original snapshots are preserved separately.
- Installed `20260913T151643Z`, wheel SHA256 `54fd3d73943ed12d9c5e2d799b0ab177aee778d0582cc662e2d6590d9af8c81d`. Both extensions,
  CLI, source hashes, original snapshots and raw evidence are archived.

## Remaining scope

Dynamic compound/tree CTR scheduling, query Ordered objectives, complete CUDA
GPU RNG state consumption, estimated text/embedding features and remaining
options require further work. Ordered group-unit bootstrap is not a CUDA gap:
upstream GPU options allow it only for YetiRankPairwise, already connected.
Public fold growth retains upstream float32 normalization before double helper
arithmetic; no precision migration is required. Native Ordered Exact/vector
objectives retain their existing gates. The standalone CTR frontend supports
Borders/FeatureFreq; native training also supplies Buckets/FloatTargetMeanValue.

No CPU CatBoost fitting or NVIDIA/other M-series comparison was performed.
Full CUDA feature, numerical and performance parity remains unproven.
