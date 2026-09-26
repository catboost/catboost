# Native group-aware CTR histories

Native categorical preparation now passes original-row query ordinals to the
existing Metal group-prefix primitive when ctr_history_unit=Group and the
learn grouping is nontrivial. This follows CUDA BuildCtrTarget in
`cuda/gpu_data/dataset_helpers.cpp` and THistoryBasedCtrCalcer in
`cuda/ctrs/ctr_calcers.h`. External 64-bit group hashes are never truncated;
query ordinals are unique within the row-bounded training provider.

The CUDA-derived Metal target-history kernels read the category cursor before
the entire current group. No row can use a target from its own query in its
training CTR. Group-preserving history permutations keep each query together;
category sorting and group-prefix scanning run on GPU. Final inference tables
use all learn targets as before. FeatureFreq is independent of history unit.
With trivial grouping or Sample, the same native entry point has its previous
sample-exclusive behavior. CTR statistics use unit row counts/targets,
independently of the weights used to train trees.

The existing native Plain permutation controller now supports group-aware
single-feature CTRs for RMSE, QueryRMSE, QuerySoftMax and PairLogit, including
P1/P4. QueryRMSE/QuerySoftMax's CUDA default history unit is Group. Four native
CTR types remain available: Borders, Buckets, FloatTargetMeanValue, FeatureFreq.
Native YetiRank and full-matrix ranking now also accept CTR P1 banks; see
RANKING_CTR_P1_PORT.md. Their multiple-cursor/oracle controllers remain open. Grouped
Ordered and dynamic tree-dependent CTR projections are still gated.

98 new GPU cases pass: 24 full forests use independently calculated Sample or
Group histories, a fixed Uniform CTR grid and the resident trainer; 64
raw/prequantized P1/P4 native lifecycles cover all four tested losses and CTR
types, exact recovery, changed-query rejection, validation and CBM/JSON/GPU
readers; two cases verify default history selection; eight verify trivial
Group equals Sample exactly. The existing 60 group-prefix primitive tests
separately check whole-query exclusion and extreme floating-point histories.
No CatBoost CPU fitting or NVIDIA execution is used in this validation.

Checkpoint `20260913T101052Z` is installed: **6295 tests +16 subtests**,
**1089 alternate acceptance cases**, **76 installed GPU paths** and **21 CLI
variants**. Wheel SHA256 `db6581082aaad14cfc9027ad57664bc577e675d7a9a4e7e23b7fd0e81ba69c76`. Changed sources, both extensions, native
CLI, scripts and logs are preserved. Prior source checkpoint `20260913T100621Z` and
its installed native wheel checkpoint 20260913T095519Z remain recoverable.
