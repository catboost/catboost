# Native ranking CTR feature banks: P1

Native YetiRank, YetiRankPairwise, PairLogitPairwise and QueryCrossEntropy now
accept the four existing single-feature CTR types: Borders, Buckets,
FloatTargetMeanValue and FeatureFreq. Group or Sample histories feed scalar
quantized CTR bins into the existing GPU split-search, full-matrix/point leaf
solvers, model-size penalties, cursor updates and standard CTR model tables.
This adds no CPU CatBoost training path.

This stage requires **permutation_count=1 or has_time=true**. CUDA's has_time
rule collapses the categorical dataset bank to P1 even when a larger count is
requested. With has_time=false, the shared preprocessing shuffle remains
active and preserves whole queries. A request for multiple CTR datasets fails
before creating histories. One-hot-only data still collapses to one dataset,
using learn plus validation category counts for the CUDA one-hot decision.

The category and group-exclusive history mathematics were already translated
and validated. The change removes the blanket CTR gate only for the supported
single-cursor configuration; every objective's existing leaf, sampling,
backtracking, score and depth restrictions remain in force. Original category
hashes, full-learn inference tables and CTR used-feature penalty state are
preserved in models and snapshots. Supplied-pair one-hot training can remain
unlabeled; target-dependent CTR training still requires targets.

112 new actual-GPU cases cover all supported leaf methods across the four
losses, all four CTR types, No/Bernoulli sampling, scaled QCE, raw/prequantized
Pools, independently calculated group histories and fixed-grid resident
forests, exact snapshots, explicit P1 preprocessing shuffle, Group/Sample
options, has_time collapse, P4 rejection, initial models/baselines and CBM/JSON
readers with unseen categories. Four separate comparisons demonstrate that
increasing model_size_reg changes the chosen feature from CTR to numeric.
178 focused cases pass including updated one-hot/P4 boundary checks.

Native group CTR P1/P4 for RMSE/QueryRMSE/QuerySoftMax/PairLogit is described in
GROUPED_CTR_PORT.md. Standalone ranker arrays/DataFrames still expose one-hot
categorical features only. Generated/full-matrix multiple-permutation cursor
and oracle orchestration, complete CUDA mutable GPU RNG consumption, grouped
Ordered and other documented parity work remain open. No NVIDIA execution or
full CUDA equivalence is claimed by this checkpoint.

Checkpoint `20260913T101959Z` is installed: **6407 tests +16 subtests**,
**1201 alternate acceptance cases**, **85 installed GPU paths**, **25 CLI
variants**. Wheel SHA256 `0864bad35600c7287be31f3f69089694d68c207492b6904389027b3ba9467e6c`. Both extensions, CLI, changed sources,
scripts and logs are preserved. Prior checkpoint `20260913T101052Z` remains intact.

PairLogitPairwise and QueryCrossEntropy now additionally support multiple CTR
datasets, including default P4; see MATRIX_CTR_PERMUTATIONS.md. YetiRank now also supports P4; YetiRankPairwise retains the P1 condition described above.

Classic YetiRank CTR P4 is now installed; see YETIRANK_CTR_PERMUTATIONS.md.
Generated-pair YetiRankPairwise P4 needs its separate target schedule.

YetiRankPairwise now also accepts native CTR P4. Its fixed pair targets are
created separately for each original cursor; Simple reuses the searched
model. See YETIRANK_PAIRWISE_CTR_PERMUTATIONS.md for streams and evidence.
