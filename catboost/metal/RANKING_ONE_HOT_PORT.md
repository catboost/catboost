# Native one-hot categorical ranking

The native Metal path now accepts one-hot categorical features for classic
YetiRank, YetiRankPairwise, PairLogitPairwise and QueryCrossEntropy. Numeric and
categorical candidates share the existing GPU search, leaf-estimation and
prediction kernels; there is no CPU training path.

The source rule is CUDA's `TBinarizedFeaturesManager::UseForOneHotEncoding` in
`catboost/cuda/data/binarizations_manager.cpp`. The threshold counts unique
hashes on **learn plus validation**, including eval-only categories. Native
preparation preserves the original CatBoost hashes in standard OneHotFeature
model splits and uses dense byte equality bins for training. Cardinalities up
to 255 are supported. Ignored/constant columns retain the shared Pool layout.

The previous native checks rejected every categorical column before feature
preparation. They now reject a feature only when CUDA's rule selects a CTR for
one of these four objectives. That check runs before history computation.
One-hot-only input uses one dataset and needs no target history, so supplied
PairLogitPairwise pairs work without relevance labels. Target-presence validation and permutation order allocation occur only when
actually building a CTR. The existing multi-target categorical restriction
remains explicit.

This does not enable ranking CTR histories or their multiple-permutation
controller. Group-aware CTR/P4 integration, the complete CUDA mutable random
buffer protocol, grouped Ordered boosting and other documented parity work
remain pending. One-hot support does not alter the existing objective,
bootstrap, leaf-method or depth limits. The standalone ranker now accepts one-hot arrays/DataFrames too; see
STANDALONE_RANKING_ONE_HOT.md. Its categorical Pool adapter remains native-only.

66 new actual-GPU cases cover all supported leaf methods, No/Bernoulli
sampling, raw/prequantized and categorical-only Pools, mixed feature indices,
Unicode and unseen categories, 255/256-category limits, constant/ignored
columns, unlabeled explicit pairs, initial models and baselines, exact
snapshots and changed-input rejection, and CBM/JSON/GPU readers. Fixed-bin
native forests match the independent host reconstruction of the category bank
fed into the resident Metal trainer. Existing resident-kernel tests separately
compare derivatives, candidates and leaf solvers with CUDA-derived equations.

Checkpoint `20260913T095519Z` is installed: **6118 tests +16 subtests**,
**912 alternate native acceptance cases**, **45 installed GPU paths** and
**17 native CLI variants** (eight prior numeric and nine categorical ranking).
Wheel SHA256 `f592c177c1950414bb315f9c32d96ab1f60b709ded92caaefb391fc7868c9454`. Both extensions, CLI, all changed
sources and test logs are preserved; the previous checkpoint `20260913T093443Z`
remains intact. This is not a claim of complete CUDA parity or an NVIDIA
comparison. No CPU CatBoost fitting occurred.
