# Standalone categorical ranking

CatBoostMetalRanker now accepts one-hot categorical arrays and DataFrames for
all seven supported query objectives: QueryRMSE, QuerySoftMax, PairLogit,
PairLogitPairwise, QueryCrossEntropy, YetiRank and YetiRankPairwise. Existing
objective-specific leaf methods, scores and sampling restrictions still apply.

Feature preparation checks unique original CatBoost hashes across learn and
validation against one_hot_max_size before creating any CTR or training
session. This follows CUDA's OnAll threshold rule. One-hot dictionaries contain
learn categories; unknown evaluation/prediction hashes use the existing unseen
bin and never match a learned equality split. Text/integer hashing preserves
CatBoost's integer/string equivalence and Unicode values. Named DataFrames
require consistent feature names and order at validation and prediction.

Training uses existing GPU equality search and leaf kernels. Metadata exports
actual categorical settings, and CBM/JSON retain original OneHotFeature splits.
Categorical-only models, model application and best-model trimming use the
existing feature layout and GPU reader. No CatBoost CPU fitting occurs.

Snapshots now include the learned hash dictionary and hashes of original
validation categorical columns, in addition to the existing bin/query/weight
fingerprints. Renaming learn categories can retain identical dense bins while
changing model meaning; renaming two unseen validation categories can retain
identical unseen bins. Both changes are rejected on resume. Numeric snapshots
keep their prior metadata schema, with no new empty categorical fields.

79 new GPU cases cover all supported leaf methods for the seven losses,
No/Bernoulli sampling, exact recovery, array/DataFrame equivalence, mixed
integer/string categories, learn/eval threshold boundaries, dictionary and
unknown-category snapshot identity, original pair weight semantics, unlabeled
pairs, shared PFound selection, trimmed models and CBM/JSON readers. An
independent QueryRMSE case proves a category in the middle of hash order is
selected by equality and yields the expected query-centered predictions.

Raw categorical or prequantized Pool training uses the native
CatBoostRanker(task_type="GPU") entry point documented in RANKING_ONE_HOT_PORT.md.
The standalone adapter still cannot extract raw categorical values from Pool.
CTR/P4 ranking, group-aware history controller integration and other CUDA
parity work remain pending. This is not an NVIDIA equivalence measurement.

Checkpoint `20260913T100621Z` preserves 6197 tests +16 subtests, 991 alternate
acceptance cases and 60 installed-runtime/reader paths. The native extension
and CLI are unchanged from `20260913T095519Z`; its 17 passing CLI variants and exact
wheel are retained. Wheel SHA256 `f592c177c1950414bb315f9c32d96ab1f60b709ded92caaefb391fc7868c9454`. The standalone source adapter uses
`PYTHONPATH=catboost/metal/python` alongside that installed native package.
