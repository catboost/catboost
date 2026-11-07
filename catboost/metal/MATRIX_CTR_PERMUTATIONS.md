# Full-matrix CTR permutation datasets

Native PairLogitPairwise and QueryCrossEntropy now accept multiple categorical
CTR datasets in Plain boosting, including the default P4 bank. The existing
CUDA structure-permutation chooser and common feature grid remain unchanged:
with P4, structure search selects dataset 0 or 1 and model export uses dataset 3.
Every dataset retains its own feature bins and prediction cursor. Group/Sample
histories, four simple CTR types, original-target weighting, QCE scales,
model-size penalties and normal model tables remain supported.

The CUDA behavior comes from `cuda/methods/doc_parallel_boosting.h` and
`pairwise_oblivious_trees/pairwise_oblivious_tree.h`. CUDA copies the searched
model into each dataset; Simple's NeedEstimation is false, so its values and
weak Hessian leaf weights are reused exactly. Other methods estimate the same
structure independently on each cursor. Metal now applies those two paths:

- Simple applies the existing scaled leaf-value buffer to each dataset's fixed
  structure. It preserves values and weights without division/re-rounding,
  additional weak-target sampling or a second leaf solve.
- Newton/Gradient reset each dataset's topology and raw leaf increments, then
  run the existing original-target solver/backtracking against its cursor.
  QueryCrossEntropy retains its Newton-only restriction for iterative leaves.

The shared allocator still accounts for every copied feature matrix and cursor
under its 1 GiB workspace limit; no extra Simple backup buffer is needed.
Saved permutation cursors and selection by absolute iteration preserve exact
recovery. Native callbacks, initial models, baselines and ordinary readers use
the existing lifecycle. One-hot-only inputs still collapse to one dataset.

55 new GPU runtime cases check distinct datasets against independently computed
fixed-structure directions, exact Simple model reuse, all selectable datasets,
P1/P2/P4/P7/P64 identical-bank equivalence, and exact staged cursor restoration.
99 native cases cover raw/prequantized P4 CTRs, all four CTR types, original
query/edge metadata, all supported leaf methods and backtracking modes, exact
snapshots, changed-input rejection, initial models, baselines, validation and
CBM/JSON/GPU readers. The expanded focused native run has 277 passing cases.

YetiRank and YetiRankPairwise still require P1 CTR banks while their stochastic
oracle schedules are connected for multiple datasets. The complete CUDA GPU
random-buffer protocol, grouped Ordered, dynamic tree-dependent CTRs and other
documented parity work remain open. Standalone public ranker arrays expose
one-hot categories; this P4 support is native and in the private resident API.
No CPU CatBoost fitting or NVIDIA execution was performed.

Checkpoint `20260913T103310Z` is installed: **6561 tests +16 subtests**,
**1300 alternate acceptance cases**, **90 installed GPU paths**, **30 CLI
variants**. Wheel SHA256 `7aea8c3f20db378a82868aaee58dc75b978a745db012aa82cb86df491cab0177`. Both extensions, CLI, changed sources,
scripts and logs are preserved. Prior P1 checkpoint `20260913T101959Z` remains intact.

Classic YetiRank CTR P4 is now installed; see YETIRANK_CTR_PERMUTATIONS.md.
Generated-pair YetiRankPairwise P4 needs its separate target schedule.

YetiRankPairwise now also accepts native CTR P4. Its fixed pair targets are
created separately for each original cursor; Simple reuses the searched
model. See YETIRANK_PAIRWISE_CTR_PERMUTATIONS.md for streams and evidence.
