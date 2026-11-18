# CUDA score semantics checked for the Metal port

This audit describes the checked-in CUDA sources. Numerical references and
Metal GPU tests do not call a CPU CatBoost trainer. Checked-in CUDA canonical
histories are comparison fixtures; agreement with them is not a live NVIDIA
validation run.

## Plain SolarL2 and LOOL2

`cuda/methods/kernel/score_calcers.cuh` defines the child contributions below,
where `G` is the weighted negative-gradient sum and `W` is summed observation
weight. `oblivious_tree_doc_parallel_structure_searcher.cpp::ComputeWeakTarget`
uses first-order weights for these scores, even if leaf estimation uses Newton.

- SolarL2: `-G² * (1 + 2*log(W + 1)) / W` when
  `W > float32(1e-20)`, otherwise zero.
- LOOL2: `a = float32(W / (W - 1))` when `W > 1`, otherwise zero;
  `a = float32(a*a)`. The contribution is `-G²*a/W` for positive `W`,
  otherwise zero. Fractional weights, rather than document counts, control
  the boundary at one.

Both constructors discard their L2 argument. Neither calculator receives
normalization or score-noise parameters. L2 regularization still affects the
independent leaf estimator. Bootstrap multiplies both `G` and `W` before
structure scoring.

CUDA passes each child statistic to `AddLeaf` as double, but its running score
is float32. Parent leaves are visited in ascending order, adding the left then
right child; rounding happens after every child contribution. Metal has no
native double arithmetic. Its scaled Solar and LOO expressions avoid losing
representable scores through an intermediate float32 `G*G` underflow or
overflow. Tests include weights near `1e-20`, gradients near `1e-25`, and
weights of `1e30`.

The reference is `tests/cuda_auxiliary_score_reference.py`. Dense GPU checks
are in `tests/test_auxiliary_scores.py`; the independent compact oracle in
`tests/test_streaming_histograms.py` also covers both tiled scorers.

`private/libs/options/catboost_options.cpp` calls
`IsPlainOnlyModeScoreFunction` and rejects Ordered boosting for both scores.
A separate legacy dynamic Solar kernel exists in
`cuda/methods/kernel/pointwise_scores.cu::FindOptimalSplitSolarImpl`; its
learn/test-fold equation is not the Plain equation above. SatL2 also has a
kernel, but the current `IsSecondOrderScoreFunction` gate rejects that score.

## Multiclass symmetric greedy CTR penalty

Multiclass reaches
`cuda/methods/greedy_subsets_searcher/kernel/compute_scores.cu::ComputeOptimalSplits`.
It visits each parent, then each stored class, adding left and right children.
MultiClass also reconstructs the omitted class from the negative sum of the
stored class statistics. CUDA keeps partition sums and Cosine accumulators in
double; left histograms and explicitly cast right child statistics are float.

`greedy_search_helper.cpp::ComputeOptimalSplits` refreshes the CTR multipliers
before every symmetric depth through `UpdateFeatureWeightsForBestSplits`.
For every currently unused CTR feature, let `U` be its maximum unique count,
and let `Umax = max(1, U of every unused CTR feature)`. Its multiplier is
`float32(pow(1 + float32(U)/Umax, -model_size_reg))`. Already-used CTRs,
numeric features, and one-hot features have multiplier one.

`binarizations_manager.h::GetMaxCtrUniqueValues` defines `U` as the smaller of
`MaxObjectsCount` and `2^(tensor split count)` times the product of the source
categorical features' `OnAll` unique counts. Distinct CTR configurations are
distinct feature IDs even when they share a categorical source.

The greedy kernel computes `raw_score = float32(calcer.GetScore())`, obtains
`before_score` from an **empty** copy of that calculator, and compares
`gain = float32((raw_score - before_score) * multiplier)`. This does not
subtract the previous depth's score. For Cosine, identical feature noise is
present in both scores and cancels except for float32 rounding. The raw score
must remain separate from the gain used by the winner reduction.

The selected CTR feature ID is marked used immediately after selection, before
`SplitLeaves` applies its raw-score stopping condition. That set belongs to
the persistent features manager, so it survives across trees. The maximum
unused count can decrease as CTRs become used.

### Intentional parameter-forwarding correction

Upstream `greedy_subsets_searcher.h::MakeStructureSearcherOptions` copies L2,
score function, bootstrap, and other options, but does not copy `ModelSizeReg`.
Consequently `TTreeStructureSearcherOptions` retains its hard-coded default
`0.5`, including when a caller requests a different value. Metal's contract is
to honor the explicit `model_size_reg` value. This is an intentional plumbing
correction; the default `0.5` used by the canonical fixture is unchanged.

## Candidate ordering and canonical comparison limits

CUDA `compute_by_blocks_helper.cpp` groups features by binary, half-byte, and
one-byte 5/6/7/8-bit histogram specialization, then appends each feature's bins
in ascending order. Candidate order therefore need not follow global feature
ID order. Within a score block, equal float32 gains choose the lower flat
candidate index. Across returned blocks/devices,
`TBestSplitProperties::operator<` compares `(Gain, FeatureId, BinId)`.

The Cloudness canonical fixture has 101 rows, 102 numeric columns and 42
categorical columns. Its numeric columns alone yield 4,745 candidates with
the fixture's 128-border budget. It is not a numeric-only comparison and it
necessarily exercises multiple winner groups. The missing CTR penalty and
candidate gain transport must be resolved before attributing its history
differences to precision or tie ordering. The fixture contains error histories,
not the originating CUDA model's split sequence.

## Histogram and scalar calcer precision

`tests/scalar_histogram_probe.mm` dispatches the production root histogram,
prefix scan, smaller-child rebuild, sibling subtraction, partition reduction,
and split-scoring kernels. Its Python driver retains a compiled library so
before/after comparisons do not accidentally use newly edited shader strings.
The tests never invoke a CatBoost training backend on the CPU.

The Adult RMSE fixture exposed two separate losses of precision. Threadgroup
float additions, bank merges, and inclusive prefixes discarded rounding
residuals. In addition, the scalar scorer used float means and accumulators
where CUDA uses double. For a concrete two-parent fixture, the old Metal
Cosine scores were `-3.223452568054199` and `-3.2234530448913574`; the CUDA
arithmetic rounds both to `-3.2234528064727783`. The difference selected the
later candidate and changed the tree. The permanent regression supplies those
statistics directly to the actual GPU scorer, independently of histograms.

Dense, compact, and reused histogram kernels now retain local addition
residuals and use float-pair bank merges and inclusive prefixes. The local
histogram footprint grows from 8 to 16 KiB, and prefix scratch from 2 to
4 KiB; external buffers and bindings are unchanged. Scalar and streamed
scorers share float-pair quotient, product, accumulation, and square-root
helpers. L2 still rounds its running score after each child, as CUDA does;
therefore it can legitimately distinguish reversed child order by one ULP.
Cosine retains its wider intermediate sums until the final float score.

Both scalar scorers also preserve CUDA's selected-histogram-first arithmetic
order for one-hot features. `ScanHistogramsImpl` leaves equality bins raw,
and `FindOptimalSplitSingleFoldImpl` calls `AddLeaf(selected)` before
`AddLeaf(parent-selected)` for every feature type. Prediction still routes an
equality match to its right-child bit. Reversing the arithmetic terms to match
that routing reversed two L2 candidate scores by one ULP in the direct GPU
fixture; a separate regression now covers both feature types.

The direct GPU suite checks cancellation such as `[1e8, 1, -1e8]`, different
bank placements, prefix versus equality bins, empty partitions, and all four
L2/Cosine score identifiers. These changes do not emulate the full exponent
range of IEEE double. Device histogram storage, inter-tile atomics, sibling
subtraction, and published parent totals remain float32, so reduction-order
differences are still possible.

With independently reconstructed Adult bins and CTR histories, the combined
correction reproduces all 20 checked-in CUDA RMSE L2 and Cosine learn/test
values within `2.18e-8`, and preserves Logloss Cosine agreement within `1e-7`.
This comparison uses historical CUDA fixtures rather than a live NVIDIA run.
Logloss L2 still has a validation-only difference: a sixth split can leave all
learn leaf values unchanged while assigning validation rows to empty learn
children. Appending `fnlwgt > 38811` reproduces the historical third-tree metric
within `1.5e-9` without dividing any occupied learn leaf; the originating CUDA
split sequence is unavailable. This
remaining structure-selection discrepancy must not be hidden by widening the
comparison tolerance.

For the third Logloss L2 tree, supplying independently summed statistics to
the production GPU scorer gives exactly eight tied candidates at
`-5.310746669769287`: flat indices 39, 40, 144, 318, 327, 328, 484, and 485.
All preserve each occupied training leaf in one child. The stable comparator
selects 39 with these statistics; cached histogram rounding selects a different
equivalent training partition. A 32-leaf regression checks that empty-child
orientation itself does not alter either calcer. This establishes the local
tie equivalence without claiming that all remaining history differences, or
the separate multiclass L2 discrepancy, have been explained.
