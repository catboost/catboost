# FeatureParallel tree CTR scoring contract

The source below distinguishes a dynamically generated CTR feature from
Ordered boosting's dynamic **fold** score kernel. Plain FeatureParallel uses a
single fold for numeric, simple CTR and tree CTR candidates alike.

## Model-size denominators

At every depth, CUDA's `TTreeCtrDataSetsHelper::GetMaxUniqueValues` computes

```
dynamic_max = max(1, unique_bound(f) for active tree CTR f that is not used)
```

Unknown CTRs count as unused. `UpdateFeatureWeightsForBestSplits` receives this
maximum **by value** and separately increases its local maximum over globally
registered unused CTRs, including registered tree CTRs:

```
static_max = max(dynamic_max,
                 unique_bound(f) for registered CTR f that is not used)
```

The tree visitor receives the original `dynamic_max`, not `static_max`.
Static/simple CTRs have weight one after use; an unused static/simple CTR has
weight `(1 + unique_bound/static_max)^(-model_size_reg)`. Tree CTRs always have
weight `(1 + unique_bound/dynamic_max)^(-model_size_reg)`, even after use.

An append-only runtime therefore needs separate flags for:

- whether a column is a dynamic tree CTR;
- whether it is active at this depth;
- whether its exact configuration is globally registered;
- whether that exact configuration has been used.

Unused transient columns from prior trees do not belong to the global static
denominator. CUDA registers a tree-CTR winner, and eagerly registers grids for
predicate-free tensors whose category count is strictly less than
`MaxCtrComplexityForBordersCaching`. Identity includes the projection and full
`TCtrConfig`, including its binarization configuration. Exported `TModelCtr`
alone cannot distinguish two training configurations with different grids.

## Noise and gain ordering

`dynamic_boosting.h` uses `SetTarget` for Plain, producing `foldCount == 1`.
`pointwise_scores.cu::FindOptimalSplit` then dispatches both static and dynamic
CTR candidates to `FindOptimalSplitSingleFoldImpl`. The Cosine score calculator
adds feature noise before the categorical penalty:

```
score = float(raw_score + cosine_noise) * categorical_weight
gain = float(score - score_before_split) * user_feature_weight
```

SolarL2, SatL2, LOOL2 and L2 calculators do not add Cosine noise. L2's separate
meta-regularization draw is a different mechanism.

Ordered boosting's paired estimate/quality folds produce `foldCount > 1` and
dispatch to `FindOptimalSplitDynamic`. Its Cosine implementation applies the
categorical penalty before feature noise:

```
score = float(raw_score) * categorical_weight + cosine_noise
gain = float(score - score_before_split) * user_feature_weight
```

Its SolarL2 implementation adds no score noise. Kernel selection follows fold
count; it must not follow the dynamic-CTR feature flag.

FeatureParallel's host RNG stream also supplies separate seeds for the static
feature dataset, simple CTR dataset and tree visitor. The tree visitor further
mixes the base-tensor hash, device seed and compressed-dataset feature policy.
Reusing a DocParallel per-iteration seed or one global runtime feature ID for
these consumers is not a claim of exact CUDA random-strength parity.

## Source pointers

- `cuda/methods/tree_ctrs.cpp`: `GetMaxUniqueValues` visits current persistent
  and pure-tree packs, excluding used configurations.
- `cuda/methods/update_feature_weights.cpp`: grows the local static maximum
  across globally registered unused CTRs, then computes only static weights.
- `cuda/methods/tree_ctrs_dataset.h`: `GetCtrWeights` applies the dynamic factor
  to every candidate, including used configurations.
- `cuda/methods/tree_ctr_datasets_visitor.cpp`: winner and eager grid
  registration, base-tensor seed mixing, and strict gain comparison.
- `cuda/methods/oblivious_tree_structure_searcher.cpp`: passes the original
  dynamic maximum to the tree visitor and consumes per-dataset host seeds.
- `cuda/methods/kernel/pointwise_scores.cu`: single-fold versus paired-fold
  score dispatch and penalty/noise/gain ordering.
- `cuda/methods/kernel/score_calcers.cuh`: Plain score calculators and noise.
