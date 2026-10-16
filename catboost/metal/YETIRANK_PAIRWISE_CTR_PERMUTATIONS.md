# Generated-pair CTR permutations

Native classic YetiRankPairwise now accepts multiple categorical CTR datasets,
including default P4. Simple copies the exact searched model and weak Hessian
weights into every dataset, as CUDA's NeedEstimation=false path does.
Newton/Gradient construct an independent fixed pair target at each dataset's
original prediction cursor, then keep that target fixed throughout its leaf
iterations. The last dataset supplies exported leaf values and weights.

`cuda/methods/doc_parallel_boosting.h`, `cuda/targets/pfound_f.cpp` and
`cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.h` define this
flow. Fixed pair targets use CUDA's default Bayesian temperature one,
regardless of the weak target's bootstrap settings. No additional weak
target is sampled while replaying the structure on the other datasets.

## Explicit Metal random streams

The existing P1 protocol is preserved exactly. Multiple datasets record
`item_iteration_dataset_domains_v2`. The domain depends on absolute tree
iteration, original dataset index and weak/fixed target purpose, so execution
order does not alter a dataset's target and snapshots need no mutable random
buffer. Query masks and pair weights occupy separate stream slots.

For dataset p, sampling uses `(fixed ? 0x50464c00 : 0x50465700) + 4*p`, query
masks add one, and Bayesian pair weights add two. The inner permutation seed
adds `mix((uint64(p)<<32)|0x50464453)` by XOR before the existing outer mix
when p is nonzero; p=0 adds zero. P1 model/snapshot behavior is unchanged.

This is an explicit extension of the current Metal protocol. CUDA's mutable
GPU seed-buffer consumption remains unported; these streams do not promise
identical NVIDIA random draws. Pair mathematics, fixed-target lifetime,
dataset selection and model aggregation follow the inspected CUDA code.

## Validation

- 55 new runtime cases plus 111 existing generated-pair forest cases:
  independent weak structures and fixed matrices, all three leaf methods,
  No/Bayesian/Bernoulli Object/Group sampling, every selected dataset, P1/P2/
  P4/P7/P64, empty structures/query samples and exact cursor/metric recovery.
- 156 new native cases: four CTR types, Sample/Group histories, raw/quantized
  Pools, snapshots and incompatible-data/geometry rejection, initial models,
  baselines, standard readers and recorded stream version. These include 48
  independent CTR-history/resident-forest comparisons using upstream Pool
  shuffle goldens.

All training in these checks runs on Metal. No CPU CatBoost fitting or NVIDIA
execution is used. Remaining work includes full CUDA GPU random state,
dynamic categorical tensors, grouped Ordered, other feature pipelines and
options, memory scaling, and measured CUDA/M-series quality and speed parity.

Checkpoint `20260913T110339Z` is installed: **6991 tests +16 subtests**,
**1578 alternate acceptance cases**, **112 installed GPU paths**, **47 CLI
variants**. Wheel SHA256 `7afe8cd0eae71f4635e026b8660ffb540b17b74d4f92e8f9e6feaba9d8b1d0b2`. Sources, both extensions, CLI, scripts
and logs are preserved; prior checkpoint `20260913T105212Z` remains intact.
