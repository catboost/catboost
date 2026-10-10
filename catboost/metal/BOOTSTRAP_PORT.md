# Object bootstrap on Metal

The native kernels in `native/metal_bootstrap_kernels.h` translate CUDA's
structure-search sampling. Bootstrap multipliers change the weighted gradient
and split-scoring denominator. Final leaf estimation uses the original
observation weights and every object, following
`cuda/gpu_data/bootstrap.h::BootstrapAndFilter` and the separate CUDA leaf
estimator. Leaving zero-weight objects in their partitions is equivalent for
these pointwise sums; compacting them is a later performance improvement.

| Type | Multiplier | Source |
| --- | --- | --- |
| No | 1 | `cuda/gpu_data/bootstrap.h` |
| Bayesian | `(-log(U + 1e-20)) ** bagging_temperature` | `cuda/cuda_util/kernel/bootstrap.cu::BayesianBootstrapImpl` |
| Bernoulli | `U < subsample` | `bootstrap.cu::UniformBootstrapImpl` |
| Poisson | Poisson draw with `lambda = -log(1 - subsample)` | `bootstrap.cu::PoissonBootstrapImpl`, `random_gen.cuh::NextPoisson` |
| MVS | `1 / probability` when sampled, otherwise 0 | `cuda/cuda_util/kernel/mvs.cu::MvsBootstrapRadixSortImpl` |

For MVS, `magnitude = sqrt(weighted_gradient ** 2 + lambda)` and
`probability = min(1, magnitude / threshold)`. Each tile contains at most 8192
objects, matching the CUDA tile boundary. Metal solves the expected sample
count equation using 32 monotone bisection steps instead of CUB radix sort and
prefix sums. This solves the mathematical sampling equation; it deliberately
does not reproduce CUDA's truncated integer denominator or discarded radix
bits. Different devices are not expected to select identical objects.

Explicit `mvs_reg` supplies lambda. Without it, CUDA uses the square of the
mean absolute weighted gradient for the first tree, then the square of the
mean absolute **shrunk** leaf values of the preceding tree. See
`cuda/models/oblivious_model.h::GetL1LeavesSum` and
`cuda/models/additive_model.h::GetL1LeavesSum`. `ReduceBootstrapStatistics`
computes bounded first-tree partial means on Metal; the small final sum is
performed in host double. Snapshot/resume must restore the previous lambda
alongside the absolute iteration index.

## RNG contract

`BootstrapNextUint` preserves the unsigned integer multiply-with-carry step
from `cuda/cuda_util/kernel/random_gen.cuh::AdvanceSeed`. CUDA maintains mutable
seeds indexed by launch threads. Metal expands `(random_seed, absolute
iteration, stream, object)` into an independent state with domain-separated
SplitMix64, then uses the CUDA multiply-with-carry generator. This makes the
Metal sequence independent of launch size and allows resumed training to
reuse the uninterrupted sequence. It does **not** reproduce CUDA's seed
initialization or mutable launch-thread assignment.

Metal rounds uniform variates to float32 and caps them below 1. Logarithmic
Poisson/normal transforms also exclude 0. CUDA's float/double conversion and
transcendental implementations differ. Exact cross-device random variates or
tree structures are not promised.

`BootstrapNormalForItem` provides the Box–Muller normal transform for split
noise. CUDA Cosine shares one draw per feature across all its candidate
thresholds. The CUDA L2 score calculator does not apply that noise. The helper
does not itself implement the score-scale schedule from
`cuda/methods/random_score_helper.h`.

## Bounds and kernel dispatch

Public validation must require finite nonnegative temperature and MVS
regularization, Bernoulli/MVS subsample in `(0, 1]`, and Poisson subsample in
`(0, 1)` **after float32 conversion**. CUDA passes a lambda of -1 at Poisson
subsample 1, producing all-zero weights; Metal rejects that unusable boundary.
Bayesian temperature 0 and Bernoulli/MVS subsample 1 are exact no-ops.

`BootstrapParams` is 48 bytes: eight uint32 fields followed by four floats.
Fields are `rows, type, seed_low, seed_high, iteration, stream, reserved0,
reserved1, temperature, subsample, mvs_lambda, noise_scale`. Type values are
No 0, Bayesian 1, Bernoulli 2, Poisson 3, MVS 4.

| Kernel | Buffers before parameters | Dispatch |
| --- | --- | --- |
| GenerateBootstrapWeights | output multipliers, derivatives | One thread per row |
| ApplyBootstrapWeights | in/out gradients, original weights, multipliers, output structure weights | One thread per row |
| ReduceBootstrapStatistics | derivatives, output float2 partials | Full groups of 256; one partial per group |
| ComputeMvsThresholds | derivatives, output thresholds | `ceil(rows / 8192)` groups of 256 |
| GenerateMvsBootstrapWeights | output multipliers, derivatives, thresholds | One thread per row |
| GenerateBootstrapNormals | output normals | One thread per item |

The first statistic component is partial `mean(abs(g))`; the second is partial
`mean(g*g)`, which is not the weighted split-noise variance formula.

## Verification

`tests/test_bootstrap.py` runs the kernels on the actual Apple GPU through a
small isolated native probe. Nineteen tests cover exact integer RNG fixtures,
seed/iteration/launch-length behavior, Bayesian and Poisson scalar formulas,
zero/one boundaries, weight application, distribution checks over 131071
objects, MVS thresholds against an independent sorted-prefix solution, and
automatic-lambda partial reductions over 1000031 weighted derivatives.

`tests/test_bootstrap_training.py` adds 33 checks through the public Metal
estimator: all samplers train with original-weight final leaf estimates,
repeatable seeds, identity boundaries, and seven-tree snapshot continuation
from three saved trees. Resumed training matches uninterrupted model
structure, leaves, learn predictions, and validation histories, including
automatic MVS state. Invalid option and changed-seed snapshot checks fail
before GPU training. All 52 bootstrap checks passed on the Apple M3 Pro.
No CPU CatBoost training is used.
