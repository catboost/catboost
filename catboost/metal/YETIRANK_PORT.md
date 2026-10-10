# Classic YetiRank on Metal

The classic CUDA YetiRank target is connected to resident scalar tree search,
the standalone `CatBoostMetalRanker`, and native
`CatBoostRanker(task_type="GPU", loss_function="YetiRank")`.
Training uses Metal for derivatives, split search, leaf estimation and cursor
updates. Shared host code handles metrics, quantization and model serialization.

The connected scope is numeric Plain DocParallel, symmetric depth 0–16,
all seven scalar scores, five object samplers, and score noise. It uses CUDA's
one-iteration Newton default, zero default L2, and no backtracking. The classic
loss accepts `permutations` (default 10) and `decay` (default 0.85).
`mode=Classic` is accepted; other modes are rejected. Query sizes are limited
to 1023 rows. Public relevance labels must lie in [0,1] for PFound.

PFound is the default quality metric. NDCG/MAP/PFound can select the best
iteration and drive validation stopping. The stochastic derivative oracle's
zero function value is internal and is never reported as training quality.
Native fitting includes raw/prequantized Pools, group/object weights, baseline,
initial models, callbacks, snapshots, and normal CBM/JSON/GPU evaluation.
Standalone query baselines, categorical histories and subgroup inputs remain
outside its current frontend scope.

## CUDA mathematics

`native/metal_yeti_rank_kernels.h` translates
`cuda/targets/kernel/yeti_rank_pointwise.cu`. Whole queries are packed into
at most 1024 rows per GPU task. CUDA's per-task/per-lane 32-bit LCG seed and
four draws per lane are retained. Stable tuple bitonic sorting replaces
CUDA's two radix passes. Adjacent ranks contribute
`0.15 * decay^(rank-1) * abs(weighted_relevance_difference) / permutations`.
The outputs are ascent gradient and incident pair mass, not logistic curvature.
Both first-order and Newton split scores use this incident mass. Original
object-times-group weights remain separate for weighted relevance and model
leaf weights. Final Newton leaves are centered across all leaf coordinates,
including empty leaves, before applying the learning rate.

`native/metal_yeti_rank_runtime.h` owns persistent query/task/oracle buffers and
encodes into the scalar trainer's command buffers. Each target call gets an
explicit 64-bit seed. Invalid GPU intermediates set a checked status buffer.
The runtime's allocations participate in the trainer's 1 GiB workspace guard.

## Persistent host RNG

`train_lib/yeti_random.h` and `python/catboost_metal/_yeti_rng.py` preserve the
single-device numeric CUDA DocParallel host draw sequence:

1. The boosting constructor draws `BaseIterationSeed` once.
2. Each tree draws one weak-target seed, reused for noise/MVS statistics.
3. The first non-No bootstrap allocates its StripeMapping seed cache, consuming
   one base draw plus 65536 FillSeeds draws. No sampling skips that allocation.
4. Each attempted numeric split search consumes one draw, including a rejected
   duplicate winner. Zero-depth and empty-candidate trees consume no search draw.
5. Newton leaves consume one seed for a single iteration, or I+1 for I>1. The
   final unused derivative evaluation still advances the stream; its dead GPU
   dispatch is omitted.

Leaf seeds are supplied after structure search, preserving this order.
Standalone snapshots carry checked 312-word MT19937-64 state and index; native
snapshots carry a tagged draw count validated exactly against the saved tree
structures, then replay only host draws. They never replay completed training.
The existing symmetric native v6 snapshot prefix is unchanged. An initial
model starts a fresh host stream for its new fit, separately from the absolute
bootstrap/noise iteration offset. GPU bootstrap/noise streams remain the
separately documented Metal adaptation; this is not full NVIDIA RNG parity.

## Explicit source differences

The pinned `cuda/targets/kernel.h` passes `QuerySizes.Size()` to
`RemoveQueryMeans`, whose size parameter counts rows. This centers only the
first query-count rows. Metal defaults to centering all rows, restoring
invariance to a common shift within a query. Native models record
`metal_yeti_centering=all_rows`. The standalone ranker exposes
`yeti_legacy_prefix_centering=True` to reproduce the source extent, recorded
in its training statistics. This deliberate correction precludes a claim of
bit-identical CUDA training.

Extremely negative logits can underflow CUDA's exponential ranking keys and
produce 0/0 derivatives. When a task contains tiny exponentials, Metal evaluates
the same ranking perturbation and pair probability in log space. Normal-range
tasks retain multiplicative keys and arithmetic order. Floating-point ties
can still differ between devices.

## Verification and remaining work

Target kernels: 37 M3 cases. Persistent runtime: 37. Resident tree walk: 58.
Independent host RNG and recovery: 59. Standalone lifecycle: 24. Native API: 38,
including 14 comparisons with a separately driven fixed-quantization oracle.
Tests cover all samplers, both centering choices, variable search stops,
weighted relevance, corrupt snapshots, quality metrics and export. Check
IMPLEMENTATION_STATUS.md for the final aggregate and installed wheel revision.

Classic YetiRank categorical P4 is now connected; dynamic tensors and generated-pair P4 remain open.
YetiRankPairwise needs its own pair generator and full coupled tree search.
No CPU CatBoost training or live NVIDIA execution is used in these tests.

## Native one-hot categories

The native trainer now supports one-hot categorical features through ordinary
Pools and the CLI. CUDA counts learn plus validation categories; CTR history
is still gated for these objectives. See RANKING_ONE_HOT_PORT.md for validation.

Classic YetiRank CTR P4 is now installed; see YETIRANK_CTR_PERMUTATIONS.md.
Generated-pair YetiRankPairwise P4 needs its separate target schedule.
