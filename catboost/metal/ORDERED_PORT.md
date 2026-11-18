# Numeric Ordered boosting foundation

Status, 2026-09-12: numeric Ordered training now runs end-to-end on Metal through
the persistent `_ordered.Session` and the standalone frontend/lifecycle path.
The standard native CatBoost adapter is being integrated separately. All twelve
scalar objectives, Newton/Gradient leaf steps, Cosine/NewtonCosine scores,
weights, multiple permutations, all five bootstrap modes, score noise and full
fold-state resume are implemented. AnyImprovement and Armijo now use CUDA's
common step across all prefix and full-estimation tasks.
Production search now uses packed row partitions, tiled feature histograms and
smaller-child reuse with compensated sums. Histogram allocation is capped;
deeper partitions use a bounded candidate reduction over their own rows.
No CPU CatBoost training is used by these checks.

Files:

- `native/metal_ordered_kernels.h`: numeric fold builder and seven MSL kernels.
- `native/metal_ordered_trainer.{h,mm}` and `metal_ordered_session_kernels.h`:
  persistent native sessions, shared scalar objective/Exact/sampling kernels,
  bounded candidate batches and GPU tree search/leaf estimation.
- `native/metal_ordered_backtracking.h`: GPU directions and task objectives;
  one host acceptance decision over their compensated reductions.
- `native/metal_ordered_histogram_{kernels,runtime}.h`: packed fold partitions,
  compact numeric overflow buckets, paired histogram sums, sibling reuse and
  bounded feature/cache fallback. The probe uses the same host pipeline.
- `python/catboost_metal/_ordered.py`: scalar-compatible Session/TrainResult,
  deterministic permutation selection and full state export/restore.
- `tests/ordered_probe.mm`: diagnostic host dispatch, validation and result copies.
- `tests/test_ordered_kernels.py`: 36 tests, including independent numerical
  equations, leak checks, bootstrap isolation, separate estimation cursors,
  normalization, empty leaves, permutations and boundary validation.
- `tests/cuda_ordered_reference.py`, `test_ordered_reference.py` and
  `test_ordered_training.py`: independent full-training equations and actual GPU
  comparisons, sampling/state/inference invariants, and explicitly labeled
  private Exact-extension checks. `test_ordered_backtracking.py` adds 89 cases,
  including fixtures where the common accepted step worsens one prefix and
  therefore differs from incorrectly accepting each task independently.
  The standalone lifecycle adds its own tests.
- `tests/test_ordered_rng.py` and `test_ordered_session_rng.py`: 67 independent
  host and 20 actual-GPU checks of persistent chooser draws, full MT-state
  resume, repeated/empty search attempts and 50,003-row block permutations.
- `tests/test_ordered_histograms.py` and `ordered_histogram_probe.mm`: 54 actual
  GPU checks of tiling, sibling subtraction, candidate batches, conservation,
  cancellation, sparse/empty partitions and deep-cache fallback.

Verified core envelope: 545 tests plus 16 subtests in one complete run. The
standalone controller also passes all 101 Ordered lifecycle tests against this
histogram runtime. Native adapter/package suites are tracked by their owner.

Run:

```sh
PYTHONPATH=catboost/metal/python /tmp/catbooster-metal-venv/bin/python -m pytest catboost/metal/tests/test_ordered_kernels.py -q
```

## Source map

All references are to this checkout's upstream CUDA source.

| Behavior | Source |
| --- | --- |
| Minimum estimate size, geometric folds | `cuda/methods/dynamic_boosting.h:176–232` |
| Numeric next-row rounding | `cuda/gpu_data/samples_grouping.h:52–54` |
| Permutation count and search selection | `cuda/methods/dynamic_boosting.h:250–253,285–288` |
| Independent fold targets/cursors | `cuda/methods/dynamic_boosting.h:315–334,574–619` |
| Prefix and full-model leaf estimation | `cuda/methods/dynamic_boosting.h:374–410` |
| Apply prefix model to its complete cursor | `cuda/methods/dynamic_boosting.h:411–473` |
| Packed estimate/quality occurrences | `cuda/methods/oblivious_tree_structure_searcher.cpp:307–361` |
| Bootstrap only quality tails by default | `cuda/methods/oblivious_tree_structure_searcher.cpp:59–73` |
| Gradient/Newton scoring targets, quality-only noise variance | `cuda/methods/oblivious_tree_structure_searcher.cpp:376–466` |
| Dynamic Cosine/NewtonCosine score | `cuda/methods/kernel/pointwise_scores.cu:318–418` |
| Legacy dynamic Solar score | `cuda/methods/kernel/pointwise_scores.cu:42–128` |
| Dynamic score dispatch | `cuda/methods/kernel/pointwise_scores.cu:443–467` |
| Public Ordered restriction for SolarL2/LOOL2 | `private/libs/options/catboost_options.cpp:980–986` |
| Per-task normalized leaf derivatives and regularization | `cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.cpp:140–165,253–261` |
| Permutation index, seed and shuffle | `cuda/data/permutation.h:79–95`, `cuda/data/permutation.cpp:7–17` |
| Snapshot state | `cuda/methods/dynamic_boosting_progress.h:12–58` |

## Persistent-session contract

1. **Immutable data and row maps.** Keep numeric bins, targets, original weights
   and baseline in original input order. For each permutation, retain a map
   `permutation[position] = original_row`. Candidate bin lookup and tree leaf
   IDs use original row IDs; cursor indexing uses permutation positions. The
   diagnostic supports arbitrary supplied permutations. The Python session
   translates CUDA's numeric row/block shuffle. Permutation 0 is identity. The CUDA
   permutation seed is `1664525 * permutation_id + 1013904223 + block_size`;
   the block size is forced to 1 for fewer than 50,000 rows. At or above that
   threshold, zero/unset becomes 64, then the requested block is rounded up to
   a power of two and halved until `block_size*128 <= row_count`. Shuffle
   whole blocks, preserving each block's internal row order and truncated tail.

2. **Fold boundaries.** On a single GPU the numeric pool must contain at least
   four rows. `MinEstimationSize(N)` is 1 below 500 rows. Otherwise let
   `k = ceil(log2(ceil(N/min_fold_size)))`; use `ceil(N/2^18)` if `k >= 18`,
   and `min(min_fold_size, N//50)` otherwise. `min_fold_size` defaults to 100.
   Crucially, numeric `NextQueryOffsetForLine(x)` is `min(x+1,N)`.
   Therefore the first estimate end is `min(MinEstimationSize(N)+1,N)`;
   each quality end is `min(floor(estimate_end * growth_rate)+1,N)`.
   The next estimate end equals the previous quality end. Growth defaults to
   2.0. For N=33 the pairs are `(2,5),(5,11),(11,23),(23,33)`. For N=500 they
   start `(11,23)`. These are exclusive endpoints, despite the next-row
   rounding. The host helper preserves float-config-to-double multiplication
   and caps diagnostic state at 4,096 folds, uint32 packed indices and 2^24 rows.
   Group-aware rounding and distributed folds are outside this numeric helper.

3. **Independent persistent cursors.** For every learning permutation and every
   fold retain a cursor of length `quality_end`, initialized from that
   permutation's baseline or starting constant. A packed descriptor is
   `uint4(estimate_end, quality_end, cursor_offset, 0)`; descriptors have no
   gaps. The same document can appear in several folds with different current
   predictions. Its gradient must be recomputed from each fold's own cursor.
   A single prefix scan of one global gradient vector is mathematically wrong.
   Also retain an independent full-length estimation cursor and validation
   cursor(s). A model's exported predictions cannot reconstruct fold cursors.

4. **Permutation selection.** CUDA uses `P-1` as the full-model estimation
   permutation and `P-1` learning permutations, with a one-permutation fallback.
   Ordered input data forces P=1; the configured default is P=4. The current
   CUDA search selector is literally `random % (learn_permutation_count-1)`
   when learning count exceeds 1, otherwise 0. This excludes the final learning
   permutation from search selection while every learning permutation is still
   updated. Preserve that policy for an exact upstream comparison, or document
   any deliberate correction. The Python session now follows numeric Ordered's
   persistent MT19937-64 host stream. CUDA's numeric Ordered
   stream consumes a chooser draw (only when P>2), then once 65,537 draws when
   creating its bootstrap MirrorMapping seed cache even for bootstrap No, then
   one draw per attempted search depth, including the final repeated split.
   Explicit C ABI search-permutation inputs let the native caller supply this
   policy. The Python session saves all 312 MT words, the current index and
   bootstrap-cache flag. This corrects the earlier DocParallel chooser reuse.
   GPU bootstrap/noise launch streams still use the separately documented Metal
   adaptation, so this does not claim complete CUDA random-stream identity.

5. **Search derivatives and sampling.** For the selected permutation, pack each
   fold's estimate rows `[0,estimate_end)` immediately followed by its quality
   rows `[estimate_end,quality_end)`. Each occurrence has its own weighted
   gradient G and structure denominator W. For Cosine, W is observation weight;
   for NewtonCosine it is weighted curvature. `PrepareOrderedRmseDerivatives`
   supplies RMSE's `G=w*(target-cursor), W=w`; other losses must use their
   existing objective kernels against each separate fold cursor. Generate
   bootstrap factors over **packed occurrences**, not unique source documents.
   The default `observations_to_bootstrap=TestOnly` replaces all prefix factors
   with 1. Multiply both G and W by the chosen factor. Preserve original G/W
   for noise calculation, and original observation weights for leaf estimation.

6. **Candidate statistics.** After selecting every new symmetric split, update
   original-row leaf IDs and aggregate independently for each candidate, current
   leaf, fold and side. The new kernel returns
   `float4(estimate_weight, estimate_gradient, quality_weight, quality_gradient)`
   in `[candidate,leaf,fold,side]` order. The diagnostic foundation directly
   sums both sides. Production instead partitions packed occurrences by
   `leaf*power_of_two_fold_slots+fold`, retaining original row and estimate/
   quality membership. Shared stable radix kernels append each new leaf bit.
   Each feature gets a compact histogram through its largest requested border
   plus one overflow bucket, which includes encoded bin255. Prefix sums give
   the left side; subtracting it from the terminal total gives the right side.
   All four channels retain paired float components through local collisions,
   tile merges, prefix sums and sibling subtraction. Candidate output remains
   float32, preserving the existing scoring interface.

   One-cache layouts compute the strictly smaller child (equal sizes choose
   right), then subtract from the cached parent without clamping signed
   channels. Larger feature layouts stream tiles and rebuild their histograms.
   Cache leaf capacity ensures one feature fits the 128 MiB target; beyond
   that capacity, active partition reductions process candidate batches using
   only their own rows. Thus fallback work is proportional to packed rows
   times candidates, with no extra full-fold scan for every leaf. Empty
   partitions have no histogram jobs. All buffers remain under the combined
   conservative 1 GiB workspace limit. Leaf estimation and scoring equations,
   bootstrap/noise draws, cursor updates and snapshot formats are unchanged.

7. **Cosine/NewtonCosine scoring.** For each leaf/fold/side, compute
   `mu=G_est/(W_est+lambda)` when W_est>0, else 0. Lambda is L2 by default;
   with fold-size normalization it is `L2*W_est`. Accumulate
   `a += G_quality*mu` and `b += W_quality*mu^2`, starting b at 1e-20.
   Score is `-a/sqrt(b)` if b>1e-15, else float32 maximum. Apply categorical
   feature multiplier, then the feature's precomputed noise. Gain is
   `(score-score_before_split)*feature_penalty_multiplier`. Minimize gain;
   break ties by candidate index. All thresholds of one feature share noise.
   Ordinary L2/NewtonL2 are absent from CUDA's dynamic dispatch. Legacy dynamic
   Solar arithmetic is tested here, but current CUDA public validation rejects
   Ordered+SolarL2/LOOL2, so the public Ordered port must initially reject them.

8. **Score noise.** Before bootstrap, only quality occurrences contribute
   `sum(W*(G/(W+1e-15))^2)`, with the CUDA tiny-gradient zero rule. Divide by
   the total number of quality rows, including zero-weight rows, then take the
   square root. Multiply by random_strength and
   `logistic(log(N)-absolute_iteration*learning_rate)`. Keep the scale fixed
   for the tree and draw fresh feature noise per depth. `OrderedQualityStatistics`
   returns `(sum,quality_row_count)` per fold; reduce this small result in host
   double. Prefix duplicates must never enter either count or squared sum.

9. **Leaf estimation and cursor update.** After choosing structure, estimate
   one separate leaf-value vector for every fold in every learning permutation,
   using that fold's **estimate prefix only**, its current cursor, and original
   unsampled weights. Independently estimate a full-model vector on all rows of
   the estimation permutation using its full estimation cursor. For a single
   RMSE step, this module computes `G_leaf/(W_leaf+L2)` or, with normalization,
   `G_leaf/(W_leaf+L2*total_task_weight)`. Note that normalization here uses
   total prefix weight, whereas split scoring uses each child's estimate weight.
   Other objectives and multiple leaf-estimation iterations must reuse their
   objective and solver sequence for each task. CUDA backtracking, however,
   chooses ONE common step using the SUMMED objective across every prefix and
   full-estimation task; independently accepting a step per fold is incorrect.
   The implemented AnyImprovement/Armijo walker uses that common decision,
   counts rejected trials against the leaf-iteration budget, and allows up to
   100 trials until its first acceptance. A one-iteration solve bypasses the
   acceptance loop, matching CUDA. Exact's private extension ignores the rule.
   Apply the
   fold vector times learning_rate to **both** prefix and quality rows in that
   fold's cursor, never to later rows. Apply only the full-model vector to the
   full estimation and validation cursors, and append only that vector to the
   exported CatBoost tree. The kernel accepts a synthetic descriptor
   `(N,N,offset,0)` to estimate and update the independent full-model task;
   it contributes no quality statistics and is not a search fold.

10. **Progress and restart.** Snapshot every fold cursor, estimation cursor,
    validation cursor, retained and trained model structures, all permutation
    maps/fingerprints, fold descriptors, objective/normalization options,
    absolute iteration and selection/bootstrap RNG state. Include adaptive MVS
    state if enabled. Resuming with only the exported model changes Ordered's
    training trajectory. Init-model predictions may seed all cursors consistently
    before Ordered training, but do not recreate an earlier Ordered session.

## Connected session details and remaining validation

The Ordered persistent session has its own fold search path, task-specific leaf
solves, full-model-only export and snapshot state. It accepts depth 0–16 within
a conservative 1 GiB peak workspace limit and batches candidate statistics to
bound their memory. FeatureParallel is the upstream Ordered training mode;
upstream validation rejects Ordered+DocParallel. The public frontend preserves
that parameter boundary. The native adapter must preserve it too.

`Session.state()` returns version 1, a dataset/options/permutation fingerprint,
`uint32[task_count,4]` descriptors, `float32[packed_rows]` cursors, absolute
`iteration_offset`, optional scalar `mvs_lambda` and `selection_rng`. The latter
is `{version:1, words:[312 uint64 integers], index:0..312,
bootstrap_initialized:bool, completed_iterations:absolute_iteration_offset}`.
The lifecycle stores MT words as a uint64 array, includes them and the metadata
in the continuation checksum, and validates the entire state before GPU work.
Pass this dictionary as
`initial_state` when continuing; only the newly trained trees are returned.
Starting a fresh session at nonzero `iteration_offset` uses a fresh host MT
stream while sampling/noise use that absolute offset; prior attempts cannot be
reconstructed from an offset alone. Exact continuation requires the full state.
The fingerprint contains a selector policy tag, so older incomplete snapshots
cannot silently resume with the corrected RNG. Numeric arrays are stored
separately from JSON metadata.

Ordered MVS intentionally preserves the feature-parallel CUDA call site:
`oblivious_tree_structure_searcher.cpp:61` passes `target.Weights` as MVS's
derivative input. Thus magnitudes use original observation weights for Cosine
or weighted curvature for NewtonCosine. The first automatic lambda is the
square of their mean over packed fold occurrences. Later lambda is the square
of the mean absolute shrunk leaf value of the **exported** last tree, following
`feature_parallel_pointwise_oblivious_tree.h:53–59` and
`models/additive_model.h:82–86`. TestOnly still overwrites prefix multipliers
with one after sampling. RNG launch-stream initialization uses the documented
Metal adaptation in `BOOTSTRAP_PORT.md`.

The private runtime also composes the shared GPU Exact solver for each Ordered
task. This is an extension, not CUDA public parity: options validation at
`private/libs/options/catboost_options.cpp:347–350` explicitly rejects Exact with
Ordered on GPU. Public/native frontends retain that rejection; Quantile, MAE
and MAPE use Gradient estimation in CUDA-compatible Ordered mode. Private
Exact tests cover full-float residual ordering and corrected original-target
MAPE weights, as the shared Exact solver does.

The histogram benchmark compares the old and new Metal runtimes using the
same seeded numeric data, weights, P4 selection sequence and training options.
Five fresh sessions per case are timed after warming shader initialization.
The measurements include session setup, training, result copies and state copy;
GPU time comes from Metal command buffers. Medians on this M3 Pro:

| Rows / features / bins | Depth / trees | Old wall | New wall | Old GPU | New GPU |
| --- | --- | --- | --- | --- | --- |
| 8,192 / 8 / 16 | 4 / 20 | 0.135 s | 0.129 s | 0.0336 s | 0.0191 s |
| 32,768 / 16 / 32 | 6 / 10 | 1.022 s | 0.202 s | 0.835 s | 0.0384 s |
| 65,536 / 24 / 32 | 8 / 5 | 3.417 s | 0.144 s | 3.309 s | 0.0605 s |

All saved splits, leaf values, weights, predictions and every fold cursor are
bit-identical across implementations in these workloads. Small workloads are
dominated by host setup; larger cases show 5.1x and 23.8x wall-time reductions.
These are internal synthetic Metal comparisons, not held-out quality results
or NVIDIA measurements. Reproduce with `tests/benchmark_ordered_histograms.py`;
the old cached library and five-run raw outputs are recorded in
`.build/ordered-histograms-before-final.json`, `ordered-histograms-after-final.json`
and `ordered-histograms-comparison-final.json`, with complete model/cursor NPZs.

One-hot Ordered is now connected; see [ORDERED_ONEHOT_PORT.md](ORDERED_ONEHOT_PORT.md).
Higher-cardinality features need each permutation's CTR view in addition to
these cursors. Numeric/one-hot Ordered does not require CTRs; Plain categorical CTR
training is also independent. Group-aware scalar folds are now connected; see [ORDERED_GROUPS_PORT.md](ORDERED_GROUPS_PORT.md). Query/ranking Ordered objectives,
categorical permutation feature views, complete CUDA
RNG reproduction and broader large-data performance validation still need work. The numerical
and lifecycle checks establish the supported envelope on this M3; they do not
establish NVIDIA numerical and performance parity.
