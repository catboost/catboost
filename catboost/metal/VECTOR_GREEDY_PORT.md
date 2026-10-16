# Vector greedy CUDA port

Native Metal Depthwise, Lossguide and Region are connected for the three vector
objectives CUDA registers: MultiClass, MultiClassOneVsAll and
RMSEWithUncertainty. The private quantized `_multiclass.Session` accepts these
policies as well. Standalone estimators now connect numeric, one-hot and
Borders/FeatureFreq CTR inputs to this runtime and its training lifecycle.

The CUDA references are `cuda/train_lib/multiclass_non_symmetric.cpp`,
`multiclass_region.cpp`, `methods/greedy_subsets_searcher/kernel/compute_scores.cu`
and `methods/kernel/score_calcers.cuh`. MultiRMSE, MultiLogloss and
MultiCrossEntropy are not registered for CUDA greedy training and remain gated.

## Runtime

Per-leaf candidate scoring uses shared observation weights, vector gradient
planes and MultiClass's negative-sum implicit coordinate. Parent statistics
retain expanded mantissas; explicit child histograms and right statistics round
to float at CUDA's boundaries. L2/Solar/LOO/Sat round after each calcer leaf;
Cosine retains expanded accumulators until its final score. Separate binary
exponents prevent intermediate square overflow/underflow before CUDA's float
rounding. This approximates double arithmetic with float pairs; it does not
claim full binary64 mantissa precision or NVIDIA numerical equivalence.

The existing Metal variable-leaf frontier and stable partition kernels implement
all three policies. Depthwise is bounded by depth16, Region by depth65535, and
Lossguide by its leaf capacity with arbitrary uint32 requested depth. Fixed-tree
replay on each permutation repeats the selected GPU partition rounds, preserving
stable row order. Search uses sampled gradients/weights; every dataset estimates
its own leaves with original observation weights and its own optimizer cursor.
The final dataset supplies exported values, weights and loss.

Vector Newton/Gradient solves, iterative estimation and backtracking reuse the
symmetric vector solver. Published full-dimensional predictions and internal
optimizer predictions remain separate, including the MultiClass gauge. Failed
steps restore all affected cursors and the iteration number. Dynamic symmetric
CTR feature penalties are not applied to greedy search.

## Native integration and recovery

The additive C ABI uses `CBMVectorGreedyOptions`,
`cbm_multiclass_session_create_greedy` and
`cbm_multiclass_session_step_greedy`. Existing vector configuration, permutation,
optimizer recovery and close functions accept these handles. Symmetric and
greedy step functions reject the other handle mode before touching output.

Native model construction retains a vector per terminal leaf. Snapshot offset
units remain node IDs and leaf IDs; values are leaf-major with the approximation
dimension taken from validated training options. Scalar storage remains byte
compatible. Vector snapshots preserve all permutation cursors, optimizer state,
full untrimmed forest and metrics before best-model selection. Native readers
can repack leaf storage during JSON import; validation compares per-tree vectors
and predictions rather than assuming physical leaf ordering is persistent.

## Validation

- 158 shader cases: CUDA equations, implicit coordinate, padding, all scores,
  parent-weight residuals, Sat pole, tiny products and large Cosine intermediates.
- 273 private GPU cases: all three policies/objectives, five scores, Newton and
  Gradient, independent leaf equations, 64 outputs, P1/P2/P7/P64, sampling,
  backtracking, exact two-cursor recovery, failure rollback and depth40 paths.
- 225 native cases: fixed numeric forests, original class labels, one-hot/CTR
  inputs, four samplers, raw/quantized Pools, baselines, snapshots, initial
  models, best-model selection, JSON/CBM and GPU prediction.
- 8283 combined tests +16 subtests; 2293 alternate acceptance cases; 316 installed GPU paths; 92 CLI variants. Nine snapshots written by the prior native scalar binary resume exactly across numeric, one-hot and CTR inputs.

No CPU CatBoost fitting or NVIDIA execution is used for these checks. Remaining
CUDA parity work includes dynamic CTRs,
grouped Ordered boosting, complete CUDA mutable GPU RNG and other backend gaps.

Checkpoint `20260913T123335Z` is installed. Wheel SHA256 `f639c50c4b87a26e54a3685ba78a53baa4b066848ca5948b7f15b68a12dde41b`. Both extensions, CLI, changed sources, scripts and logs are preserved; prior checkpoint `20260913T115334Z` remains intact.

## Standalone lifecycle and vector evaluation

Checkpoint `20260913T130953Z` is installed, wheel SHA256 `0b291c2a9b5a10de95fb87f3d64c8545f155beaa64519b47d894fc7745554aff`. The preceding
`20260913T123335Z` checkpoint remains preserved. The C ABI adds counted vector evaluation
creation/tree-upload calls while retaining the scalar entry points. Each row
walks the tree once and adds all output coordinates. Uploads validate graph,
counts, finite values and peak resident bytes before replacing either buffer.

Standalone greedy models retain vector terminal values and scale/bias. The
normal category preparation path preserves OnAll one-hot thresholds, original
learn/evaluation category fingerprints and full inference CTR tables. Public
multiclass label/class-weight handling is shared with symmetric training;
RMSEWithUncertainty uses the same scalar-target category preparation.

Snapshots retain numeric leaf matrices, offsets counted in leaves, all published
permutation cursors and all class-major optimizer cursors, even with one dataset.
MultiClass's optimizer gauge is never reconstructed from published predictions.
Scalar snapshot arrays/fingerprints remain compatible. Negative uncertainty NLL
is valid. Validation stays on the GPU, callback metrics are detached, snapshots
retain the full forest before best-model trimming, and extension restores the
untrimmed state. Scalar and vector models use the iterative JSON serializer.

Training's fused raw-leaf-times-rate update may differ by float32 rounding from
evaluation of already-rounded exported leaf values. Snapshot continuation is
bit exact; reader comparisons allow this documented float32 arithmetic boundary.
Standalone multiclass Exponent now follows CatBoost's first-output convention.

- 68 evaluator cases and 336 public/lifecycle cases pass.
- 8687 combined tests +16 subtests; 2629 alternate acceptance cases;
  343 installed GPU paths; all 92 native CLI variants rerun successfully.
- Nine scalar snapshots written by the preceding standalone sources/runtime
  resume exactly across numeric, one-hot and CTR data for all three policies.
- Five alternating warmed measurements on 65536 rows and 16 trees, vector versus
  repeated scalar **Metal** evaluation: 3 outputs: 1.10x wall / 2.99x GPU; 7 outputs: 1.21x wall / 3.95x GPU; 64 outputs: 1.25x wall / 10.00x GPU. All compared prediction arrays
  are bit identical. Raw timings, GPU counters and source scripts are archived.

No CPU CatBoost fitting, live NVIDIA comparison or other M-series hardware run
was performed. Remaining standalone CTR types/target grids and broader CUDA
feature, random-state and performance parity remain open.

The later validation allocation optimization improves end-to-end evaluation
without changing GPU arithmetic or prediction bits. See
[VALIDATION_FASTPATH.md](VALIDATION_FASTPATH.md) for measurements against this
vector-lifecycle checkpoint.
