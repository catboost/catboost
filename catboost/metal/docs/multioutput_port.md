# CUDA vector objectives on Metal

The shared vector tree session supports `MultiRMSE`, `RMSEWithUncertainty`,
`MultiLogloss`, and `MultiCrossEntropy`, in addition to the two multiclass losses.
All output dimensions share each selected tree structure. Target derivatives,
histograms, leaf directions, cursor updates, and trial objectives run on Metal.
There is no CPU fitting path and no collection of independently fitted ensembles.

The port follows `cuda/train_lib/multiclass.cpp`,
`cuda/targets/multiclass_targets.{h,cpp}`, and
`cuda/targets/kernel/multilogit.cu`. The registered CUDA trainer includes these
four losses. `MultiRMSEWithMissingValues` is not included in this increment.

| Objective | Target shape | Raw output | Weighted row loss | Negative gradient |
| --- | --- | --- | --- | --- |
| MultiRMSE | N × D | D real values | w sum(error²) | w error per dimension |
| RMSEWithUncertainty | N | mean, log standard deviation | w (log(2π)/2 + log σ + error²/(2σ²)) | w error; w (error²/σ² − 1) |
| MultiLogloss | N × D binary | D logits | w mean(binary cross entropy) | w (target − sigmoid) |
| MultiCrossEntropy | N × D in [0,1] | D logits | w mean(binary cross entropy) | w (target − sigmoid) |

Uncertainty uses CUDA's natural mean gradient: its first component deliberately
omits inverse variance. Its diagonal Hessian is `(w, 2w error²/σ²)`, with zero
cross terms. The inverse variance uses `exp(min(-2 log σ, 70))`, matching CUDA.
MultiRMSE's diagonal Hessian is w. The multilabel Hessian is `w p(1−p)`;
unlike MultiClassOneVsAll, these objectives do not clip probabilities to 1e−7.
The multilabel loss is averaged over dimensions, but its derivatives are not.

CUDA sends these diagonal matrices through its symmetric block Cholesky solver.
Metal solves the equivalent diagonal system directly on the GPU. Newton adds
`l2_leaf_reg` to each diagonal. Gradient mode uses leaf weight in place of the
curvature. Nonempty singular Newton blocks report an error; empty or masked
leaves return zero directions. All reductions use compensated float expansions
because M-series Metal does not provide the CUDA host solver's double precision.

MultiRMSE reports `sqrt(sum(weighted SSE)/sum(weights))`. The other losses report
the mean weighted row loss. The shared backtracking walker uses the negative
unnormalized objective, including MultiRMSE's SSE and the multilabel dimension
average. Sampling changes structure gradients/weights; final leaves use original
weights. The implementation retains each permutation's independent cursor and
leaf estimates, exports the last permutation, and snapshots exact optimizer state.

`_multioutput.train` and `Session` accept row-major targets and infer D. They expose
the same options and results as `_multiclass`, including leaf values `[trees,L,D]`
and predictions `[N,D]`. The additive C constructor is
`cbm_multioutput_session_create`: it accepts dimension-major targets and the same
64-byte parameter structure as multiclass. Its objective IDs are 2 (MultiRMSE),
3 (uncertainty), 4 (MultiLogloss), and 5 (MultiCrossEntropy). Existing multiclass
step, sampling, permutation, feature-penalty, and snapshot functions accept this
handle. There is no C−1 gauge for these objectives.

`_multioutput_math.objective_and_leaf_directions` provides an independent hashed
diagnostic dylib. Its public arrays are dimension-major, and its diagnostic C IDs
are 0 through 3, separate from the training IDs. Tests compare actual Metal output
with scalar equations and fixed-partition leaf references; they do not fit CPU
models. The connected tests check shared-tree learning, traversal, original-weight
leaves, all backtracking modes, and bit-exact four-permutation continuation.

These tests establish implemented behavior and math agreement on the M3. They do
not establish full numerical, quality, or performance parity with NVIDIA CUDA.
