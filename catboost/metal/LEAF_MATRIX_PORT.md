# Coupled leaf matrix solver

[`metal_leaf_matrix_kernels.h`](native/metal_leaf_matrix_kernels.h) implements
the common leaf-system regularization and solve needed by PairLogitPairwise
and QueryCrossEntropy. Its 41 actual Metal tests pass on the M3. This is a
mathematical component; it does not implement pairwise split search or expose
either objective as a complete trainer.

## CUDA correspondence

The leaf matrix comes from
[`matrix_per_tree_oracle_base.h`](../cuda/methods/leaves_estimation/matrix_per_tree_oracle_base.h).
For `L` leaves and the complete projected Hessian `H`, regularization adds
`-non_diag_l2/L` to every off-diagonal and
`non_diag_l2*(1-1/L)+l2` to every diagonal. An original diagonal equal to zero
receives an additional `10` before regularization.

Pure PairLogitPairwise fixes the last coordinate at zero and solves the
leading `(L-1)` square system. QueryCrossEntropy has a point-diagonal component
and retains all `L` coordinates. GPU pairwise scoring is limited to depth
eight by the inspected CUDA options, so the helper supports at most 256
leaves. Split-score matrices use different average-diagonal stabilization;
that rule must not replace leaf regularization.

[`descent_helpers.cpp`](../cuda/methods/leaves_estimation/descent_helpers.cpp)
solves CUDA's leaf system in host double precision. This Metal implementation
uses high/low float pairs for regularization, Cholesky factorization, and both
triangular solves. Device-backed matrix storage avoids a threadgroup-memory
limit; independent rows of each sequential factorization column execute
together. Tests retain a ridge of `0.25` beside Hessian entries of `2^30`,
where ordinary float32 matrix formation would lose the ridge entirely.

The result is an unscaled incremental direction. CUDA's
[`RegularizeImpl`](../cuda/methods/leaves_estimation/oracle_interface.h) masks
underweight leaves after adding the direction to the current point; it does
not remove their coordinates from the coupled solve. Pure pairwise's final
fixed coordinate remains zero during estimation. Its eventual unweighted
leaf centering happens only after estimation, before shrinkage.

## GPU contract

The independent source string is `CBMMetalLeafMatrixSource`.
`LeafMatrixParams` is 32 bytes:

```text
uint leaves, has_diagonal_part, reserved0, reserved1;
float l2, non_diag_l2, min_leaf_weight, step;
```

Inputs and outputs use a physical row stride of `leaves`, including when the
last coordinate is removed. The caller validates `1 <= leaves <= 256`, finite
nonnegative regularization and minimum weight, and a finite trial step.

| Kernel | Buffers | Dispatch |
| --- | --- | --- |
| `RegularizeLeafMatrix` | 0 full row-major float Hessian; 1 output float2 matrix workspace; 2 parameters | `leaves*leaves` threads |
| `SolveLeafMatrix` | 0 float2 workspace, overwritten with its factor; 1 float gradient; 2 float direction; 3 atomic uint status; 4 parameters | One full 256-thread group |
| `UpdateLeafMatrixPoint` | 0 float current point; 1 float direction; 2 original float leaf weights; 3 float trial point; 4 atomic uint status; 5 parameters | `leaves` threads |
| `ReduceLeafMatrixDirectionalDot` | 0 float gradient; 1 float direction; 2 output float2 high/low dot product; 3 parameters | One full 256-thread group |

Clear the status before starting a transaction. Status bits are `1` for
nonfinite/asymmetric input or invalid leaf weight, `2` for a nonpositive
Cholesky pivot, and `4` for a nonfinite result. The owner must observe status
before accepting the update. The checked-in CUDA LAPACK wrapper only rejects
negative LAPACK status; its nonpositive-definite behavior is not reproduced.
This helper explicitly rejects a nonpositive pivot.

## Verification and remaining work

[`test_leaf_matrix_kernels.py`](tests/test_leaf_matrix_kernels.py) compiles
[`leaf_matrix_probe.mm`](tests/leaf_matrix_probe.mm) and compares actual GPU
results with independent NumPy double matrix equations. It covers both
coordinate conventions through 256 leaves, diagonal rescue, non-diagonal
regularization, disconnected graphs with a small ridge, underweight masking,
directional products, repeatability, and explicit invalid-system status.
No CPU CatBoost model is trained.

Production integration still needs complete pairwise/query projections,
candidate matrix scoring, iterative objective recomputation and backtracking,
and standard model/lifecycle checks. Compensated float pairs improve precision
but do not establish bitwise equivalence to CUDA's host-double solver.
