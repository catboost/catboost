# PairLogitPairwise on Metal

Objective 15 is connected to the standalone CatBoostMetalRanker, private resident session and normal native
`CatBoostRanker(task_type="GPU", loss_function="PairLogitPairwise")` entry point.
The PairLogitPairwise introduction checkpoint `20260913T054109Z` passed 4941
combined tests plus 16 subtests and twelve installed GPU paths. Its native
acceptance packages remain in `.build/recovery/pair-matrix/{standard,no_cuda}`.
The newer installed QueryCrossEntropy checkpoint also passes every pairwise
regression; see IMPLEMENTATION_STATUS.md for current release hashes and counts.

## CUDA mapping

- `cuda/targets/pair_logit_pairwise.cpp`: supplied/generated edge weights are
  sampled before differentiating the weak target. L2 search uses edge mass;
  all other accepted scores use curvature. Score noise is ignored by this
  CUDA searcher, including when random_strength is supplied.
- `cuda/methods/pairwise_oblivious_trees/pairwise_structure_searcher.cpp`:
  every candidate solves a complete child-leaf Laplacian. Repeated winning
  splits remain in the tree until maximum depth, unlike pointwise symmetric
  search. Numeric and one-hot candidate layouts are checked separately.
- `cuda/methods/kernel/linear_solver.cu` and pairwise score kernels:
  stabilize the candidate matrix, fix its last coordinate, center the solution,
  then evaluate beta.G - 0.5 beta.H.beta using the original Hessian. Leaf
  regularization is a different operation and must not reuse this stabilizer.
- `cuda/methods/leaves_estimation/matrix_per_tree_oracle_base.h` and
  `descent_helpers.cpp`: final estimation uses original edges and original
  document weights. Newton uses curvature, Gradient uses edge mass. Empty
  document leaves mask the updated point. Fix the last coordinate during the
  walk; center all leaves once before shrinkage.
- AnyImprovement and Armijo consume attempted steps from the leaf iteration
  budget, allow up to 100 attempts before first acceptance, and halve rejected
  steps. One leaf iteration bypasses backtracking. The acceptance objective
  removes same-leaf edges, as CUDA's support-pair builder does. Reported PairLogit
  metrics include all original edges. Trial numerical failures have a separate
  status so rejected candidates cannot invalidate subsequent smaller steps.

## Resident implementation

`native/metal_pairwise_candidate_kernels.h` builds stable candidate/child-pair
keys. GPU radix sorting and compensated segmented reductions construct each
candidate's gradient and full symmetric Laplacian. Batched Cholesky solves and
quadratic scoring select the winner on the GPU. Only the small winner record
returns to the host to advance tree topology.

`native/metal_pairwise_matrix_runtime.h` owns target metadata, bounded candidate
batches and persistent sort workspace. Candidate batches shrink to fit the
runtime budget. The full session includes these allocations in its 1 GiB
workspace limit; outputs retain the existing 512 MiB limit. Scratch reuse is
necessary: allocating a fresh sort for every encoded tile would retain all
scratch until command completion. A fixed tree's sorted edge layout is reused
across leaf trials. Data, edges, derivatives, matrices, partitions and cursor
updates remain on Metal; no CPU CatBoost fitting is involved.

CUDA forms leaf systems with host double arithmetic. Metal uses compensated
float pairs for stabilization, factorization, solve and score accumulation.
This improves numerical fidelity without requiring GPU float64. It does not
establish bitwise agreement with an NVIDIA run.

## Verification and limits

Component coverage: candidate batches 26, reusable sort workspace 13, resident
matrix runtime 26. Full private training: 65 cases. Standalone public lifecycle: 37 cases. Native API: 36 cases,
including 14 fixed-quantization forest comparisons with a separately driven
resident trainer, weights, baselines, initial models, snapshots, generated and
unlabeled supplied pairs, metrics, best-model selection, standard CBM/JSON and
GPU prediction. Complete suite results are recorded in IMPLEMENTATION_STATUS
when the next checkpoint is finalized. Counts overlap broader regressions.

Connected scope: native and standalone public numeric Plain/DocParallel P1, depths 0–8, Newton and
Gradient leaves, No/Bayesian/Bernoulli/Poisson edge sampling, and all three leaf
backtracking modes. Private sessions also verify one-hot candidate equations.
Numeric data uses one effective permutation even when the shared option says
four. Native categorical/P4 and dynamic features are not connected. CUDA
Simple leaves are now connected in the installed build; see
SIMPLE_LEAVES_PORT.md. MVS is rejected, matching the CUDA non-diagonal target limitation.
The standalone ranker supports supplied pairs, original object/group weights,
weighted ranking metrics, best models, early stopping, callbacks, snapshot
continuation, and normal model export. Its `bayesian_matrix_reg` option maps to
CUDA's non-diagonal regularization. Only the native entry point uses shared
automatic pair generation; standalone fitting requires explicit pairs.

Sampling uses the existing Metal edge-index/absolute-iteration stream. Snapshot
continuation is exact for that stream; the complete CUDA global RNG consumption
protocol is not reproduced here. No live NVIDIA hardware comparison or other
M-series hardware validation has been performed.

## Local quality check

One warmed M3 Pro run trained 20 depth-four trees on 16384 rows, 16384 supplied
edges, six features and 16 borders in 0.519 seconds. On 4096 held-out rows,
NDCG@10 improved from 0.350 (constant-score shared tie convention) to 0.982;
PairLogit fell from 0.693 to 0.580. GPU and standard model-reader predictions
matched exactly. This is synthetic local quality evidence, not NVIDIA parity.
The release preserves the script, data-generation seed and measured report.

## Native one-hot categories

The native trainer now supports one-hot categorical features through ordinary
Pools and the CLI. CUDA counts learn plus validation categories; CTR history
is still gated for these objectives. See RANKING_ONE_HOT_PORT.md for validation.
