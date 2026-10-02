# Full-matrix Simple leaves

Numeric PairLogitPairwise and QueryCrossEntropy now accept
`leaf_estimation_method="Simple"` through native task_type="GPU" and the
standalone ranker. The installed build passes 100 resident/default cases,
34 native acceptance cases and 12 standalone lifecycle cases. Checkpoint `20260913T070739Z` passes **5325 combined tests plus 16 subtests**,
**682 alternate acceptance cases**, eighteen installed GPU paths and
Simple CLI fit/CBM/GPU prediction for both targets. The previous optimized
QCE checkpoint 20260913T065034Z remains available for recovery.

## CUDA correspondence

`cuda/methods/pairwise_oblivious_trees/pairwise_structure_searcher.cpp` reads
the final winning candidate's solution and matrix diagonal for Simple leaves.
`pairwise_score_calcer_for_policy.cpp` records the diagonal before split
stabilization. `pairwise_kernels.cpp::TCholeskySolverKernel::Run` centers the
solution only when removing the final coordinate (pairwise Laplacians).
`FixSolutionLeavesValuesLayout` moves the candidate's adjacent child index
to the model's highest leaf bit and reverses the final one-hot split.

The Metal candidate runtime can now reproject a bounded candidate range.
After selecting the last split, the trainer solves only that winner again
using the same resident weak target and parent IDs. A GPU export kernel
copies the solution and original diagonal into model leaf order. Learning
rate application and prediction updates use the existing GPU path.

No original-target Newton/Gradient leaf walk or document-weight masking is
applied in Simple mode. Its model weights are the sampled matrix diagonal,
which may all be zero if sampling removes the entire weak target. Metrics
continue to use original target weights. Backtracking settings have no effect
on its single solve, as on CUDA. Default leaf iteration count is one; explicit
counts above one are rejected. This implementation requires depth 1..8 and
a split candidate. CUDA's depth-zero Simple search does not produce a final
solution; this port rejects that case explicitly.

## Verification

- Complete forests match independently assembled/stabilized candidate matrices
  across seven pairwise/six QCE scores, every supported target sampler, numeric
  and one-hot predicates, and candidate banks wider than a resident tile.
- Depth one/eight, repeated splits, empty leaves and three backtracking options
  preserve the single candidate solve and raw diagonal semantics.
- Native prequantized models match independent forests, including prepared
  winner-order pair sampling, baselines and non-diagonal regularization.
- Native and standalone snapshots preserve predictions, leaves, matrix weights
  and metric histories exactly; CBM/JSON models predict correctly on Metal.
  JSON weights preserve their float32 source values; its double text reader
  can round the promoted value by one double ULP. Empty weak targets work.
- 122 existing Newton/Gradient full-matrix resident regressions also pass.

No CPU CatBoost fitting or NVIDIA execution was used. Full CUDA global RNG
consumption and categorical/P4 target histories remain incomplete.
