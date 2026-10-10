# Classic YetiRank CTR permutations

Native classic YetiRank now accepts multiple categorical CTR datasets, including
default P4. Each dataset owns its prediction cursor and MVS state; all share
the searched tree structure. The last dataset supplies the exported leaves.
YetiRankPairwise uses a different generated-pair oracle and remains P1.

## CUDA source and seed order

`cuda/methods/doc_parallel_boosting.h` selects one dataset for weak-target
search, then submits leaf-estimation tasks in dataset order.
`cuda/methods/leaves_estimation/doc_parallel_leaves_estimator.h` completes each
task's entire walk before starting the next task.
`cuda/targets/querywise_targets_impl.h` draws a fresh host uint64 seed for
every YetiRank point approximation; target constructors share that stream
without consuming extra draws. A one-iteration leaf walk takes one seed;
longer walks take I+1, including the unused final-point evaluation.

`train_lib/yeti_random.h` now generates those contiguous chunks in dataset
order. The resident trainer maps each chunk to its original dataset index
even when it estimates the searched dataset first. Other datasets replay the
fixed structure and run fresh leaf oracles, without preparing extra weak
targets. All leaves retain CUDA's zero-average correction.

Native snapshots persist every cursor, MVS state, used-feature state and the
total host draw count. Restore validates the draw count against completed
trees and the configured dataset count. Initial-model continuation uses the
absolute iteration for structure selection and a new fit's target stream.
The private automatic Python driver also persists dataset count in its RNG
state; existing P1 state dictionaries remain unchanged.

This preserves the translated host target-seed order. Bootstrap/score-noise
GPU streams retain the documented Metal adaptation; it does not establish
bitwise equivalence to complete CUDA GPU execution.

## Validation

- 97 runtime checks: independent MT64 host sequences, independent per-cursor
  leaf equations, all five samplers, selected datasets including the export
  dataset, P1/P2/P4/P7/P64, empty/depth-zero structures, delayed seeds and exact
  cursor/RNG/MVS recovery.
- 122 native checks: all four simple CTR types, Sample/Group histories,
  raw/quantized Pools, one/three leaf iterations, all five samplers, exact
  snapshots, changed-data/geometry rejection, initial models, baselines,
  evaluation and standard CBM/JSON/GPU readers.
- Those native checks include 32 independent-history forest comparisons. A
  small C++ probe executes upstream FastRng64 and Shuffle to obtain the shared
  Pool preprocessing order. Tests compute CTR histories independently, then
  compare the native host controller with the private resident driver.

No CPU CatBoost fitting or NVIDIA execution is used.

## Remaining

Generated-pair YetiRankPairwise P4 needs its separate weak/fixed target
schedule. Dynamic categorical tensors, grouped Ordered boosting, remaining
feature pipelines/options, memory scaling and cross-device CUDA comparisons
remain open. Public standalone ranking currently accepts one-hot categories;
the CTR P4 integration described here is the native CatBoost interface.

Checkpoint `20260913T105212Z` is installed: **6780 tests +16 subtests**,
**1422 alternate acceptance cases**, **100 installed GPU paths**, **35 CLI
variants**. Wheel SHA256 `2fa9362ba52f7879b952ded78f143861b5674a1d2c4adf7ee1926045886adb3b`. Both extensions, CLI, changed sources,
scripts and logs are preserved. Prior checkpoint `20260913T103310Z` remains intact.

YetiRankPairwise now also accepts native CTR P4. Its fixed pair targets are
created separately for each original cursor; Simple reuses the searched
model. See YETIRANK_PAIRWISE_CTR_PERMUTATIONS.md for streams and evidence.
