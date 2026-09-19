# Sparse PFound pair-ID scaling

Checkpoint `20260913T091701Z` is installed: 6002 combined tests +16 subtests,
836 alternate native acceptance, 36 installed GPU paths and six CLI variants.
Wheel SHA256 `bb05e69f211510719493fd613ba58ceabd9f7a293ac89fee0f23e8c065c4aa22`.

The sparse generator retains CUDA's dense triangular pair IDs as keys and
random-stream inputs, while storing only sampled adjacency contributions.
The former shared 2^24 ID/storage bound prevented larger ranking datasets
from training even when their sparse working set fit in memory.

The planner now separates these quantities. At most 2^32-2 original logical
pairs are allowed; UINT32_MAX remains reserved for invalid/saturated entries.
Stored contributions and candidate-sort entries remain capped at 2^24.
Row count remains capped at 2^24. Generator, sampler, core and matrix-solver
allocation together must fit the existing 1 GiB workspace budget, so the
logical-ID upper bound does not promise training at every such size.

The query sampler's positive prefix scan saturates each wide SIMD shuffle
addition and block carry before uint32 overflow. Row and small-pair scans
retain their native SIMD sum. GPU dispatch-group rounding uses uint64 on
the host, including rare dense-ID validation passes at high Bayesian
temperature. No dense pair matrix or 64-bit pair buffer is introduced.

Validation on this M3 Pro:

- 58 prefix cases: lane, block and recursive boundaries, exact small sums,
  uint32 saturation and the reserved sentinel.
- 63 sampler cases, including five new cases above the old bound. A
  4,204,530-row / 4,110-query sampler returns 2,148,514,830 logical pairs with
  exact offsets and original-order document IDs, without dense allocation.
  This is sampler coverage, not a claim of complete training at that size.
- 58 resident cases, including four new 33/257-query tests with 1,023 rows
  per query. Weak/fixed raw targets and Bayesian edge bits match smaller
  reference prefixes exactly. Real generated endpoints map beyond 24-bit IDs.
- Twelve native cases train 33,759-row forests with Simple/Newton/Gradient,
  No/Bayesian/Bernoulli Object/Group sampling, exact snapshots and CBM GPU
  reloads. Fixed-quantization native and standalone resident results agree.
- Complete 6002 +16 run, 836 alternate cases, 36 installed GPU paths, six CLI.

Previous subgroup checkpoint 20260913T090442Z and sparse benchmark checkpoint
20260913T085149Z remain intact. This extends capacity while preserving the
explicit Metal item/iteration RNG protocol. Full CUDA mutable random-buffer,
categorical/P4 ranking, grouped Ordered, estimated features and other
previously documented parity work remain open. No CPU fitting or NVIDIA run.
