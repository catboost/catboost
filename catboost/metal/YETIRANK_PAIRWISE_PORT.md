# YetiRankPairwise / PFoundF on Metal

Numeric Plain and native one-hot/CTR P4 training are connected through ordinary CatBoostRanker with
`task_type="GPU"`, the native CLI, and CatBoostMetalRanker. Simple (default),
Newton and Gradient leaves use the existing full-matrix search/solver, with
all seven structure score enums and depth up to 8. Checkpoint `20260913T115334Z` is installed: **7627 combined tests plus 16
subtests**, **2068 alternate acceptance cases**, 289 installed GPU paths and
65 native CLI variants (eight numeric, fifteen one-hot, forty-two CTR). Wheel SHA256 `9ece3715d82bc136b173b65c90bd9baf3588f9d9e95ffa80b3b1f283a1cc0b82`.

## CUDA source correspondence

- `cuda/targets/pfound_f.cpp`: query/document sampling, ordered gathers, query
  maximum removal, exponentiation, pair-matrix generation, Bayesian scaling,
  tiny-weight filtering and final point gradients.
- `cuda/targets/kernel/pfound_f.cu`: packed 1024-row tasks, 256 lane seeds, four
  draws per lane and permutation, descending perturbed ranks, adjacent weights
  `0.15*decay^(rank-1)*abs(relevance difference)/permutations`.
- `GetMaxQuerySize`: min(2*mean_query_size+8,1023). This cap also applies with
  No/Bayesian bootstrap. Bernoulli Object takes ceil(fraction*query_size), with
  a minimum of two when available. Bernoulli Group samples whole queries.
- Pair generation uses original relevance. The final pair mass uses the
  lower-index sampled document's supplied weight, after raw-weight filtering
  at 1e-20. Weak curvature is pair mass for every score enum.
- Simple exports the winning weak split solution and raw matrix diagonal.
  Newton/Gradient regenerate a fixed target before leaf estimation, using
  Bayesian temperature 1 independently of the weak bootstrap configuration.
  Fixed pairs are oriented by relevance; each leaf step uses PairLogit pair
  derivatives. Original document weights are retained for these leaf methods.
- The stochastic oracle's value is zero. The public tracker evaluates PFound,
  using original effective query weights, rather than this placeholder value.

## Resident implementation

`metal_query_sampler_runtime.h` sorts random priorities and query IDs on GPU,
computes bounded/saturated prefix sums and compacts sampled documents back
into original order. `metal_pfound_pair_runtime.h` gathers target data and
creates PFound pair matrices on GPU; only compact shape/offset metadata is
read on the host to construct the CUDA task geometry. Generated endpoints and
statistics feed the same resident full-matrix solver as PairLogitPairwise.
Logical pair count can change without reallocating its capacity. Empty weak
targets and all-singleton groups are valid. Pending consumers must complete
before generated pair buffers are replaced. Auxiliary GPU time is included in
training statistics.

Combined core, sampler, generator and matrix allocations are bounded before
allocation by the existing 1 GiB workspace limit. Storage uses the smaller
of capped sampled rows times permutations or dense pair count; candidate-sort
entries remain capped at 2^24. Original logical pair IDs can span up to
2^32-2 pairs, with UINT32_MAX reserved as a sentinel. This removes quadratic
GPU storage when sparse generation is cheaper; the combined 1 GiB workspace
budget still determines which sizes can train.

## Explicit differences and remaining scope

Sampling and Bayesian weights use the existing Metal per-item/per-absolute-
iteration RNG convention with separate weak/fixed domains. The inner PFound
permutation generator follows CUDA lane seed arithmetic. This is deterministic
across snapshots, but is not the complete mutable CUDA global GPU random-
buffer consumption protocol and does not promise identical CUDA random draws.
P1 models record `item_iteration_domains_v1`; multiple datasets record
`item_iteration_dataset_domains_v2`. See YETIRANK_PAIRWISE_CTR_PERMUTATIONS.md.

CUDA stores local query IDs in uchar, which aliases queries when more than 256
short queries share a packed task. Metal keeps uint IDs and tests per-query
isolation. This is a correction of the source indexing defect, not bitwise
replay for those affected CUDA layouts. Stable sorting and compensated matrix
reductions also replace source nondeterministic tie/atomic reductions.

Poisson, MVS, backtracking, non-Classic modes and Ordered ranking remain
outside this path. Native simple CTR P4 is connected. Simple requires depth 1..8 and one
leaf iteration; Newton/Gradient also support depth zero. Public PFound
training validates relevance in [0,1]. Text/embedding/custom/Combination
objectives and global CUDA parity remain separate open work.

## Verification

- 68 pair-kernel cases: seeded independent pairs, task boundaries, uint64 seed
  limits, Bayesian scaling, tiny filtering, underflow, finite-output guards.
- 58 sampler cases: minimum-two/ceil cardinality, caps, masks, tie order,
  recursive prefixes and recovery after invalid masks/capacity.
- 46 resident integration cases: independent weak scores, fixed Bayesian leaf
  matrices, Newton/Gradient updates, empty targets, repeated shapes and budgets.
- 111 complete private forest cases: seven scores, three leaf methods, four
  sampling configurations, incremental topology, depth edges and exact replay.
- 91 native API cases: fixed-quantization independent forests, baselines,
  snapshots, PFound history, saved CBM/JSON GPU readers and invalid options.
- 23 standalone lifecycle cases: snapshot identity and incompatibility,
  original-weight PFound, best-model maximization, serialization and defaults.

Snapshot replay and CBM reads are exact. JSON's decimal parsing can shift a
promoted double by one ULP; tests retain exact source float32 leaf values and
strict double prediction tolerances. No CPU CatBoost fitting or NVIDIA run
was performed. Full CUDA quality/performance equivalence is not established.

## Zero-pair compaction and measured performance

Generated pairs are stably compacted at CUDA's raw-weight 1e-20 threshold
before matrix projection, reusing the existing solver radix workspace.
A four-bit flag sort requires one radix pass; full-key sorting retains all
eight. Filtering is before document/query-weight multiplication. Dense
pair generation capacity is still reserved; this step reduces search work.

99 focused sorter/runtime cases cover partial-key sorting, odd/even passes,
aliases, command reuse, compacted counts and filter order. 226 additional
forest/sampler/solver checks pass, followed by the full 5762-test run plus
16 subtests and 773 alternate native acceptance cases.

Three alternating warmed runs on this M3 Pro used 4096 learn rows, 1024
validation rows, four features and ten depth-4 Simple trees:

| Query size | Dense median seconds | Compacted median seconds | Speedup |
|---|---:|---:|---:|
| 16 | 0.3032 | 0.2401 | 1.26x |
| 64 | 1.1146 | 0.3796 | 2.94x |
| 256 | 4.4741 | 0.4354 | 10.28x |

All saved predictions, leaf values, leaf weights and metric histories are
bit-identical across every run. Full arrays, timings, hashes and scripts
are preserved in this release's pfound-compact-benchmark directory.
These measure the optimization against the earlier Metal dense build;
no CUDA/NVIDIA speed or quality equivalence is implied.

## Adaptive sparse generation

Sparse generation is now connected to the resident trainer and native package.
It emits one contribution slot per sampled row/permutation, stably sorts by
original dense pair ID and adds contributions in original permutation order.
Bayesian multipliers still use that original pair ID. The allocation planner
reserves min(dense pairs, maximum capped rows * permutations), plus bounded
sampler/core workspaces, before constructing GPU buffers. Each weak/fixed
sample selects its cheaper dense or sparse representation within that bound.
The existing solver sort workspace is reused for contribution reduction and
zero-pair selection. Sample target buffers also respect the CUDA query cap.

56 component and 54 resident cases verify raw float32 matrix/edge bits,
complete scores/leaves, sampling changes, query caps, allocation budgets and
Bayesian extremes. A reproduced sparse-only acceptance bug at temperature
43.8215866 is fixed: unused dense pair 1371 could have an infinite multiplier.
Above temperature 20 an allocation-free GPU pass validates all dense pair
multipliers, retaining dense rejection behavior without dense storage. At or
below 20 the bounded exponential's power is provably finite in float32.

378 focused tests, 5872 combined tests +16 subtests, 773 alternate acceptance,
24 installed GPU paths and six native CLI variants pass.

Three alternating warmed native runs compare this change with the previously
compacted dense generator, using the same 4096 learn/1024 validation rows,
four features and ten depth-4 Simple trees:

| Query size | Previous median seconds | Adaptive median seconds | Speedup |
|---|---:|---:|---:|
| 16 | 0.2032 | 0.2042 | 1.00x |
| 64 | 0.2966 | 0.3040 | 0.98x |
| 256 | 0.3433 | 0.3116 | 1.10x |

Every saved prediction, leaf, weight and metric history byte matches the
previous build. Workspace probes include both target generation and the
resident full-matrix solver (eight leaves, nine candidates, ten permutations):

| Original rows / query | Groups | Previous bytes | Adaptive bytes | Reduction |
|---|---:|---:|---:|---:|
| 16 | 256 | 9398176 | 9398176 | 1.00x |
| 64 | 64 | 38298784 | 12565616 | 3.05x |
| 256 | 16 | 153930032 | 12563696 | 12.25x |
| 1023 | 1 | 153833716 | 3156760 | 48.73x |
| 65537 | 1 | 157979892 | 7302936 | 21.63x |

Arrays, timings, memory reports, scripts, binaries and source hashes are
preserved in this release. Previous dense/compacted checkpoints
20260913T080844Z and 20260913T082455Z remain intact. These are M3 Pro measurements
against earlier Metal builds; CUDA/NVIDIA equivalence is not established.

Standalone and native PFound subgroup metadata are verified in the current
checkpoint, including Pool slicing and snapshot identity. See
[SUBGROUP_METRICS_PORT.md](SUBGROUP_METRICS_PORT.md). Prior sparse performance
reports remain in checkpoint 20260913T085149Z.

Original pair IDs now extend beyond 24 bits through the usable uint32
domain, separately from the stored-contribution limit. See
[WIDE_PAIR_IDS_PORT.md](WIDE_PAIR_IDS_PORT.md) for bounds and actual large-ID
GPU validation.

Classic YetiRank now accepts native CTR P4 with per-dataset oracle seeds,
cursors and MVS state. 97 runtime and 122 native checks include 32 independent
history/forest comparisons. See YETIRANK_CTR_PERMUTATIONS.md.

Generated-pair YetiRankPairwise CTR P4 is installed with per-cursor fixed
Bayesian targets and exact Simple model reuse. 55 new runtime and 156 native
cases pass; full CUDA GPU seed-buffer consumption remains open.
See YETIRANK_PAIRWISE_CTR_PERMUTATIONS.md.

Native and standalone Depthwise/Lossguide/Region now support one-hot
categories for all eleven registered scalar losses. 81 native and 75
standalone cases verify exact routing/recovery and readers. CTR scheduling
and vector greedy remain open. See GREEDY_ONE_HOT_PORT.md.

Native Depthwise/Lossguide/Region simple CTR P1/P4 is installed. All eleven
registered scalar losses, four CTR types, Sample/Group histories and complete
per-dataset recovery pass 204 new native and 146 private runtime cases.
Standalone greedy CTR lifecycle and vector greedy remain open. See GREEDY_CTR_PORT.md.

Standalone greedy Borders/FeatureFreq CTR P1/P4/P64 lifecycle is verified,
including OnAll thresholds, complete cursor snapshots and final-bank trimming.
130 new GPU tests and six exact legacy P1 recoveries pass. The native wheel
is reused byte-for-byte from 20260913T113921Z; its 65 CLI variants are preserved
without rerunning unchanged native code. See STANDALONE_GREEDY_CTR_PORT.md.
