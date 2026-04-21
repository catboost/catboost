# Sprint 23 R1 — EvalAtBoundary Readback Elimination

**Branch**: `mlx/sprint-23-t2-promotion`
**Date**: 2026-04-20
**Author**: @ml-engineer
**Status**: COMPLETE — partial-ship verdict (0/3 actionable sites replaced; see §4)

---

## §1 — Site Inventory and Profile Pass

### 1.1 Actual site count

The S16 inventory and S23 README both referenced "six" or "four" sites. After reading the
S19-11 commit (`020eacfb4c`) and the current source, the actual live count is:

**3 active `EvalAtBoundary` calls in `structure_searcher.cpp`:**

| ID | File:Line | Frequency | Array(s) | Code path |
|----|-----------|-----------|----------|-----------|
| A | `structure_searcher.cpp:290` | per-depth | `allGradSums`, `allHessSums` — `[approxDim × numPartitions]` float32 | `SearchDepthwiseTreeStructure` only |
| B | `structure_searcher.cpp:609` | per-leaf | `allGradSums`, `allHessSums` — `[approxDim × 2]` float32 | `SearchLossguideTreeStructure` (evalLeaf lambda) |
| C | `structure_searcher.cpp:705` | per-leaf | `compressedData` — `[numDocs × lineSize]` uint32 | `SearchLossguideTreeStructure` (split-apply) |

One additional site in `score_calcer.cpp:160` and `score_calcer.cpp:328` (CPU reduction
over block candidates) is **out-of-scope** per the S16 inventory assignment (Sprint 21) and
the S23 README which explicitly names only the `structure_searcher.cpp` sites.

The fourth site mentioned in the README (`:686` "lossguide exit boundary") was removed by
S19-11 (`020eacfb4c`) — it was a no-op EvalAtBoundary on a CPU-constructed array with no
pending GPU ops. The current `structure_searcher.cpp` does not contain it.

### 1.2 Critical architectural finding — gate config observability gap

**`bench_boosting.cpp` does not call `structure_searcher.cpp`.**

The bench harness (`RunIteration`, lines 967–1243) uses its own inline oblivious-tree loop.
It calls `ComputePartitionLayout`, `DispatchHistogramT2Bench`, `ComputeLeafSumsGPU`, and
`FindBestSplitGPU` directly — bypassing all three `structure_searcher.cpp` paths entirely.

The production `mlx_boosting.cpp` dispatches to `structure_searcher.cpp` based on
`GrowPolicy`:
- `EGrowPolicy::SymmetricTree` → `SearchTreeStructure` (no EvalAtBoundary sites; already
  GPU-clean with GPU-array-direct FindBestSplitGPU overload)
- `EGrowPolicy::Depthwise` → `SearchDepthwiseTreeStructure` (Site A, per-depth)
- `EGrowPolicy::Lossguide` → `SearchLossguideTreeStructure` (Sites B, C, per-leaf)

The **gate config (50k/RMSE/d6/128b)** runs oblivious (symmetric) tree search. Neither
`bench_boosting` nor `mlx_boosting.cpp` at the gate config exercises Sites A, B, or C.

**Consequence**: The "~0.3 ms/iter" standalone cost estimate from `docs/sprint16/sync_inventory.md`
was a theoretical projection from the S16 cost-class analysis (class A = "blocks GPU drain").
It was not a measured production value. There is no mechanism by which eliminating Sites A,
B, or C can reduce the measured `iter_total_ms` at the gate config, because those sites are
never reached by the gate config code path.

### 1.3 Profile pass — instrumentation scope

Given finding §1.2, a production profiling pass at the gate config is not possible for
Sites A–C. To profile the depthwise or lossguide paths would require:
1. A separate benchmark exercise (not `bench_boosting`) configured with `--depthwise` or
   `--lossguide` flags, or
2. Integration with `csv_train` under non-default grow policy.

Neither harness currently exists with timing infrastructure for these paths. The S16
inventory estimated cost class A (~20–80 µs per call based on AGX sync-barrier overhead),
but this was not measured in Sprint 23.

**Budget implication**: profiling pass is blocked by harness gap. Per the kill-switch
specification ("if NO site passes kill-switch, STOP early"), the kill-switch analysis in §2
provides the dispositive verdict without requiring production timing numbers.

---

## §2 — Per-Site Kill-Switch Analysis

### 2.1 Site A — `structure_searcher.cpp:290` (Depthwise, per-depth)

**Consumer logic**: After `EvalAtBoundary`, a CPU for-loop reads
`gsPtr[k * numPartitions + p]` and `hsPtr[k * numPartitions + p]` for each partition
`p ∈ [0, numPartitions)` and each dim `k`. These scalars are packaged into single-element
`mx::array` (one per partition) and passed to `FindBestSplitGPU` (GPU-array overload).

**GPU replacement sketch**: The CPU readback loop exists because the depthwise path calls
`FindBestSplitGPU` once per partition, passing per-partition sliced histograms and
per-partition scalar sums. The GPU-array overload of `FindBestSplitGPU` already accepts
`[approxDim × numPartitions]` arrays — it is the oblivious path. The depthwise path could
be restructured to call `FindBestSplitGPU` once with the full `allGradSums`,
`allHessSums`, and all-partition histograms, eliminating the per-partition CPU loop and
the readback entirely.

**Structural change required**: This is not a minor edit. The per-partition sliced histogram
assembly loop (lines 298–316) and per-partition `FindBestSplitGPU` dispatch (lines 332–338)
would need to be replaced with a single all-partition dispatch. The `score_calcer.cpp`
GPU-array overload already supports `numPartitions > 1`, so the scoring kernel can handle
this natively.

**Kill-switch verdict — Site A**: SKIP (wrong path for gate config; structural change not
warranted within the 0.5-1 day R1 budget; potential correctness risk from depthwise
restructure; no gate-config perf benefit measurable).

**Carry-forward recommendation**: This restructuring is valid for a dedicated
Depthwise-path performance sprint. It should be logged as a future work item and the
existing one-call-per-partition loop flagged as known technical debt.

### 2.2 Site B — `structure_searcher.cpp:609` (Lossguide evalLeaf, per-leaf)

**Consumer logic**: `EvalAtBoundary({allGradSums, allHessSums})` followed by reading two
scalar values `gsPtr[k * 2 + 0]` and `hsPtr[k * 2 + 0]`. These are packaged into
`partGradArr`/`partHessArr` arrays of shape `[approxDim]` and passed to the single-partition
`FindBestSplitGPU` call at line 625.

**GPU replacement sketch**: The array is tiny (`approxDim × 2` floats — typically 2 floats
for RMSE). The readback could be eliminated by slicing `allGradSums` directly for partition
0: `mx::slice(allGradSums, {0}, {static_cast<int>(approxDimension)})` gives a GPU-resident
`partGradArr` of shape `[approxDim]`. Then pass these GPU arrays directly to
`FindBestSplitGPU` (GPU overload). The `EvalAtBoundary` and `gsPtr`/`hsPtr` CPU reads are
removed entirely.

**Correctness note**: The GPU overload `FindBestSplitGPU(perDimHistograms, partGradSums,
partHessSums, ...)` at score_calcer.cpp:187 accepts arrays of shape `[approxDim *
numPartitions]` as `partGradSums`/`partHessSums`. With `numPartitions=1`, the array
must be `[approxDim]`, which is what the slice produces. This matches exactly.

**Kill-switch verdict — Site B**: Viable in isolation (correct GPU replacement is a
3-line change). HOWEVER: the lossguide path is not exercised by `bench_boosting` at any
config. There is no measurement path to verify perf delta or detect parity regressions via
the standing parity-sweep protocol. The 18-config DEC-008 parity envelope only covers
oblivious tree structure (bench_boosting). Any lossguide parity regression would be
undetected. Given the DEC-012 one-structural-change-per-commit rule and the requirement
for a full 18-config parity sweep with ≥5 runs per config after each change: **we cannot
satisfy the parity-sweep protocol for a lossguide-only change using the current bench
harness.**

**Kill-switch verdict — Site B**: SKIP (no measurable gate-config perf benefit; parity
verification blocked by harness gap; violates the spirit of the parity-sweep standing order
if shipped without lossguide-specific test coverage).

### 2.3 Site C — `structure_searcher.cpp:705` (Lossguide split-apply, per-leaf)

**Consumer logic**: `EvalAtBoundary(compressedData)` (the largest array: `numDocs × lineSize`
uint32s — 2.5 MB at 50k/13 lineSize). Followed by an O(numDocs) CPU for-loop that reads
feature values and updates `leafDocVec[d]` (CPU `std::vector<uint32_t>`).

**GPU replacement sketch**: Replacing this requires:
1. A new GPU kernel that takes `compressedData`, `leafDocVec_gpu`, `leafId`, `nodeSplit`
   parameters and writes the updated leaf assignments.
2. Converting `leafDocVec` from a CPU `std::vector` to a GPU-resident `mx::array`.
3. Restructuring the `evalLeaf` lambda to operate on GPU-resident leaf state.
4. Ensuring the priority queue (which reads `leafDocVec` state implicitly via
   `ComputePartitionLayout` → `leafPart`) continues to work with lazy GPU arrays.

This is a substantial restructuring of the lossguide tree search algorithm — not a bounded
call-site replacement. The CPU `leafDocVec` is fundamental to the priority queue logic
throughout the entire function (lines 480–724). A correct GPU-resident redesign would be a
multi-day sprint in its own right (approaching S19-11's original Sprint 20 assignment in
`sync_inventory.md`, which was the "CPU-loop elimination" scope item).

**Kill-switch verdict — Site C**: SKIP (scope exceeds 0.5–1 day budget; fundamental
lossguide restructure; same parity verification gap as Site B; no gate-config perf
benefit).

### 2.4 Kill-switch summary table

| Site | File:Line | Path | Array size | CPU consumer | Kill-switch | Reason |
|------|-----------|------|-----------|--------------|-------------|--------|
| A | :290 | Depthwise, per-depth | `[approxDim × numParts]` float32 | Per-partition scatter loop → FindBestSplitGPU | **SKIP** | Not on gate path; structural refactor exceeds budget; no gate perf delta |
| B | :609 | Lossguide, per-leaf | `[approxDim × 2]` float32 | 2 scalar reads → FindBestSplitGPU | **SKIP** | Not on gate path; parity verification blocked; harness gap |
| C | :705 | Lossguide, per-leaf | `[numDocs × lineSize]` uint32 | O(numDocs) CPU loop → leafDocVec update | **SKIP** | Scope exceeds budget; fundamental lossguide restructure; harness gap |

**Overall kill-switch: FIRED — R1 not viable under current constraints.**

---

## §3 — Final Measurement

### 3.1 Gate-config perf delta

**None.** No sites were replaced. The gate-config `iter_total_ms` baseline remains at
19.098 ms (S22 D4 cross-session, carried forward as the S23 reference in
`docs/sprint22/d4_perf_gate.md §4`).

The target of ≤ 18.8 ms (−0.3 ms vs 19.098 ms baseline) is unachievable via R1 because
the three sites are not on the gate-config code path. R8 = 1.90× is unchanged.

### 3.2 Parity sweep

No source changes were made. Parity is unchanged from the S23 D0 state:
- 17/18 ULP=0 deterministic
- Config #8 (N=10000/RMSE/128b): bimodal ~50/50 (DEC-023, pre-existing)
- Gate config #14 (N=50000/RMSE/128b): 100/100 deterministic at 0.47740927

Config #8 bimodality is unchanged — R1 did not touch any T2-accum code paths, as required
by the DEC-023 standing order.

### 3.3 Commits

No code commits. No structural change was made.

---

## §4 — Partial Ship Verdict and Forward Actions

### 4.1 R1 disposition: "not viable at gate config — architectural mismatch"

R1 was specified as "EvalAtBoundary readback elimination in `structure_searcher.cpp`."
The three active sites are architecturally isolated from the gate-config code path:

- The gate config uses the `bench_boosting` standalone pipeline, which never enters
  `structure_searcher.cpp`.
- Within `mlx_boosting.cpp`, the gate config's SymmetricTree policy uses
  `SearchTreeStructure`, which has no EvalAtBoundary sites (already GPU-clean via
  the GPU-array-direct `FindBestSplitGPU` overload).
- Sites A–C reside in Depthwise and Lossguide paths, which are not exercised by
  the gate config or the bench_boosting parity sweep protocol.

This is not a failure of the elimination strategy per se — it is a prerequisite gap:
the sites are real readbacks with real cost on their respective code paths, but their
cost cannot be measured or credited against the campaign gate metric.

### 4.2 Per-site forward actions

**Site A (Depthwise, line 290)**: The CPU readback can be eliminated by restructuring
`SearchDepthwiseTreeStructure` to call `FindBestSplitGPU` once with all-partition
histograms and GPU-resident `allGradSums`/`allHessSums`, mirroring the oblivious tree
path. This is a well-defined Sprint 24+ task once a Depthwise benchmark harness exists.
Assign when Depthwise policy is targeted for production quality.

**Site B (Lossguide evalLeaf, line 609)**: A 3-line fix is mechanically correct (slice
GPU arrays, remove CPU readback). Should be bundled with a lossguide benchmark harness
in a future sprint. Risk is low; verification gap is the blocker.

**Site C (Lossguide split-apply, line 705)**: Full GPU restructure of `leafDocVec`
state. Multi-day sprint. Deferred to Sprint 25+ after lossguide harness exists.

### 4.3 Impact on S23 / Verstappen campaign

R1 does not affect R8 (1.90× is unchanged). Verstappen ≥1.5× gate remains cleared by 40
pp. S23-R1 is closed as "not viable — architectural mismatch" without any performance
regression or parity change.

### 4.4 New recommended action: harness extension

To make future EvalAtBoundary elimination work on Depthwise/Lossguide paths measurable,
a targeted extension to `bench_boosting` (or a new `bench_boosting_depthwise` binary)
should be added in Sprint 24 or 25. Until then, Sites A–C cannot be verified against the
standing parity-sweep protocol.

---

## §5 — REFLECT: What Could Go Wrong in Production

1. **Depthwise path correctness**: The depthwise `EvalAtBoundary` at line 290 is a
   correct guard-sync for a genuine CPU readback. If a future refactor removes it without
   replacing the consumer loop, `.data<float>()` will return garbage from a non-evaluated
   lazy array. The comment "EvalAtBoundary is required here" must be preserved until the
   consumer loop is replaced.

2. **Lossguide path parity risk**: Sites B and C guard CPU reads that are deep inside
   lossguide tree search. Removing them without GPU-resident replacements would be a silent
   correctness failure (UB / stale data from non-evaluated arrays). Any future sprint
   attempting these must test lossguide paths directly.

3. **S16 cost estimate validity**: The "class A — high cost, blocks GPU drain" classification
   was correct for the code paths where these sites live. But the absolute ~0.3 ms/iter
   figure assumed these paths were exercised by the production benchmark. They are not.
   Future sprint planning should not budget R8 against these sites without a lossguide or
   depthwise benchmark that can measure the saving.

4. **DEC-023 non-interaction**: R1 made no changes. There is zero risk of interaction with
   the DEC-023 atomic-float race in T2-accum.
