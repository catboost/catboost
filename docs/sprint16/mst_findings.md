# Sprint 16 -> Sprint 17 Scout: MST Findings & Kernel Teardown

Captured 2026-04-17. This document follows up the Sprint 16 per-stage baseline
(`baseline_results.md`) by zooming in on the one dominant stage: the histogram
Metal kernel (97.5-99.2% of iter time in every config). The goal is a ranked
set of Sprint 17 levers grounded in the actual kernel source.

## TL;DR

The production histogram kernel (`kHistOneByteSource` in
`catboost/mlx/kernels/kernel_sources.h:85-201`) ends with a **fixed-order
sequential reduction across all 256 threads** of a threadgroup. It contains a
`for (t = 1; t < BLOCK_SIZE; t++)` loop with a `threadgroup_barrier` every
iteration where **only thread `t`** contributes. That is 255 barriers + 255
passes over a 1024-float staging array per threadgroup — a strict serial tail
with 99.6% of the threads idle at each barrier. Replacing this with a standard
log-step tree reduction collapses the serial tail from ~255 steps to 8 steps
and is the recommended Sprint 17 headline lever.

Additionally, every thread owns a 1024-float `privHist[]` stack array (4 KB
per thread, 1 MB per threadgroup at BLOCK_SIZE=256). This almost certainly
spills to device memory on M-series GPUs whose register file is O(256 regs
per thread). Most of the kernel's wall time is likely spent loading/storing
this spilled private array rather than doing the histogram math.

---

## A. Metal System Trace capture

### A.1 Environment

- `xcrun xctrace` — version 16.0 (17E202). Installed and functional.
- macOS 25.3 (Darwin 25.3.0), Apple Silicon.
- `csv_train_profiled` built with `-O2 -DCATBOOST_MLX_STAGE_PROFILE` against the
  shipped MLX package (`/opt/homebrew/.../mlx`).

### A.2 Capture command (executed)

```bash
xcrun xctrace record --template "Metal System Trace" \
  --output .cache/profiling/sprint16/mst_10000_rmse_2026-04-17.trace \
  --time-limit 30s --launch -- \
  ./csv_train_profiled /tmp/bench_10000_50f.csv \
    --target-col 50 --loss RMSE --depth 6 --bins 128 \
    --iterations 20 --lr 0.1 --output /tmp/_m.cbmx
```

Trace bundle recorded successfully (4.2 GB at
`.cache/profiling/sprint16/mst_10000_rmse_2026-04-17.trace/Trace1.run/Attachments/trace-data.atrc`).
`RunIssues.storedata` is empty — no capture errors.

### A.3 Export limitation (known `xctrace` gap)

`xctrace export --toc` against this bundle fails with
`Document Missing Template Error`. This is a well-known behaviour of the
current `xctrace` CLI on Metal System Trace bundles: the raw `.atrc` stream
must be parsed by Instruments.app once before `xctrace export` can read it,
because Instruments lazily writes the template metadata on first open. The CLI
does not do this itself.

**Consequence.** I cannot emit per-kernel timing, SIMD occupancy, or
compute-vs-memory-bound ratios from the command line in this session. The
headline levers in section D therefore rest on **static kernel analysis** (B)
plus the per-stage profiler data we already have (baseline_results.md). The
trace bundle is preserved so that:
1. You (or a human-in-the-loop) can open it in Instruments and read off the
   numbers I could not extract here.
2. A future Sprint 17 ml-engineer can use it as the before-state against which
   to measure the kernel changes.

### A.4 What you should see when you open the trace

Open `Trace1.run` in Instruments and select the Metal System Trace tab. You
should see, for each of the 20 iterations × 6 depths × 3 `DispatchHistogram()`
calls per depth (1 hist + 1 count-hist dispatch if MinDataInLeaf > 1; just 1
otherwise for RMSE), one `histogram_one_byte_features` kernel invocation.
Specifically for `RMSE, 10k docs, depth 6, 128 bins` I expect:

- per-depth kernel wall time roughly matching the baseline profile:
  ~45 / 24 / 24 / 39 / 74 / 114 ms for depths 0..5.
- grid size `(256 * maxBlocksPerPart * 13, numPartitions, 2)`:
  at depth 0 (1 partition, avgDocsPerPart=10000, `maxBlocksPerPart=3`):
  `(256 * 3 * 13, 1, 2) = (9984, 1, 2)` — ~78 threadgroups.
  at depth 5 (32 partitions, avgDocsPerPart=312, `maxBlocksPerPart=1`):
  `(256 * 1 * 13, 32, 2) = (3328, 32, 2)` — 416 threadgroups.
- "GPU active" should be lumpy with gaps between command buffer submits; if
  gaps exceed ~1-2 ms per dispatch, there's submit-side overhead worth
  investigating.

---

## B. Histogram kernel teardown

**Source of truth.** `catboost/mlx/kernels/kernel_sources.h:85-201`
(`kHistOneByteSource`). Note: `catboost/mlx/kernels/hist.metal` is a legacy,
non-production kernel with a different algorithm (shared-memory SIMD-slice
reduction). The production path embeds the source string above into a
`mx::fast::metal_kernel()` call at `csv_train.cpp:921-950`. Do not look at
`hist.metal` when reasoning about current perf.

### B.1 Memory access pattern per thread

The per-doc loop at `kernel_sources.h:131-148`:

```metal
for (uint d = thread_index_in_threadgroup; d < myDocCount; d += BLOCK_SIZE) {
    const uint sortedPos = partOffset + myDocStart + d;
    const uint docIdx    = docIndices[sortedPos];
    const uint packed    = compressedIndex[docIdx * lineSize + featureColumnIdx];
    const float stat     = stats[statIdx * totalNumDocs + docIdx];
    for (uint f = 0; f < 4; f++) { ... privHist[f*256 + bin] += stat; }
}
```

- **Strided doc loop**: thread t processes docs `t, t+256, t+512, ...`.
- **Loads are gather-indexed, not coalesced**. Two reasons:
  1. `docIndices[sortedPos]` is a uint32 gather — the 256 threads in a
     threadgroup read 256 consecutive positions, which is coalesced.
  2. `compressedIndex[docIdx * lineSize + featureColumnIdx]` and
     `stats[statIdx * totalNumDocs + docIdx]` both use `docIdx` (post-gather),
     NOT `sortedPos`. So unless `docIndices` happens to be the identity
     permutation, these reads are scattered across device memory.
- At depth 0, `docIndices` IS the identity permutation
  (`ComputePartitionLayout` returns a sort by leaf; at depth 0 all docs are
  in leaf 0, so the sort is stable). At depth >= 1 the permutation is
  nontrivial and loads diverge.
- The 4-feature inner loop has a data-dependent branch
  (`if (bin < foldCountsFlat[foldBase + f] + 1)`) — a warp-divergent
  predicate on every doc. With 50 features split into 13 groups of 4,
  the last group has only 2 valid features so 2 of 4 bins always miss.

### B.2 Per-threadgroup histogram storage

`privHist[HIST_PER_SIMD]` at `kernel_sources.h:123`:

```metal
float privHist[HIST_PER_SIMD];  // HIST_PER_SIMD = 4 * 256 = 1024 floats
```

- This is a **per-thread stack array**, not threadgroup memory.
- 1024 floats × 4 bytes = **4 KB per thread**.
- At BLOCK_SIZE=256 threads per threadgroup, that's **1 MB of private state
  per threadgroup**.
- Apple M-series GPUs expose ~256 registers per thread maximum; a 1024-entry
  float array spills to **device memory** (the "thread-local" L1/L2-backed
  private address space). Metal calls this "private" storage but it lives
  off-chip when it spills.
- The staging buffer `stagingHist[HIST_PER_SIMD]` at line 160 is 4 KB of
  **threadgroup** memory — on-chip. This is small enough to fit.

For depth 5 with 32 partitions: 32 partitions × 13 groups × maxBlocksPerPart=1
= 416 threadgroups × 1 MB private = **416 MB of private allocation pressure**
across concurrently-schedulable threadgroups. The GPU can obviously only
schedule O(tens) concurrently at a time; but every launched threadgroup must
zero and touch a 1024-float array, forcing at minimum 4 KB × 256 of
initialisation traffic per threadgroup before any real work starts.

### B.3 Reduction strategy — the smoking gun

`kernel_sources.h:161-181`:

```metal
threadgroup float stagingHist[HIST_PER_SIMD];  // 4 KB threadgroup memory

if (thread_index_in_threadgroup == 0u) {
    for (uint i = 0u; i < HIST_PER_SIMD; i++) stagingHist[i] = privHist[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint t = 1u; t < BLOCK_SIZE; t++) {              // t = 1..255
    if (thread_index_in_threadgroup == t) {
        for (uint i = 0u; i < HIST_PER_SIMD; i++) {
            stagingHist[i] += privHist[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // 255 barriers
}
```

This is a **serial fold**. Only one thread out of 256 does work per iteration
of the outer loop; the other 255 are stalled at a barrier. The addition is
deterministic (threads contribute in index order) but the cost is
proportional to BLOCK_SIZE, not log(BLOCK_SIZE).

Per-threadgroup reduction cost:
- 255 barriers
- 255 × 1024 = **261,120 float adds on the serial path**
- Each active thread reads its own `privHist[i]` (private/spilled) and
  read-modify-writes `stagingHist[i]` (threadgroup memory, ~1 cycle)

If `privHist` is spilled (B.2 suggests it is), each of those 261,120 adds
pulls a float from device memory. At 4 bytes × 261,120 = **1 MB of device
traffic per threadgroup just in the reduction** — likely as expensive as the
entire accumulation phase.

**Why it was written this way.** The header comment at `kernel_sources.h:57-82`
("BUG-001 FIX") explains the design intent: the original implementation used
CAS atomics and was non-deterministic across dispatches. Switching to a fixed
serial addition order made the output bit-exact across runs. The serial
reduction was the quickest correctness fix. It is now the performance
bottleneck.

**Why a log-step tree reduction would preserve bit-exactness.** A Hillis-Steele
or butterfly reduction over threadgroup memory is itself a fixed-order
deterministic algorithm as long as the stride schedule is fixed. The
`kSuffixSumSource` kernel in the same file (lines 264-311) already uses
exactly this pattern for a 256-element scan — the authors know how to write
one. The trade-off is one additional 4 KB threadgroup-memory load per round
(8 rounds for 256 elements) vs. 255 sequential private-to-threadgroup
accumulations. Log-step should be 20-30× faster on the reduction phase.

### B.4 Grid / threadgroup geometry for 10k / depth 5 / 50 features

- `numFeatures = 50`, `numFeatureGroups = ceil(50 / 4) = 13`
- `numPartitions = 2^5 = 32` at depth 5
- `trainDocs = 10000`, so `avgDocsPerPart = 10000 / 32 = 312`
- `maxBlocksPerPart = clamp(ceil(312 / 4096), 1, 8) = 1`
- Grid: `(256 * 1 * 13, 32, 2) = (3328, 32, 2)` threads total
- Threadgroup: `(256, 1, 1)`
- Threadgroups: `13 * 32 * 2 = 832` threadgroups
- Threads per threadgroup: 256
- Total threads: 213k

At depth 0 with 1 partition and maxBlocksPerPart=3: grid
`(256*3*13, 1, 2) = (9984, 1, 2) = 78` threadgroups, 20k threads. The
threadgroup count is ~10× lower at depth 0 but each threadgroup processes
~30× more documents (3333 docs/block vs 312 docs/block). The serial reduction
cost is **constant** per threadgroup (it's always 255 iterations × 1024
entries); at depth 5 with 832 threadgroups the aggregate reduction work is
~10× more than at depth 0. This is consistent with the observed depth 5
histogram time being ~2.5× depth 0.

### B.5 Wasted work in the kernel

1. **Out-of-range block early-return**
   (`kernel_sources.h:105`): `if (myDocStart >= partSize) return`. Fine.
2. **Per-doc bin-range check** (`line 144`):
   `if (bin < foldCountsFlat[foldBase + f] + 1)` runs every iteration of the
   4-feature inner loop, on every doc. For the 13th feature group with only
   2 valid features, 2 out of 4 branches always miss. Cost is modest — a
   single predicated load — but it does serialise the warp across the
   comparison.
3. **Thresholded global write** (`line 192`):
   `if (abs(val) > 1e-20f)` in the output writeback. This only matters for
   atomic contention — a zero-skip for empty bins. Rare win, usually no-op.
4. **Double-launch of feature-column-index vector**
   (`kernel_sources.h:110`): `featureColumnIndices[groupIdx]` is
   the literal identity `g` for all g (see `csv_train.cpp:900-904`:
   `colIndices[g] = g`). The kernel dereferences a device memory array to
   read back `groupIdx` that was already known. Minor. Replace with deriving
   it from `tgX / maxBlocksPerPart` only — already done at line 91.
5. **Uniformly-initialised `privHist`** (`line 126-128`): 1024 stores to
   spilled device memory to zero an array that's about to be written by a
   subset of threads. This is unavoidable given the current design but
   disappears under a SIMD-group-local reduction where each SIMD group uses
   shared memory directly.
6. **NO wasted work from `maxBlocksPerPart`** — the Sprint 15 fix at
   `csv_train.cpp:891-894` correctly scales blocks with partition size.
   Baseline data confirms this; historical note in `bottlenecks.md` B2 is
   outdated.

---

## C. Explaining the stage-profile shape

### C.1 Why histogram time grows 45 -> 114 ms (depth 0 -> 5)

From the baseline per-depth table:

| depth | partitions | threadgroups | hist_ms |
|-------|------------|--------------|---------|
| 0     | 1          | 78 (3 blocks/part) | 45.48 |
| 1     | 2          | ~104 (2 blocks/part, approx) | 23.55 |
| 2     | 4          | ~208 | 23.64 |
| 3     | 8          | 208 (1 block/part × 8 × 13 × 2) | 38.55 |
| 4     | 16         | 416 | 74.16 |
| 5     | 32         | 832 | 114.42 |

Three effects compose:

1. **Serial reduction cost grows linearly with #threadgroups** — each
   threadgroup carries its own 255-step reduction, and the GPU can only
   hide a finite number in flight. From depth 3 to 5 the threadgroup count
   quadruples; hist time roughly triples. Consistent with reduction-bound.
2. **Document count is constant** (10k total docs) but spread across more
   threadgroups (smaller partitions). The per-thread doc loop shrinks, but
   the fixed-cost reduction tail does not — giving an increasingly poor
   ratio of useful work to overhead as depth grows.
3. **Scattered reads** (B.1) — `docIndices` at higher depths creates more
   random memory traffic since docs within a single partition are
   non-contiguous in the original compressedIndex layout. Minor compared
   to (1).

The 45 ms at depth 0 is anomalously high (more than depths 1 and 2
combined). This is most likely **kernel compile / first-launch warm-up**
amortised across the 20-iteration loop. The profiler code does not skip
iteration 0 in its accumulation. Recommend re-running with
`--iterations 25` and skipping the first 5 iters to confirm.

### C.2 Why `mx::eval(histogram)` at csv_train.cpp:953 is there

Tracing the call chain:

1. `DispatchHistogram()` returns an `mx::array`.
2. In csv_train.cpp:3197, the return value is pushed into `histArrays[k]`.
3. At csv_train.cpp:3246, Phase 2 calls `mx::eval(toEval)` over the
   full vector including each `histArrays[k]`.

So yes, the inner `mx::eval(histogram)` at line 953 is **redundant for
the csv_train caller** — the outer `mx::eval(toEval)` would materialise
these arrays anyway in exactly one batched eval.

**Why it's there.** The `DispatchHistogram` function was originally written
as a generic utility returning a materialised histogram; the Phase-2
batched eval was added later (Sprint 10+). The inner eval was never
removed. It breaks the lazy graph between per-dim dispatches: for
multiclass (approxDim=3), the three histograms are evaluated serially
(Phase 1 iteration k blocks on iteration k-1's kernel). Removing it would
allow MLX to submit all three command buffers concurrently to the Metal
driver; the driver then schedules them with any parallelism the
dependencies allow.

**Expected effect of removal.** For multiclass (approxDim=3), this could
save up to 2× the kernel-encoding overhead per dispatch — modest. For
binary/RMSE (approxDim=1) there's only one dispatch per depth so removal
is a no-op for latency; it just cleans the code. **Not the headline
lever.** Mention it as a free-win cleanup in Sprint 17.

### C.3 Why bin count has near-zero effect

Bin count affects only the **global-memory writeback** phase
(`kernel_sources.h:186-200`). At 32 bins vs 128 bins:

- Threadgroup histogram size: 1024 floats either way (sized for the worst
  case, 256-bin one-byte features).
- Per-doc inner loop: 4 iterations either way.
- Reduction cost: 255 × 1024 float adds either way.
- Writeback cost: for 32 bins, only 32 out of 256 slots per feature are
  written; for 128 bins, 128. This is a few KB of global memory traffic
  per threadgroup — trivial compared to the reduction tail.

So the kernel is **bound on the per-thread private-memory reduction**,
not on the bins. The observed 1-2% difference between 32 and 128 bins
reflects only the extra writeback and atomic contention at higher bins.

---

## D. Ranked Sprint 17 levers

| # | Lever | Where | Expected gain | Risk | Depends on |
|---|-------|-------|---------------|------|------------|
| **1** | **Replace serial 255-step reduction with tree reduction** | `kernel_sources.h:161-181` | **30-60% on histogram_ms** | Low (same algorithm family as kSuffixSumSource) | nothing |
| 2 | Replace `privHist[1024]` with SIMD-group-sliced threadgroup memory | `kernel_sources.h:115-148` | 20-40% additional (composes with #1) | Medium — register pressure trade-off, needs profiler re-validation | #1 |
| 3 | Drop per-doc `docIndices[sortedPos]` gather; pre-permute stats/compressed | `kernel_sources.h:132-133` | 10-20% on depth >= 3 | Medium — requires layout change upstream | nothing |
| 4 | Remove redundant `mx::eval(histogram)` at `csv_train.cpp:953` | `csv_train.cpp:953` | 0-5% (multiclass only) | Low | nothing — pure cleanup |
| 5 | Fuse per-dim multiclass dispatches into Z-grid dim | `csv_train.cpp:3190-3209`, kernel grid | 30-50% on multiclass only | High — layout change, blast radius | #1, #4 |
| 6 | Drop per-doc `foldCountsFlat[foldBase+f]` range check; enforce bins packed densely | `kernel_sources.h:144` | 2-5% | Low | nothing |

### D.1 — Serial reduction -> tree reduction (HEADLINE)

**Justification.** Section B.3 shows the reduction contains 255 sequential
barriers each followed by a 1024-entry read-modify-write where **99.6% of
the threads are idle**. A standard butterfly/Hillis-Steele tree reduction
over threadgroup memory does the same total work in ~8 log-steps with
**all 256 threads active at every step**. The `kSuffixSumSource` kernel
in the same file already demonstrates the pattern (lines 297-302) — this
is a small, well-understood change. Determinism is preserved because the
butterfly schedule is fixed at compile time. Baseline evidence: histogram
is 97.7% of iter time at RMSE 10k (310 ms out of 318 ms). Cutting the
reduction by 2-3× would drop iter time to the 200-230 ms range, landing
us at a 30-40% end-to-end speedup.

**Expected gain range.** 30-60% on histogram_ms. Conservative 30% if
reduction is only half the kernel's wall time; optimistic 60% if (as the
private-memory analysis in B.2 suggests) reduction dominates because
privHist spills.

### D.2 — Per-thread private histogram -> SIMD-group-sliced shared histogram

**Justification.** The current design is "1 private histogram per thread,
serial-reduce at end" (256 privates, 1 final). The original `hist.metal`
design was "1 private histogram per SIMD group of 32 threads, SIMD-reduce
at end" (8 privates, 1 final) — see `hist.metal:89-95`. The SIMD-sliced
design uses 8× less private memory and has 1/8 as many things to reduce,
at the cost of needing atomics **inside a SIMD group** (which is a
32-thread unit and on Apple Silicon has well-defined convergent
behaviour). The original hit BUG-001 because cross-SIMD writes on shared
memory were not actually SIMD-local. A modern rewrite would keep the
histogram in threadgroup memory partitioned per-SIMD (no cross-group
contention) and rely on `simd_sum` / `simdgroup_barrier` primitives for
the intra-SIMD fold. Requires a careful BUG-001 parity test. Defer if #1
lands the headline number.

### D.3 — Pre-permute stats & compressedData to remove `docIndices` gather

**Justification.** `docIndices[sortedPos]` (line 133) forces every per-doc
load of stats and compressedData to be a gather. A preprocessing pass
(on-GPU scatter before histogram dispatch) writes stats and packed
features in the permuted order. The per-doc reads then become
coalesced. Cost: one extra kernel dispatch and ~2× memory for
packed+stats at depth > 0. Gain grows with depth (where the permutation
diverges most from identity). Secondary after (1) and (2).

### D.4 — Remove redundant `mx::eval(histogram)` at csv_train.cpp:953

**Justification.** C.2 established this eval is redundant. It's a
one-line removal in a non-critical path. The gain is small (only helps
multiclass) but the cost is zero. Include as a drive-by cleanup in the
same PR.

### D.5 — Fuse per-dim multiclass dispatches

**Justification.** Baseline shows multiclass is 2× binary (approxDim=3
serialises three histograms). Adding an approxDim axis to the Z grid
would let one dispatch cover all three. But this is a layout change that
touches the stats array, the leaf accumulator, the scorer — blast
radius is wide, and the gain is bounded by 2× on multiclass only.
Defer to Sprint 18 unless Sprint 17 finishes #1+#4 early.

### D.6 — Drop per-doc fold range check

**Justification.** Line 144 `if (bin < foldCountsFlat[foldBase + f] + 1)`
is a warp-predicated branch on every doc. If the quantiser guarantees
bins are dense in `[0, folds)` per feature, this check is unnecessary.
Verify the guarantee holds and delete the predicate. Minor.

---

## E. Sprint 17 plan proposal

**Headline lever:** D1 — replace the serial 255-step reduction in
`kHistOneByteSource` with a log-step tree reduction over threadgroup
memory. Include D4 (remove redundant `mx::eval`) as a drive-by cleanup.

### Acceptance criterion

- `histogram_ms` at (N=10000, RMSE, depth=6, 128 bins) drops from 310 ms
  (baseline) to **≤ 200 ms** (35% improvement), measured by the same
  stage profiler on the same `bench_10000_50f.csv` fixture.
- `iter_total_ms` at the same config drops from 318 ms to ≤ 210 ms.
- Multiclass at (N=10000, depth=6, 128 bins) drops from 596 ms to
  ≤ 380 ms (also 35% improvement; 3× histogram calls scale linearly
  with the kernel win).

### Measurement protocol

- Run the full 18-config baseline sweep (matching baseline_results.md
  exactly) with the new kernel. Store JSONs under
  `.cache/profiling/sprint17/post_optD1_<N>_<loss>_d6_<bins>bins.json`.
- Compute per-config delta vs. Sprint 16 baseline. Sprint 17 success =
  **stage-4 (histogram) drops by ≥ 30% in all 18 configs**.
- Re-capture Metal System Trace for (10k, RMSE, depth 6, 128) and the
  multiclass equivalent; open in Instruments to verify the reduction
  phase disappeared from the kernel timeline.
- No acceptance on suffix_scoring_ms, leaf_values_ms, etc. — those are
  all <1% and under noise.

### Risks and mitigations

- **Determinism regression (BUG-001 re-emerges).** The current serial
  reduction was written specifically to fix a non-determinism bug
  (`kernel_sources.h:57-82`). A butterfly reduction over threadgroup
  memory is itself deterministic (fixed stride schedule), but the
  mitigation is a pre-merge numerical test: run the kernel 10 times on
  fixed input and assert bit-exact histogram output across all runs.
  Add this test to the existing BUG-001 regression suite.
- **Threadgroup memory exceeded.** Current design uses 4 KB threadgroup
  memory (`stagingHist`). The butterfly reduction needs 4 KB for its
  working buffer — same size, just different usage. Apple M-series GPUs
  allow 32 KB threadgroup memory. No risk.
- **Private-memory pressure unchanged.** `privHist[1024]` still exists
  in this lever; spill behaviour is unchanged. D1 does not fix B.2's
  spill problem — it only accelerates the reduction tail. If D1
  results fall short of 30%, escalate to D2 (SIMD-group-local shared
  histogram) in the same sprint.

### Numerical parity gate

- **Bit-exact** histogram output for all 18 baseline configs, validated
  by a new test that snapshots baseline histograms and diffs byte-by-byte
  with the new kernel output.
- **End-to-end RMSE / logloss / multiclass-ll** parity: within 0 ulp of
  Sprint 16 baseline on the final model loss. No tolerance; the
  reduction is still exact-sum.

### What we will NOT do in Sprint 17

- D2 (SIMD-group-local shared histogram). Deferred to Sprint 18 unless D1
  underperforms. Needs its own BUG-001 parity campaign.
- D3 (pre-permute stats/compressedData). Layout change; Sprint 18.
- D5 (multiclass fusion). Wide blast radius; Sprint 18+.
- D6 (drop range check). Tiny win; save for a cleanup sprint.
- Library-path histogram (`methods/histogram.cpp`). Still has
  `maxBlocksPerPart = 1` and per-group serial loop, but it is dead code
  for csv_train. A separate cleanup issue should either remove it or
  port the csv_train kernel back into it.

---

## Divergence between production and library histogram paths

`catboost/mlx/methods/histogram.cpp:105` still hardcodes
`maxBlocksPerPart = 1` and still has the per-feature-group serial loop at
`histogram.cpp:112-155`. This path is **dead code** for csv_train
(and by extension the Python bindings / CLI). The production kernel lives
entirely inside `csv_train.cpp:869-955` with the source string at
`kernel_sources.h:85-201`. Any Sprint 17 kernel change must be made in
`kernel_sources.h`, NOT `histogram.cpp`. A follow-up cleanup issue
should either delete the library path or resync it with the production
kernel to avoid this two-source-of-truth situation biting a future
contributor.

---

## Reproducibility

- Stage profiler data: `.cache/profiling/sprint16/baseline_*.json` (18 configs).
- Metal System Trace: `.cache/profiling/sprint16/mst_10000_rmse_2026-04-17.trace`
  (4.2 GB; requires Instruments.app open-once before `xctrace export` works).
- Binary: `csv_train_profiled` (built 2026-04-16, `-O2 -DCATBOOST_MLX_STAGE_PROFILE`).
- Data: `/tmp/bench_10000_50f.csv` (50 features + target column).
- Environment: macOS 25.3, Apple Silicon, Xcode 16.0 / xctrace 16.0,
  MLX from `/opt/homebrew/Caskroom/miniconda/base/lib/python3.13/site-packages/mlx`.
