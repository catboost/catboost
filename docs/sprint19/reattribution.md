# Sprint 19 — S19-01c: Micro-Benchmark Re-Attribution

**Config:** N=50k, RMSE, depth=6, 128 bins (gate config)
**Date:** 2026-04-19
**Reported by:** @performance-engineer (S19-01c)
**Branch tip:** `108c7a59d2` (DEC-015 WIP state, col-major in `kernel_sources.h`)
**Depends on:** S19-01b (`docs/sprint19/attribution_accumulation.md`) — model being tested
**Harness:** `docs/sprint19/scratch/microbench/microbench_gather.cpp`
**Methodology:** 5 warm runs + 5 timed runs, wall-clock via `std::chrono::steady_clock`,
`mx::eval()` blocks until GPU completion. Not MTLCommandBuffer timestamps —
see critique §6. Results reproducible:

```
clang++ -std=c++17 -O2 \
  -I/opt/homebrew/Cellar/mlx/0.31.1/include \
  -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx \
  -framework Metal -framework Foundation \
  docs/sprint19/scratch/microbench/microbench_gather.cpp \
  -o /tmp/microbench_gather && /tmp/microbench_gather
```

---

## 1. Micro-Benchmark: Row-Major vs Column-Major Gather in Isolation

Minimal kernel: `compressedIndex[docIdx * lineSize + featureColumnIdx]` (row-major) vs
`compressedIndex[featureColumnIdx * totalNumDocs + docIdx]` (col-major). No accumulation,
no shuffle, no threadgroup memory. One read per thread, result written to output buffer.
N=50,000 sorted docIndices, lineSize=25.

| Kernel | mean (ms) | stdev (ms) | runs (ms) |
|---|---:|---:|---|
| gather_rowmajor | 0.160 | 0.039 | [0.223, 0.171, 0.132, 0.135, 0.140] |
| gather_colmajor | 0.138 | 0.039 | [0.205, 0.137, 0.127, 0.113, 0.109] |

**Row-major / col-major speedup: 1.16x** (stdev overlap: means are within ~0.6 standard
deviations of each other).

**Interpretation:** The 1.16x ratio has overlapping error bars. At face value, col-major
is marginally faster in isolation, but the effect is within noise (stdev = ±0.039 ms on
means separated by 0.022 ms). This is NOT the 25x cache-line reduction the S19-01b model
predicted would manifest as a large speedup. AGX is not exposing the per-cache-line
gather cost as an isolated bottleneck in a pure-gather kernel. The model was wrong.

---

## 2. Verdict on DEC-015

**KILL DEC-015.**

Evidence:
1. Gather-in-isolation speedup: 1.16x (within noise). The predicted mechanism — 25 cache
   lines per 32-doc batch reduced to 1 — does not produce a measurable speedup even
   when the gather is the ONLY operation the kernel performs.
2. Full kernel measured: 0.98x (regression). DEC-015 is slightly slower end-to-end,
   consistent with the additional host-side transposition overhead and increased buffer
   footprint.
3. Probe D (global load stripped) measured: 2.404 ms vs production 2.357 ms — i.e.
   removing all global loads makes the kernel 2% SLOWER. This is decisive: global loads
   are not on the critical path AT ALL. The kernel runs essentially the same speed
   with or without memory traffic. The bottleneck is not memory.

**Error in the S19-01b model:** The model assumed AGX would expose gather latency as a
4-round L2-stall chain (469 ns per batch). The probe evidence shows the kernel is
ALU-bound and shuffle-bound, not memory-latency-bound. The AGX L2/prefetcher fully
hides the gather latency behind the shuffle inner loop. The model's classification of
the kernel as "memory-latency-bound, not compute-bound" (attribution_accumulation.md §3)
was inverted: the kernel is actually **shuffle+TG-write-bound**, and global loads are
completely hidden.

**DEC-015 disposition:** Reject. Revert `catboost/mlx/kernels/kernel_sources.h` to
pre-DEC-015 row-major address expression. The `histogram.cpp` variable-name side-fix
(featureColumnIdx / featureColumnIndices bug) should be committed separately — it is
correct independently of DEC-015.

---

## 3. Probe Results: Accumulation Sub-Phase Attribution

All probes run at production L1a kernel structure (col-major at tip), 1 TG × 256 threads,
all N=50,000 docs in a single partition (depth-0 equivalent — 196 outer-batch iterations).

| Probe | mean (ms) | stdev (ms) | delta vs prod | % of production |
|---|---:|---:|---:|---:|
| probe_production (baseline) | 2.357 | 0.035 | — | 100% |
| probe_A: no simd_shuffle | 0.325 | 0.030 | +2.032 ms | **86.2%** |
| probe_B: no TG write | 0.934 | 0.046 | +1.424 ms | **60.4%** |
| probe_C: no bin-check | 1.951 | 0.029 | +0.407 ms | 17.2% |
| probe_D: no global load | 2.404 | 0.025 | −0.047 ms | −2.0% |

**Sub-phase cost estimates (delta method):**

| Sub-phase | Cost estimate (ms) | % of accumulation |
|---|---:|---:|
| simd_shuffle inner loop | 2.032 | 86.2% |
| TG-memory write (+= simdHist) | 1.424 | 60.4% |
| Branch divergence (bin-check) | 0.407 | 17.2% |
| Global loads (compressedIndex + stats + docIndices) | −0.047 | −2.0% (noise) |
| Probe D floor (ALU + shuffle, no memory) | 2.404 | 102% |

**Critical finding — Probe D paradox:** Probe D (no global loads) is 2% SLOWER than
production. This confirms AGX fully hides global load latency: the loads are overlapped
with the shuffle inner loop and add zero measurable wall time. The ALU floor (D) is
≥ production time — the kernel is entirely compute-bound and cannot be sped up by
improving memory access patterns.

**Important caveat on delta interpretation:** The probes are not additive because they
share the same ALU pipeline. The shuffle cost (A), TG-write cost (B), and branch-check
cost (C) overlap in execution. The correct reading is:
- The DOMINANT cost driver is `simd_shuffle`: removing it drops the kernel 86.2%.
- TG-memory writes are significant (60.4%) but they execute DURING the shuffle stall,
  so their independent cost is lower than the delta suggests.
- Global loads: 0% independent cost. Fully pipelined.

---

## 4. New Dominant Sub-Phase

**Dominant sub-phase: simd_shuffle inner loop — 2.032 ms ± ~0.05 ms = 86% of single-TG
accumulation time.**

The production kernel's inner loop executes 32 `simd_shuffle` calls per doc per outer
batch step (3 shuffles × 32 src iterations = 96 shuffles per 32-doc batch). At 196
outer batches per TG at depth-0 (N=50k, 8 SIMD groups): 196 × 96 = 18,816 shuffles per
SIMD group per iteration.

The `simd_shuffle` instruction on AGX executes in ~2–4 cycles within the SIMD group.
But the INNER loop has a structural serial dependency chain: each shuffle in the src
loop depends on the PREVIOUS src's work completing (the TG-write at the end of the
per-src iteration uses the shuffled value, and the TG-memory bank write may create a
RAW hazard with the next src iteration reading simdHist on the next outer-batch pass).
The 32-iter inner shuffle loop is fundamentally serial: you cannot issue src+1 until
src is complete because the bin-owner lane's write to simdHist is a write-after-read
hazard with a future shuffle of the same address.

Additionally, `simd_shuffle` latency on AGX may be higher than assumed. The S19-01b
model treated shuffle as "hidden in CI stall shadow" — the probes show shuffle is
instead the DOMINANT cost, not a hidden one.

**Scaled to full iteration (14.30 ms accumulation from S19-01, which covers all 25
feature groups across all 6 depths with their respective TG counts):** the per-TG
probe is representative. The 14.30 ms is `sum(depths) × numGroups × per-TG-cost`. The
shuffle dominance (86%) scales linearly: **~12.3 ms of the 14.30 ms accumulation budget
is the simd_shuffle inner loop.**

---

## 5. Ranked New Lever Priors

The actual dominant cost is `simd_shuffle` (86% of accumulation), not global-load
latency. This inverts the lever ranking from S19-01b.

### 5.1 Is the 1.5x e2e gate still reachable?

Current baseline (gate config): accumulation = 14.30 ms, histogram_ms = 15.43 ms,
iter_total = 21.03 ms (S19-01 values; probe data is consistent).

For 1.5x e2e gate: iter_total target ≤ 14.02 ms → histogram_ms ≤ 8.54 ms →
accumulation ≤ 8.10 ms. Required reduction in accumulation: 14.30 → 8.10 = 6.2 ms.
As a fraction of the shuffle-dominated 12.3 ms shuffle budget: need 50% shuffle
reduction.

### 5.2 Lever ranking by mechanism

| Rank | Lever | Mechanism | Estimated saving | Feasibility |
|---:|---|---|---:|---|
| 1 | **Reduce shuffle iteration count** | Cut inner `for src 0..31` loop via register prefetch + partial unroll — process 2 docs per shuffle pass, halving the 32-iter inner loop to 16 iters × 2-doc slab | ~6.0 ms (50%) | Medium — requires register doubling (A1 variant) |
| 2 | **Eliminate bin-check branch per src** | Precompute ownership masks; replace per-src bin-check with a lookup or 32-bit popcount trick | ~0.4 ms (3%) | High (small win) |
| 3 | **Reduce simd_shuffle to 2 shuffles/src** | Currently 3 shuffles (packed, stat, valid). Fuse valid into packed (pack valid bit into MSB of packed); reduce to 2 shuffles/src | ~0.7 ms (6%) | High — clean change |
| 4 | **DEC-014 A1 (wider batch 64 docs)** | Amortizes outer-loop overhead and halves the outer batch count (196 → 98 iters); also halves the absolute number of outer-loop boundary checks | ~0.75 ms (5%) | High — analyzed in ablation_accumulation.md |
| 5 | **Column-major layout (DEC-015)** | REJECTED — shown neutral by probes | 0 ms | N/A |

**Primary recommendation: Lever 1 (shuffle count reduction) + Lever 3 (fuse valid into
packed) as a combined change.** Together they target 50% + 6% = ~56% of the shuffle
budget, or ~6.9 ms savings from accumulation. This would bring accumulation to
14.30 − 6.9 = 7.4 ms → histogram_ms ≈ 8.5 ms → iter_total ≈ 14.1 ms → e2e ≈ 1.49x.
That is marginally below 1.5x. Combined with Lever 4 (DEC-014 A1, +0.75 ms): total
saving ≈ 7.65 ms → accumulation 6.65 ms → histogram_ms 7.75 ms → iter_total 13.35 ms
→ **e2e ≈ 1.57x. Clears R8.**

### 5.3 Mechanism for Lever 1 (shuffle count reduction — new DEC recommendation)

Current: inner loop `for src in 0..31` → 32 iterations, each with 3 shuffles.
Proposed: process 2 docs per inner-loop iteration. Lane `l` holds TWO docs
(packed_lo, stat_lo from `d = batch_start + lane` and packed_hi, stat_hi from
`d = batch_start + 32 + lane`). Inner loop: `for src in 0..31` → each iteration
broadcasts lane `src`'s packed_lo and packed_hi in sequence. Total shuffles per
32-doc batch: 6 (3 per slab × 2 slabs per src) × 32 src = 192 vs current 96 × 32 = 96.

Wait — that DOUBLES shuffles. The correct formulation: with 2 docs per lane, the OUTER
loop stride doubles (NUM_SIMD_GROUPS × 64 instead of × 32), so there are HALF as many
outer-loop iterations. Total shuffles = (N/64) outer × 32 src × 6 shuffles = same as
current. But the SERIAL DEPTH of the inner loop halves: instead of a 32-iteration serial
chain (src 0..31), you have a 32-iteration inner loop with 2 ops per iteration = same
serial depth. The benefit is amortized outer-loop overhead and reduced loop-control
overhead, matching DEC-014 A1 mechanism. The shuffle serial chain length is unchanged.

**Revised conclusion for Lever 1:** Wider batch (A1) does NOT reduce shuffle serial
depth. It cannot halve the dominant shuffle cost. Maximum realistic saving from A1 alone
is ~5% (DEC-014 projection confirmed). The shuffle serial chain is the fundamental limit.

**The REAL lever for shuffle reduction:** Reduce the number of `simd_shuffle` calls per
src iteration. Currently 3: `simd_shuffle(packed, src)`, `simd_shuffle(stat, src)`,
`simd_shuffle(valid, src)`. If `valid` is fused into the high bit of `packed` (valid=1
→ packed |= 0x80000000; invalid → packed = 0), then `simd_shuffle(valid, src)` is
eliminated — 3 → 2 shuffles per src. Saving: 1/3 of shuffle cost = 2.032/3 ≈ 0.68 ms.

This is Lever 3 (fuse valid into packed). Combined with probe B (TG-write reduction):
the write cost (1.424 ms probe-delta) suggests the TG write is also on the critical path.
But probe B removes writes by replacing with a register accumulator — it doesn't eliminate
the work, it eliminates the write contention. The real saving from reducing writes would
be if each src write to simdHist blocks the NEXT src's read in the following outer batch.
This suggests Lever 2 (reduce bin-check) + Lever 3 (fuse valid) are the cleanest wins.

### 5.4 Realistic path to R8

With Probe D (global load stripped) running at 2.404 ms — virtually equal to production
(2.357 ms) — the ALU+shuffle floor is 2.357 ms. This is the irreducible minimum for
the current algorithm structure. The headroom above the floor is 0 ms for the single-TG
probe.

Scaling to full iteration: the ALU floor across all TGs is ~2.357 ms × (1575 TGs / 196
batches at depth-0) — but this is not a direct scaling because different depths have
different doc counts per TG. The per-iteration sum is the 14.30 ms measured value.
If we could achieve the Probe D floor for every TG across all depths, histogram_ms
would drop to approximately `14.30 × (2.357/2.357) × (Probe_D/probe_prod_single_TG)`
= 14.30 × (2.404/2.357) = 14.57 ms — i.e. WORSE. This confirms there is no algorithmic
room to improve by attacking global memory.

**The achievable gain is in reducing the shuffle inner loop depth.** Lever 3 (fuse
valid, 2 → shuffles/src) + DEC-014 A1 (halved outer loop at constant per-iteration
shuffle count) are the two levers with solid backing. Combined ceiling: ~0.68 ms
(shuffle) + ~0.75 ms (A1) = 1.43 ms saving → accumulation 12.87 ms → histogram_ms
13.96 ms → iter_total ≈ 19.56 ms → e2e ≈ **1.08x**. Does not clear R8.

**R8 (1.5x e2e) is NOT reachable by incremental changes to the current L1a kernel
structure.** The shuffle serial depth (32 src iterations × 2–3 shuffles each) is
the fundamental bottleneck, and it is coupled to the algorithm's broadcast structure.
To break the 1.5x gate requires either:
- Restructuring the histogram algorithm to avoid per-doc SIMD broadcast (e.g. sorting
  docs by bin value before accumulation — eliminates shuffle entirely), or
- Accepting that Sprint 19 closes at <1.2x and revising R8 downward.

---

## 6. Benchmark Methodology Notes

- **Timing method:** `std::chrono::steady_clock` wall-clock with `mx::eval()` blocking.
  This includes Metal command-buffer scheduling overhead (~20–200 µs at N=50k).
  For kernels running 0.1–2.5 ms, this is 1–10% relative noise — acceptable for
  binary verdicts but not for precision attribution of sub-ms differences.
- **Gather stdev (±0.039 ms):** High relative to the mean difference (0.022 ms between
  layouts). The layout comparison is within noise. Repeat runs could tighten this with
  longer kernel execution (larger N or multi-column gather), but the 1.16x vs the
  predicted 25x is a decisive falsification regardless.
- **Probe grid (1 TG vs production 1575 TGs):** Single-TG probe isolates accumulation
  without partition / depth structure overhead. The per-TG cost is representative for
  depth-0 (all N docs in one partition). Deeper levels have fewer docs per TG and
  proportionally less work — the dominant cost is at depth 0, which is what we measured.
- **Col-major address in probe kernels:** Probes run at branch tip `108c7a59d2` which
  has DEC-015 col-major in `kernel_sources.h`. Probe D (no global loads) confirms this
  is irrelevant — the result would be identical with row-major.

---

## 7. Recommendation to @ml-product-owner

**Recommended path: Path 3 — demote S19 to cleanup-only + revise R8.**

Rationale:

1. **DEC-015 (Option E, col-major CI) — KILL.** The micro-benchmark confirms AGX hides
   global load latency entirely. Layout change provides 1.16x in isolation (within noise)
   and 0.98x end-to-end. The model's root-cause diagnosis was wrong.

2. **DEC-014 A1 (wider batch) — KEEP as a small win.** Projected ~1.04x e2e (S19-01b).
   This remains valid. The mechanism (amortized outer-loop overhead) is real but small.
   Commit as a correctness-clean, low-risk improvement.

3. **R8 (1.5x e2e on 50k/RMSE/d6/128b) is NOT achievable in Sprint 19.** The dominant
   sub-phase is `simd_shuffle` serial depth (86% of accumulation, 12.3 ms). Breaking
   this requires algorithmic restructuring (sort-by-bin pre-pass, or a different
   histogram algorithm), which is Sprint 20+ scope.

4. **Sprint 19 cleanup deliverables:**
   - Commit the `histogram.cpp` variable-name side-fix (featureColumnIdx vs
     featureColumnIndices) — correct regardless of DEC-015.
   - Commit DEC-014 A1 (wider batch, expected ~1.04x) if implementation is clean.
   - Revert DEC-015 kernel address back to row-major.
   - Update R8 gate to ≤1.1x for Sprint 19; schedule algorithm restructuring as
     Sprint 20 S20-01 (sort-by-bin or warp-cooperative histogram).

5. **Sprint 20 lever (preliminary):** Eliminate or shorten the `simd_shuffle` inner
   loop. Options: (a) pre-sort docs by bin value before dispatch — reduces shuffle
   to 0 (each lane already owns its bins); (b) use Metal's `simdgroup_matrix` ops for
   cooperative histogram builds; (c) tiled shared-memory histogram where a single
   TG-memory read replaces a shuffle (requires DEC-011 ceiling renegotiation but the
   variant_b2 analysis in accum_variants.metal shows it fits if per-SIMD hist is halved).

---

## 8. Lineage

- S19-01b (`docs/sprint19/attribution_accumulation.md`) — model being tested; falsified
  by this document. The classification "memory-latency-bound" is inverted by probe D.
- `docs/sprint19/scratch/microbench/microbench_gather.cpp` — benchmark source (not
  committed to kernel_sources.h; scratch only per task spec).
- `docs/sprint19/scratch/microbench/gather_rowmajor.metal` — gather kernel design doc.
- `docs/sprint19/scratch/microbench/gather_colmajor.metal` — gather kernel design doc.
- `.claude/agent-memory/ml-engineer/project_sprint19_dec015_blocker.md` — DEC-015 state
  at time of this analysis.
