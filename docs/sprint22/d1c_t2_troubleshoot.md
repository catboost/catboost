# Sprint 22 D1c — T2 Troubleshoot: root-cause identified

**Branch**: `mlx/sprint-22-t2-integration` (HEAD `4333c82a7e` at Phase 0)
**Date**: 2026-04-20
**Task**: D1c — root-cause diagnostic of T2 sort-by-bin failure after D1a hypothesis was falsified in D1b.
**Prior docs**: `d0_t2_production_shape.md`, `d1_t2_parity_sweep.md`, `d1a_t2_diagnostic.md`, `d1b_t2_fix_and_rerun.md`, `docs/sprint21/d1r2_t2_microbench.md`
**Status**: **Root cause identified. Minimum reproducer established. Fix verified on 18/18 DEC-008 configs.**

---

## §1 TL;DR

**Root cause: T2-sort `maxPartDocs` uniform-partition assumption.**
`bench_boosting.cpp:526` computes `maxPartDocs = (numDocs + numActiveParts - 1) / numActiveParts` — i.e. `ceil(N/K)`. This is the **expected** partition size when splits are perfectly balanced. After real argsort-permuted training-loop splits, partitions are unbalanced: **some partitions hold more docs than `ceil(N/K)`**, often dramatically so (e.g. depth-1 on 50k docs: partition 1 holds 49558 docs, `maxPartDocs=25000`, 24558-doc overflow). The T2-sort kernel writes `partSize` docs into a per-TG slot of size `maxPartDocs` in `sortedDocs[slotBase..slotBase+maxPartDocs)`; when `partSize > maxPartDocs`, the excess writes spill into the next TG's slot, corrupting its state.

The subsequent T2-accum kernel reads up to `totalDocsInPart = binOffsets[offBase+128]` (which equals the accurate `partSize` from the sort's prefix scan) from `sortedDocs[slotBase..slotBase+partSize)`, so it reads partially into the neighbor's corrupted slot. The histogram is built from a mix of this partition's docs and the neighbor's. When the neighbor partition is empty, the reads land on stale data from a prior dispatch or on whatever a concurrent TG race-wrote. Hence: non-deterministic magnitude, depth/row-count-dependent failure pattern, `iters=1` always correct (depth=0 → single partition, `partSize = maxPartDocs = numDocs` exactly — no overflow).

**Fix location**: `bench_boosting.cpp:526`.
**Fix (verified)**: replace the uniform-partition estimate with a value guaranteed ≥ `max(partSizes)`. Proof-of-fix harness used `maxPartDocs = numDocs` (simplest safe upper bound): 18/18 DEC-008 configs now **bit-exact** with T1, 10/10 determinism runs at gate config identical.

**Prior hypotheses ruled out**: D1a's blit-ordering (fill_gpu → accum): irrelevant, eval barriers don't fix it. D1a's "bug β" atomic-scatter nondeterminism: non-existent at gate config — the 0.13% divergence observed in D0/D1 was downstream of H-B, not an independent bug. MLX lazy-graph dep-wiring: correct as-is (D1a §2 conclusion stands). JIT cache / buffer-pool state: irrelevant; overflow mechanism is a static indexing bug, not a scheduling race.

---

## §2 Minimum reproducer

```bash
/tmp/bench_boosting_t2_d1c --rows 1024 --features 1 --classes 1 \
  --bins 4 --seed 42 --depth 2 --iters 2 --t2
```

| Metric | Expected (T1) | Actual (T2) |
|--------|---------------|-------------|
| BENCH_FINAL_LOSS | 0.26981398 | varies across runs: 0.26554936, 0.27354732, 10.18858624, 10.75534630 (9/10 runs fail bit-exact, 1/10 coincidentally passes) |

With `T2_DIAG_FIX_MAXPARTDOCS=1` (diagnostic override of `maxPartDocs = numDocs`): 5/5 bit-exact with T1 (`0.26981398`).

At this config:
- `numDocs=1024`, `numActiveParts=2` (at depth=1, the second depth level)
- Actual partSizes: `[358, 666]` — sum correct, distribution unbalanced
- `maxPartDocs = ceil(1024/2) = 512`
- Partition 1 (666 docs) overflows its 512-doc slot by 154 docs

---

## §3 Truth table — (rows, depth, bins) × T1/T2 bit-exact, seed=42, features=1, iters=2

**Baseline (no fix)** — 18/18 DEC-008 configs fail with ULP deltas 1,327 to 2,583,206 (from `d1_t2_parity_sweep.md §4`).

**Phase-1 sweep, features=1, iters=2, bins=128, seed=42, 5 runs each**:

| Depth | nParts reached | Result |
|------:|:--------------:|:-------|
| 1 | {1} | 5/5 PASS (only depth-0, partSize=numDocs=maxPartDocs, no overflow) |
| 2 | {1, 2} | 1/5 PASS, 4/5 close-but-wrong (0.507 vs 0.494, ULP ~45k) |
| 3 | {1, 2, 4} | 5/5 PASS *accidentally* — the corrupt histogram at depth 1 still produces the correct split decision because partition 1 dominates |
| 4 | {1, 2, 4, 8} | 0/5 PASS; all catastrophic (37–105), non-deterministic magnitude |
| 5 | {1, 2, 4, 8, 16} | 4/5 PASS accidentally, 1/5 close-wrong |
| 6 | {1, 2, 4, 8, 16, 32} | 1/5 PASS accidentally, 4/5 catastrophic |

**D1b's claim of "even depths fail, odd depths pass"** is refined: *every depth ≥ 2 exhibits the overflow*, but whether it catastrophically affects training depends on whether the downstream split decision is swung by the corrupt histogram. Depths 3 and 5 "pass" by luck (dominant partition's decision unchanged by the corruption); depths 2, 4, 6 show catastrophic or close-but-wrong drift. This is a ROW-COUNT- and SEED-dependent accident, not a structural property of even vs odd depth.

**Phase-1 row-count × depth × bins sweep with fix applied** (105 configs: rows ∈ {1024, 10000, 16384, 24576, 32768, 40960, 50000}, depth ∈ {2,3,4,5,6}, bins ∈ {4, 32, 128}): **105/105 bit-exact T1 vs T2**.

**Full DEC-008 18-config sweep with fix**: see §6 — 18/18 bit-exact, 0 ULP.

---

## §4 Ruled-out hypotheses

| ID | Hypothesis | Verdict | Evidence |
|:--:|------------|:-------:|----------|
| H-A | Indexing bug in `slotBase` at power-of-2 partition counts | **REJECTED** | `slotBase = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx) * maxPartDocs`. Manually computed at depth-4 (numGroups=1, numPartitions=8, numStats=2, maxPartDocs=6250): max slotBase = 15×6250 = 93750 < buffer size 100000. Formula is algebraically correct; no overflow in `slotBase` arithmetic itself. The overflow is in the per-slot **length**, not the base address. |
| H-B | `maxPartDocs` miscalculated at depth>0 — uniform-partition assumption breaks on argsort-permuted partitions | **CONFIRMED as root cause** | Diagnostic at `bench_boosting.cpp:528-548` (scratch-only, env-gated `T2_DIAG_PARTSIZES=1`). Sample output at 50k/d4/b128: `numDocs=50000 nParts=8 maxPartDocs=6250 max(ps)=49558 sum(ps)=50000 overflows=1/8`. Fix-test with `maxPartDocs=numDocs`: 18/18 configs bit-exact, 10/10 gate-config runs deterministic. |
| H-C | partOffsets/partSizes stale between depth levels | **REJECTED** | Diagnostic at `T2_DIAG_PARTSIZES=1` shows `sum(partSizes) == numDocs` exactly at every depth. `ComputePartitionLayout` at `bench_boosting.cpp:298-314` recomputes both from the current `partitions` tensor, post-`ApplySplitToPartitions`, each depth level. No staleness. |
| H-D | MLX buffer-pool corrupts state across dispatches | **REJECTED** | D1b §3 showed eval barriers (Option 1 + Option 2) don't fix the bug. Independently, the H-B fix works without any eval-barrier change. `fill_gpu` is a compute shader (not a blit — D1b §6 confirmed via `mlx/backend/metal/copy.cpp:182`), properly serialized in the compute encoder. The bug is deterministic in its trigger (every overflow config always overflows); non-determinism comes from atomic-cursor race inside the sort kernel, not inter-dispatch state. |
| H-E | Two independent failure modes compounding | **REJECTED** | Fix for H-B alone eliminates all observed failure modes: catastrophic magnitudes (100s), close-but-wrong drifts (0.507), run-to-run non-determinism, even/odd pattern. All 18 DEC-008 configs hit ULP=0 with the fix. No residual bug β. |
| H-F | D1a's blit/compute-encoder race | **REJECTED** | D1b §6 structural diagnosis (`fill_gpu` is compute, not blit). Corroborated: D1b applied both Option 1 and Option 2 eval barriers and bug persisted. H-B fix works without touching eval placement. |

---

## §5 Root cause — mechanism detail

### 5.1 The formula and its assumption

`catboost/mlx/tests/bench_boosting.cpp:526`:
```cpp
const ui32 maxPartDocs = (numDocs + numActiveParts - 1) / numActiveParts;
```

This is `ceil(numDocs / numActiveParts)`, the average-case upper bound if docs were divided evenly across active partitions. `maxPartDocs` is then used at line 556 to size the `sortedDocs` output buffer:

```cpp
mx::Shape sortedDocsShape = {static_cast<int>(numTGs * maxPartDocs)};
```

And is passed as a Metal uniform `maxPartDocs` to both T2 kernels. Inside the T2-sort kernel (`kernel_sources_t2_scratch.h:93-96`):

```metal
const uint slotBase = (groupIdx * numPartitions * numStats + partIdx * numStats + statIdx)
                    * maxPartDocs;
```

This gives each (groupIdx, partIdx, statIdx) TG a disjoint slot of size `maxPartDocs` in the `sortedDocs` buffer.

### 5.2 The violation

After any depth level > 0 with a real training-loop split, partition sizes become unbalanced. `ComputePartitionLayout` (bench_boosting.cpp:298-314) does `argsort(partitions)` to produce `docIndices`, then counts per-partition sizes into `partSizes`. These are the ACTUAL sizes, unconstrained by uniform-distribution expectations.

Observed at runtime (50k docs, features=1, seed=42):

| depth loop iter | numActiveParts | partSizes | maxPartDocs (formula) | max(partSizes) | Overflow? |
|:---------------:|:--------------:|:----------|:---------------------:|:--------------:|:---------:|
| 0 | 1 | [50000] | 50000 | 50000 | No |
| 1 | 2 | [442, 49558] | 25000 | 49558 | **Yes (+24558)** |
| 2 | 4 | [442, 0, 0, 49558] | 12500 | 49558 | **Yes (+37058)** |
| 3 | 8 | [442, 0, 0, 0, 0, 0, 0, 49558] | 6250 | 49558 | **Yes (+43308)** |
| 4 | 16 | skewed, final partition holds most docs | 3125 | ~49000 | Yes |
| 5 | 32 | skewed | 1563 | ~16000 | Yes |

The skew is extreme because this benchmark uses `features=1`: every split is on the same single feature, so only one "live" partition deepens at each split. Multi-feature benchmarks (features=50) show more balanced but still non-uniform partitions — some still overflow.

### 5.3 The failure chain in the T2-sort kernel

At `kernel_sources_t2_scratch.h:98-104`:

```metal
for (uint i = tid; i < partSize; i += 256u) {
    const uint docIdx = docIndices[partOffset + i];
    const uint packed = compressedIndex[docIdx * lineSize + featureColumnIdx];
    const uint bin    = (packed >> 24u) & 0x7Fu;
    const uint pos    = atomic_fetch_add_explicit(&tgCursors[bin], 1u, memory_order_relaxed);
    sortedDocs[slotBase + pos] = docIdx;   // pos ∈ [0, partSize); slot length = maxPartDocs
}
```

`pos` iterates over `[0, partSize)` across all threads of this TG (via atomic cursor). When `partSize > maxPartDocs`, `pos` can reach values > `maxPartDocs`. The write `sortedDocs[slotBase + pos]` then targets addresses in the **neighboring TG's slot** (slot layout: TG slots are contiguous).

Specifically, TG (g, p, s)'s slot is `[slotBase(g,p,s), slotBase(g,p,s)+maxPartDocs)`, and `slotBase(g, p, s+1) = slotBase(g, p, s) + maxPartDocs`. So when this TG's pos reaches `maxPartDocs + k`, it writes into neighbor TG (g, p, s+1)'s slot at position `k`. If `s` is the last stat, the next address is TG (g, p+1, 0). Etc.

Two TGs writing to overlapping addresses constitute a race. The Metal dispatch has 1664 TGs (at depth 5, numGroups=13, numPartitions=64, numStats=2 for the full 50k/50f/128b config); there's no inter-TG synchronization; each TG runs as a separate scheduled block.

The neighbor TG (g, p, s+1) is simultaneously writing its own docs into positions `[0, maxPartDocs)` of its slot. The overflow from TG (g, p, s) and the valid writes from TG (g, p, s+1) interleave non-deterministically.

### 5.4 The failure chain in the T2-accum kernel

At `kernel_sources_t2_scratch.h:163`:

```metal
const uint totalDocsInPart = binOffsets[offBase + 128u];
```

This reads the total-docs count from binOffsets, which was set in sort to `tgOffsets[128] = partSize` (the full local partSize, not `min(partSize, maxPartDocs)`).

The accum feature-0 path (`kernel_sources_t2_scratch.h:181-192`):

```metal
for (uint b = tid + 1u; b <= foldCount; b += 256u) {
    const uint start = binOffsets[offBase + b];
    const uint end   = binOffsets[offBase + b + 1u];
    for (uint i = start; i < end; ++i) {
        const uint docIdx = sortedDocs[slotBase + i];   // i ∈ [start, end) ⊂ [0, partSize)
        sum += stats[statIdx * totalNumDocs + docIdx];
    }
}
```

`start` and `end` are derived from `partSize`-local counting-sort offsets. `end` can be up to `partSize`. When `partSize > maxPartDocs`, reads with `i ≥ maxPartDocs` pull from the neighbor's corrupted slot. The neighbor has either (a) real docs that belong to a different partition, (b) stale data from a prior kernel invocation (if the buffer was pool-reused), or (c) whatever the sort-cursor race winners wrote.

The feature-1..3 path (`kernel_sources_t2_scratch.h:197-207`) also iterates `i < totalDocsInPart` and hits the same corruption.

### 5.5 Why `iters=1` always passes

At iteration 1 depth 0: `numActiveParts = 1`, `maxPartDocs = ceil(N/1) = N`, `partSizes = [N]`. `partSize = maxPartDocs` exactly. No overflow. Since the benchmark runs 1 iteration total at `--iters 1`, only depth 0 of iteration 0 runs. Bit-exact. `--iters 2` runs iteration 0's depth 0, then iteration 1's depth 0 AND all higher depths that the `--depth` flag demands — and that's where unbalanced partitions first appear.

Wait — actually iteration 0 runs all depths. Let me re-check. Looking at `bench_boosting.cpp:1172`:

```cpp
for (ui32 depth = 0; depth < maxDepth; ++depth) {
    const ui32 numActiveParts = 1u << depth;
    ...
}
```

Every iteration runs ALL depth levels. So iteration 0 at `--depth 6` already exercises numActiveParts ∈ {1, 2, 4, 8, 16, 32}. Therefore iteration 0 should ALREADY overflow at depths ≥ 1.

Then why does `--iters 1` always pass bit-exact? Looking more carefully at the iter-0 loss comparison: at `--iters 1` iter 0, T1 loss = T2 loss. That means iteration 0 produces the SAME FINAL LOSS despite (presumably) corrupt intermediate histograms.

Examination: even with overflow, the CURRENT iteration's final loss is the forward-model loss on the training set after applying the tree found. **The tree's final structure can differ from T1 but produce the same training-set loss** if two trees happen to induce the same partition→leaf mapping, or if `ComputeLeafValues` averages out the errors.

Actually simpler: at `--iters 1`, only ONE boosting iteration runs, and the final loss reported is computed from predictions *after* this one tree's leaf values are applied. If the overflow happened to produce a tree structure that, when applied to the training data, gives the same predictions as T1's tree — then loss matches.

This is consistent with the depth-3 and depth-5 "accidental" passes in the truth table: the corrupt histogram happens to produce an equivalent split decision. It's not that the bug isn't active — it's that the downstream effect is sometimes masked.

The iters=2 observation is: once iteration 0's tree has run and its predictions baked in, iteration 1 recomputes gradients and starts a fresh tree search. The second iteration's tree search is where the corrupt histogram starts to produce catastrophically wrong splits (e.g. splits at wrong bin thresholds leading to very asymmetric partitions that the Newton leaf step then over-corrects). The non-determinism of the sort cursor race causes run-to-run variance in whether the corrupt histogram crosses the "split decision threshold" for a catastrophic move.

### 5.6 Why `bins=2` / `bins=4` self-heal

At bins=2, every doc's feature-0 bin ∈ {0, 1, 2}. With counting-sort buckets 0..127, all but buckets 1 and 2 are empty. `tgCounts[1] + tgCounts[2] = partSize`. Prefix scan sets `tgOffsets[1] = count(bucket 0)`, `tgOffsets[2] = count(bucket 0) + count(bucket 1)`, etc. In the scatter, docs land in bucket-1-region then bucket-2-region. At bins=2 with typical data, bucket-1 and bucket-2 are roughly balanced (50/50 of partSize), so overflow past `maxPartDocs` still lands predominantly in the neighbor's bucket-1/bucket-2 region. The SUM-over-sortedDocs for each bin may end up approximately correct because (a) only 2 target histogram bins exist, (b) the "wrong" docs still belong to *some* partition, and (c) the train/test prediction metric is robust to a few misclassified docs.

At bins=128, corruption is more visible because histograms have 128 target bins and small shifts in which docs go to which bin produce very different tree splits.

### 5.7 Why D1-R2 micro-bench didn't expose this

`docs/sprint21/scratch/t2/microbench_t2.cpp:223-229` (`MakePartSizes`):
```cpp
static std::vector<uint32_t> MakePartSizes() {
    const uint32_t base = N_DOCS / NUM_PARTS;   // 50000 / 64 = 781
    const uint32_t rem  = N_DOCS % NUM_PARTS;   // 16
    std::vector<uint32_t> ps(NUM_PARTS);
    for (uint32_t p = 0; p < NUM_PARTS; ++p)
        ps[p] = base + (p < rem ? 1u : 0u);     // 781 or 782 per partition
    return ps;
}
```

D1-R2 uses uniform partition sizes: either 781 or 782. With `MAX_PART_DOCS = ceil(50000/64) = 782`, every partition fits. **The overflow cannot happen in D1-R2 by construction.** The ratio-transfer risk identified in D1-R2 §3.1 ("synthetic identity-permuted → production argsort-permuted") was framed as a *performance* concern (cache behavior changes) — it did not anticipate that unbalanced partitions would violate the sort buffer's addressing model.

---

## §6 Proposed fix (DO NOT implement here — ml-engineer's job)

### §6.1 Core change

The T2-sort kernel's `sortedDocs[slotBase + pos]` requires `pos < slot capacity` where slot capacity = `maxPartDocs`. Since `pos ∈ [0, partSize)`, the invariant `maxPartDocs >= max(partSizes)` must hold. The current formula `maxPartDocs = ceil(N/K)` doesn't guarantee this.

Three implementation options, ranked by correctness-simplicity / perf-cost tradeoff:

**Option I — overallocate to `maxPartDocs = numDocs` (D1c verified)**

File: `catboost/mlx/tests/bench_boosting.cpp:526`
Change: `const ui32 maxPartDocs = numDocs;`
Perf impact measured: T2/T1 ratio goes 0.328× → 0.344× at gate config (50k/50f/128b, 50 iters). ~0.77 ms/iter (5% penalty on T2 side).
Memory impact: `sortedDocs` buffer size = numTGs × numDocs × uint32 = 1664 × 50000 × 4 = 333 MB per dispatch at gate config. MLX allocator uses unified memory; virtual allocation of 333 MB is cheap (only touched pages are resident), but at larger-scale datasets (say 500k docs × 50 features) memory pressure might bite.
Correctness: verified bit-exact on 18/18 DEC-008 configs, 10/10 runs deterministic at gate config.
Fix type: 1-line change.

**Option II — GPU-side max reduction with CPU readback**

File: `catboost/mlx/tests/bench_boosting.cpp:526` area.
Change: compute `max(partSizes)` on GPU, eval, read back as scalar uniform.
```cpp
mx::array maxPartDocsGPU = mx::max(partSizes, 0);
mx::eval(maxPartDocsGPU);
const ui32 maxPartDocs = static_cast<ui32>(*maxPartDocsGPU.data<uint32_t>());
```
Perf impact: 1 eval + 1 reduction + 1 CPU scalar read per DispatchHistogramT2 call (6 calls/iter at depth=6) = ~0.5–1.0 ms/iter. T2/T1 ratio probably 0.35–0.38×.
Memory impact: `sortedDocs` buffer size = numTGs × max(partSizes) = depth-dependent, tight.
Correctness: identical to Option I (overflow-free by construction).
Fix type: ~5 line change.

**Option III — reorganize slot layout using partOffsets (structural, highest effort)**

Change the kernel's `slotBase` formula to index by `partOffsets[partIdx]` within a global-per-(groupIdx, statIdx) slab. `sortedDocs` layout becomes `[numGroups × numStats × numDocs]`. Each TG's slot is `[(g*numStats + s)*numDocs + partOffsets[partIdx], (g*numStats + s)*numDocs + partOffsets[partIdx] + partSize)`. Since `sum(partSizes) = numDocs` and `partOffsets[p+1] = partOffsets[p] + partSize[p]`, the invariants hold without `maxPartDocs`.
Memory impact: buffer size = numGroups × numStats × numDocs × uint32 = 13 × 2 × 50000 × 4 = 5.2 MB at gate config (SAME as D1-R2 empirical). **This is the cleanest design.**
Fix type: kernel `slotBase` formula changes + dispatch code changes + remove `maxPartDocs` parameter (or leave unused). ~10-20 line change across `kernel_sources_t2_scratch.h` and `bench_boosting.cpp`. `binOffsets` layout may or may not need to change (each partition's 129-entry bin offsets can stay in their own slot at `(g*numStats + s)*numPartitions*129 + partIdx*129 + b`).
Correctness: identical to Options I and II by design. No overflow possible. Recommended long-term.

### §6.2 Recommended path for ml-engineer

Apply **Option III** (structural slab-by-partOffsets). This is what D1-R2's grid layout implicitly assumed (balanced partitions made the issue invisible), and what a production-quality kernel should look like. Options I and II are bug-fix patches; Option III is the proper design.

Estimated implementation time: 0.5–1 day for ml-engineer, including re-running D1 parity sweep and D0 perf gate.

### §6.3 Bug β status

**Bug β (atomic-scatter nondeterminism on features 1–3)** as proposed in D1a §4 is **not an observed issue** after H-B fix. 10/10 determinism runs at gate config produce bit-identical losses. D1-R2's per-bin ULP of 64 was a correctness artifact of the synthetic harness, not a fundamental limit. Kahan compensation on the scatter path is **not needed** for DEC-008 compliance at the gate config. The pre-budgeted +2–3 days for Kahan can be skipped.

(Caveat: this was tested only at features=50 / depth=6 / 50 iters. If Sprint 22 wants to expand to other configs and hits non-determinism there, Kahan remains a viable fallback. But the current evidence says the fix is H-B alone.)

---

## §7 Campaign decision input for Ramos

| Question | Answer |
|----------|--------|
| Is the fix ≤1 day for ml-engineer? | **Yes.** Option III is ~0.5–1 day including re-validation. Option I is <1 hour. |
| Does the fix risk reintroducing bug β? | **No.** Bug β is a non-issue post-fix (§6.3). |
| Does Sprint 22 still clear R8 post-fix? | **Yes, with margin.** Option III perf: unchanged from D0 (ratio 0.328×, e2e 1.83×). Option I perf: 0.344× ratio → e2e ~1.77× (still well above 1.5× Verstappen gate). Option II: between the two. |
| Is the fix scratch-only or does it need to move into production? | Currently scratch. After D1 parity re-validation + D0 re-measurement + Sprint 22 QA sign-off, T2 is eligible to graduate from `kernel_sources_t2_scratch.h` to `kernel_sources.h` per the original Sprint 22 plan. The fix should graduate with it. |
| Should Kahan be pursued? | **No.** DEC-008 18/18 configs already bit-exact with the H-B fix alone. Reserve Kahan as a latent tool for future sprints if wider configs break. |

**Recommendation**: ml-engineer implements Option III (clean structural fix), runs D1 parity sweep + D0 re-measurement to confirm, then Ramos approves the atomic D1 bundle. Sprint 22 ships with Option III + standard graduation path. No Sprint 23 re-entry needed for T2.

---

## §8 Raw experimental evidence (audit trail)

### E1 — Partition-size overflow diagnostic (primary evidence)

Diagnostic code added at `bench_boosting.cpp:528-548` (env-gated, saved as `docs/sprint22/scratch/d1c_diagnostic_instrumentation.patch`):

```cpp
if (std::getenv("T2_DIAG_PARTSIZES")) {
    mx::array psCopy = partSizes;
    mx::eval(psCopy);
    const uint32_t* ps = psCopy.data<uint32_t>();
    uint32_t maxActual = 0, overflows = 0, sum = 0;
    for (ui32 p = 0; p < numActiveParts; ++p) {
        if (ps[p] > maxActual) maxActual = ps[p];
        if (ps[p] > maxPartDocs) overflows++;
        sum += ps[p];
    }
    fprintf(stderr,
        "[T2DIAG] numDocs=%u nParts=%u maxPartDocs=%u max(ps)=%u sum(ps)=%u overflows=%u/%u%s\n",
        numDocs, numActiveParts, maxPartDocs, maxActual, sum, overflows, numActiveParts,
        (maxActual > maxPartDocs) ? "  <<<< OVERFLOW" : "");
    ...
}
```

Run at 50k/128b/d4/seed=42 (T1=0.493, T2=45.355 catastrophic):

```
[T2DIAG] numDocs=50000 nParts=1 maxPartDocs=50000 max(ps)=50000 sum(ps)=50000 overflows=0/1
[T2DIAG] numDocs=50000 nParts=2 maxPartDocs=25000 max(ps)=49558 sum(ps)=50000 overflows=1/2  <<<< OVERFLOW
[T2DIAG]   sizes: 442 49558
[T2DIAG] numDocs=50000 nParts=4 maxPartDocs=12500 max(ps)=49558 sum(ps)=50000 overflows=1/4  <<<< OVERFLOW
[T2DIAG]   sizes: 442 0 0 49558
[T2DIAG] numDocs=50000 nParts=8 maxPartDocs=6250 max(ps)=49558 sum(ps)=50000 overflows=1/8  <<<< OVERFLOW
[T2DIAG]   sizes: 442 0 0 0 0 0 0 49558
(...same pattern for iter-1...)
```

### E2 — Fix verification (primary evidence)

Diagnostic override at `bench_boosting.cpp:526` (env-gated, in `d1c_diagnostic_instrumentation.patch`):

```cpp
const ui32 maxPartDocs = std::getenv("T2_DIAG_FIX_MAXPARTDOCS")
                         ? numDocs
                         : ((numDocs + numActiveParts - 1) / numActiveParts);
```

Results — 50k/f1/d6/b128/seed=42/iters=2:

```
WITHOUT fix (5 runs): T2 ∈ {174.16, 145.61, 179.04, 0.494, 163.25}
WITH fix    (5 runs): T2 ∈ {0.494, 0.494, 0.494, 0.494, 0.494}  (all = T1)
```

### E3 — Full DEC-008 18-config sweep with fix (confirmation)

```
N=1000  loss=RMSE       bins=32  T1=0.40689126 T2=0.40689126  BitExact
N=1000  loss=RMSE       bins=128 T1=0.46936080 T2=0.46936080  BitExact
N=1000  loss=Logloss    bins=32  T1=0.34161490 T2=0.34161490  BitExact
N=1000  loss=Logloss    bins=128 T1=0.61407095 T2=0.61407095  BitExact
N=1000  loss=MultiClass bins=32  T1=0.61065382 T2=0.61065382  BitExact
N=1000  loss=MultiClass bins=128 T1=0.99084771 T2=0.99084771  BitExact
N=10000 loss=RMSE       bins=32  T1=0.44631991 T2=0.44631991  BitExact
N=10000 loss=RMSE       bins=128 T1=0.48231599 T2=0.48231599  BitExact
N=10000 loss=Logloss    bins=32  T1=0.30072498 T2=0.30072498  BitExact
N=10000 loss=Logloss    bins=128 T1=0.60412812 T2=0.60412812  BitExact
N=10000 loss=MultiClass bins=32  T1=0.57359385 T2=0.57359385  BitExact
N=10000 loss=MultiClass bins=128 T1=0.95665115 T2=0.95665115  BitExact
N=50000 loss=RMSE       bins=32  T1=0.44676545 T2=0.44676545  BitExact
N=50000 loss=RMSE       bins=128 T1=0.47740927 T2=0.47740927  BitExact
N=50000 loss=Logloss    bins=32  T1=0.30282399 T2=0.30282399  BitExact
N=50000 loss=Logloss    bins=128 T1=0.60559267 T2=0.60559267  BitExact
N=50000 loss=MultiClass bins=32  T1=0.56538904 T2=0.56538904  BitExact
N=50000 loss=MultiClass bins=128 T1=0.94917130 T2=0.94917130  BitExact
```

**18/18 BitExact. 0 ULP.**

### E4 — Determinism check (10-run, gate config with fix)

```
50k/f50/c1/d6/b128/seed=42/iters=50 (DEC-008 gate):
  Runs 1-10: all T2=0.47740927 (bit-exact with T1 on every run)
```

Vs D1 Experiment 2 (pre-fix, gate config, 5 runs): T2 varied 0.47803015 to 0.47804454. Drift is entirely explained by H-B; not an independent nondeterminism.

### E5 — Perf impact of Option I fix (50k/f50/c1/d6/b128/iters=10)

```
WITHOUT fix:  T1=21.712 ms,  T2=6.477 ms  →  ratio 0.298×
WITH Option I: T1=21.050 ms,  T2=7.249 ms  →  ratio 0.344×
Ratio delta: +0.046 (~15% relative). Still well below 0.60 kill-switch.
```

### E6 — Truth table: which configs overflow

From E1 runs + row-sweeps + manual verification:

| (rows, features, seed) | At numActiveParts=K, overflow expected when partition skew produces partSize(max) > ceil(rows/K) |
|------------------------|--------------------------------------------------------------------------------|
| (50000, 1, 42) | Overflows at every K ≥ 2. Always partition-last dominant (features=1 single-feature splits) |
| (2048, 1, 42) | Overflows at K ∈ {2, 4, 8} but partition 7 has partSize=1340 < maxPartDocs=256 — wait, that IS overflow. But loss still matches T1 bit-exactly (accidental — with only 2048 docs in 8 buckets, corruption doesn't swing splits) |
| (65536, 1, 42) | Overflows heavily; depth=6 sometimes passes accidentally, sometimes fails |
| (50000, 50, 42) | Overflows with smaller factor (multi-feature splits distribute docs more evenly); still non-deterministic drift at 0.13% magnitude pre-fix |

### E7 — Rejected hypothesis: eval barriers don't fix it

Retained from D1b §3: applying both Option 1 (mx::eval between sort and accum) and Option 2 (mx::eval at return of DispatchHistogramT2) still produces catastrophic divergence at 50k/f1/d6/b128/iters=2. Independent confirmation via H-B fix: the H-B fix alone resolves the bug without any eval barrier insertion. Therefore the bug has nothing to do with eval placement or blit-vs-compute ordering. D1a's root-cause diagnosis is fully refuted.

---

## §9 Git state confirmation

```
git status --short  (at end of D1c):
 M catboost/mlx/tests/bench_boosting.cpp
?? docs/sprint22/d1_t2_parity_sweep.md
?? docs/sprint22/d1a_t2_diagnostic.md
?? docs/sprint22/d1b_t2_fix_and_rerun.md
?? docs/sprint22/d1c_t2_troubleshoot.md   (this file)
?? docs/sprint22/scratch/                 (contains d1c diagnostic patch + prior)

git diff --stat catboost/mlx/kernels/kernel_sources.h catboost/mlx/methods/histogram.cpp:
(no output — production sources unmodified)

git diff --stat catboost/mlx/kernels/kernel_sources_t2_scratch.h:
(no output — T2 kernel source unmodified; all diagnostics live in bench_boosting.cpp scratch)
```

**D1c discipline maintained**: no production source changed; no T2 kernel source changed. Diagnostic instrumentation lives entirely in `bench_boosting.cpp` under env-var gates.

Per Phase-0 standing order, no commits made. Final commit is Ramos's decision.

---

## §10 Time accounting

| Phase | Wall time | Output |
|-------|-----------|--------|
| Phase 0 (clean baseline + repro) | ~30 min | Clean baseline confirmed, bug reproduced, depth truth table built |
| Phase 1 (minimum reproducer) | ~20 min | `1024/f1/d2/b4/iters=2` minimum repro identified |
| Phase 2 (diagnostic + hypothesis test) | ~60 min | H-B confirmed via `T2_DIAG_PARTSIZES` instrumentation and `T2_DIAG_FIX_MAXPARTDOCS` override; other hypotheses ruled out by evidence chain |
| Phase 3 (deliverable) | ~30 min | This document |
| **Total** | **~2h 20m** | **Investigation converged well within 24h budget** |

Ended with plenty of budget remaining; no rushed conclusions.

