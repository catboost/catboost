# Sprint 23 R2 — Dispatch Inversion Research Spike

**Branch**: `mlx/sprint-23-t2-promotion` (tip `be530059da`)
**Campaign**: Operation Verstappen — battle 8 of 9
**Role**: 2-day timeboxed research spike (Day 1 executed; Day 2 sanity + measurement plan)
**Author**: @research-scientist
**Status**: **NO-GO.** Structural algebraic blocker identified in §3 + worst-case dispatch math in §1 rules out a mechanism-testable win within the Verstappen campaign window.
**Deliverable spec**: `docs/sprint23/README.md §4.2`
**Prior context**: `docs/sprint21/d1r4_synthesis.md §3` Rank #2 "Dispatch inversion"; DEC-017 (T3b retired); DEC-018 (variant A retired); DEC-020 (T2 shipped 1.90×); DEC-023 (T2 atomic-float race, OPEN S24 scope).

---

## §0 TL;DR

- **Mechanism**: replace partition-fragmented dispatch (one histogram per `(partition, feature, stat)` triple) with a single histogram over all docs (`feature × stat × bin`), plus a scoring-time per-partition mask to recover left/right partition bin-sums.
- **Parity verdict**: STRUCTURAL BLOCKER. Computing per-partition bin sums from a single all-docs histogram is algebraically impossible without a per-partition accumulator — scoring needs `h_p[f][b] = Σ_{d ∈ p, bin_f(d)=b} stat(d)`, which collapses to the single-histogram total `H[f][b] = Σ_p h_p[f][b]` and cannot be un-summed. Reconstructing `h_p` from `H` requires either (a) a second per-partition pass over docs (equivalent work to today's dispatch), (b) a doc-level `bin × partition` tensor (larger than current histogram by 2-3×), or (c) bit-packed doc→partition with per-bin gather at scoring (shift work, not eliminate it).
- **Dispatch-math verdict**: even granting a miracle mask mechanism, the optimistic inversion budget at gate is 1.97 ms (shape-scaled T2 feature-0 scan at 195 docs/thread). Current production T2 is **7.014 ms (D0) / 6.796 ms (D4)**. The **≥5.05 ms headroom looks large but is already consumed by the mask-application cost**: even the cheapest per-bin mask (one gather per `(partition, feature, bin)` triple = 64 × 50 × 128 = 409,600 gathers/iter) adds ≥4-6 ms of kernel time at AGX bandwidth. Net-neutral optimistic case.
- **Parity risk**: inversion merges all 64-partition atomic writers into a single per-bin contended slot → atomic contention rises ~64× → DEC-023 structural race gets strictly worse (N=10k config #8 already bimodal with 1638 TGs; inversion dispatch would have 26 TGs × 195 docs/thread contending on 128 bins × 4 features = same 512 slots, but each slot with 64× more writers). Cannot ship within DEC-008 envelope without Kahan-class compensation.
- **Comparison to falsified predecessors**: R2 is the non-atomic-CAS cousin of DEC-017 (T3b) — both restore 195 docs/thread shape at gate by reducing TG count. DEC-017 failed at production shape (+42.3% regression) because the toy-kernel single-TG timing did not survive the atomic contention of production partition fragmentation. R2 structurally trades dispatch fragmentation for per-bin atomic fragmentation — the same contention surface, relocated to scoring time, plus a new reconstruction cost.
- **Kill-switch invoked at end of Day 1**: §§1-4 did not produce a concrete, testable mechanism-direct benchmark. Mask mechanism remains hand-wavy (§2) and parity risk is structural-unfixable without redesigning T2's feat-0 single-writer guarantee (§3).
- **Final verdict**: **NO-GO.** Defer permanently. Recommendation in §5 documents why S24+ should not re-enter this design space without new evidence of a structurally different reconstruction mechanism.

---

## §1 Dispatch shape math

### §1.1 Current T2 production dispatch shape at gate (from D4 + histogram_t2_impl.cpp)

- Config: 50k docs / 50 features / depth=6 / 128 bins / RMSE (numStats=2) / approxDim=1.
- `numGroups = ceil(50 / 4) = 13` feature groups of 4 packed features.
- `numPartitions` at depth=6: up to 64 (active partitions vary per iter; at depth=6 the symmetric-tree design dispatches for all 64).
- `maxBlocksPerPart = 1` (NIT-4 invariant).
- Grid: `(256 * 1 * 13, 64, 2) = (3328, 64, 2)` threads → **1664 TGs** (13 × 64 × 2).
- Docs/TG at depth 6: avg 50000/64 ≈ 781 docs/part × 1 block/part = **781 docs/TG** (per feature group × stat); per-thread ≈ 3 docs/thread.
- Histogram buffer: `numPartitions × numStats × totalBinFeatures = 64 × 2 × totalBins ≈ 64 × 2 × 6350 = 812,800 float` ≈ 3.2 MB.
- Measured: **hist_ms = 6.796 ms (D4 gate cross-session); iter_total_ms = 19.098 ms.**

### §1.2 Inverted dispatch shape (proposed)

Single histogram over all 50k docs, shape `(numGroups × numStats, feature × stat × bin)`.

- Buffer: `numStats × totalBinFeatures = 2 × 6350 = 12,700 float` ≈ 50 KB. **Factor-of-64 reduction in histogram buffer size.**
- TG count: at 50k docs / 195 docs/thread × 256 threads/TG = **~1 TG per (group, stat) → 13 × 2 = 26 TGs** (matches variant A shape exactly).
- Docs/thread: **~195** (instead of ~3).

### §1.3 Per-depth comparison

Assuming maxBlocksPerPart=1 (per DEC-021 Option III) and gate doc count:

| Depth | numParts | Current T2 TGs | Current docs/thread | Inverted TGs | Inverted docs/thread |
|---|---:|---:|---:|---:|---:|
| 0 | 1 | 26 | ~195 | 26 | ~195 |
| 1 | 2 | 52 | ~98 (skewed: could be [1,195] at extreme) | 26 | ~195 |
| 2 | 4 | 104 | ~49 | 26 | ~195 |
| 3 | 8 | 208 | ~24 | 26 | ~195 |
| 4 | 16 | 416 | ~12 | 26 | ~195 |
| 5 | 32 | 832 | ~6 | 26 | ~195 |
| 6 | 64 | 1664 | ~3 | 26 | ~195 |

Mechanism case-closed: inversion restores 195 docs/thread at every depth. This is the attraction.

### §1.4 Optimistic inversion-only perf budget at gate

Use D1-R2 toy-kernel inversion evidence as the optimistic scaling bound. D1-R2 measured variant A (T1 at 26 TGs × 195 docs/thread = essentially the inverted dispatch shape for T1, minus masking) at **3.43 ms**. T2 variant A (same shape, T2 accumulator) = **0.98 ms**. So:

- **Optimistic hist_ms (inverted T2 accumulator, 26 TG, 195 docs/thread, no mask cost)**: ~0.98 ms.
- **Current T2 hist_ms at gate**: 6.796 ms.
- **Optimistic savings (no mask cost)**: 5.82 ms.

But this 0.98 ms is the T2 accumulator already running at the shape that variant A attempted to force. T2 at production fragmented shape (6.796 ms) is ~6.9× slower than at restored shape (0.98 ms), confirming what DEC-017 retirement already established: **the fragmentation tax is real and ~5-6 ms at gate**. If inversion could be achieved for free, it would recover that tax.

This is the **headroom that must be spent on the mask reconstruction cost** (§2). Inversion is viable iff `mask_application_ms < 5.82 ms` with margin.

### §1.5 Worst-case memory and dispatch overhead sanity

- AGX unified memory bandwidth: ~200 GB/s sustained.
- Mask buffer candidates (§2): at minimum, doc→partition map of `50000 × 4 B = 200 KB` (fits in L2). At worst, `bin × partition` tensor of `12,700 × 64 × 4 B = 3.2 MB` — same order as current histogram, no worse.
- Dispatch overhead per TG on AGX (measured S21 D0): fixed ~1.3 µs/TG. 26 TGs vs 1664 TGs = ~34 µs vs 2.2 ms gross launch cost. **1.97 ms of savings is pure dispatch overhead elimination** — consistent with §1.4 but NOT a free lunch: the accumulator at 195 docs/thread still does the same feat-0 sort-and-scan work plus 4× the per-thread scatter work for features 1-3.

---

## §2 Scoring-time mask mechanism

This is the section that must land concretely for R2 to be GO. Three candidate mask mechanisms, each evaluated against the cost budget from §1.4.

Goal: at split scoring, for a given split candidate `(feature f, bin threshold b)`, need **per-partition**:
- `sumLeft[p]  = Σ_{d ∈ p, bin_f(d) ≤ b} stat(d)`
- `sumRight[p] = Σ_{d ∈ p, bin_f(d) > b} stat(d)`
- Currently derived from per-partition histograms by suffix-sum + `totalSum[p] - sumRight[p]` (see `kScoreSplitsLookupSource` :541-548).

After inversion, only global `H[f][b] = Σ_d stat(d)·[bin_f(d)=b]` is available directly. Per-partition split scores require reconstructing `h_p[f][b] = Σ_{d ∈ p, bin_f(d)=b} stat(d)`.

### §2.1 Mechanism A — doc-level gather at scoring (hand-wavy)

For each `(p, f, b)` triple at scoring time, loop over all docs in partition `p`, extract `bin_f(d)`, conditionally accumulate `stat(d)`.

- Work: `numPartitions × totalBinFeatures × avg_docs_per_partition = 64 × 6350 × 781 = 3.2e8` doc-level reads per iter.
- Equivalent to recomputing the histogram from scratch per scoring pass. **Strictly more expensive than current T2 (which does this work once and caches in `h_p`).**
- **Rejected**: this is not inversion, this is lazy evaluation of the same histogram.

### §2.2 Mechanism B — per-bin partition bitmap

Maintain a `bin × partition` bitmap `M[f][b][p]` = mass of docs in bin `b` of feature `f` that fall in partition `p`. But this IS the per-partition histogram `h_p[f][b]` expressed differently. Computing `M` requires the same per-partition accumulation the inversion was supposed to avoid.

- Rejected as circular.

### §2.3 Mechanism C — doc→partition lookup + gather-by-sorted-order

Leverage T2's existing sort-by-bin: for feature 0, `sortedDocs[]` gives docs in bin order; bin range `[binOffsets[b], binOffsets[b+1])` lists docs falling in bin `b`. For each doc, look up its partition from `partitions[d]` (already exists, written by tree-apply kernel `kTreeApplySource:886`).

At scoring for feature 0, to recover `h_0[p][b]`:
- For each bin b in [1..128]: walk `sortedDocs[binOffsets[b]..binOffsets[b+1])`, accumulate `stat[d]` into `h_0[partitions[d]][b]`.
- This IS feat-0 T2 accumulation, just moved from histogram phase to scoring phase. **Same cost.**
- For features 1-3: no sort available; must do `numDocs × {1,2,3}` gathers per scoring pass — larger cost than current T2.

**Rejected**: mask application cost equals or exceeds the ~5.82 ms headroom from §1.4. Inversion shifts work from hist phase to scoring phase without reducing it.

### §2.4 Mechanism D — segmented prefix-sum over docs sorted by (bin, partition)

Sort docs by `(bin, partition)` composite key; run segmented prefix sum over the partition-change boundaries within each bin slab.

- Sort cost: `O(N log N)` or `O(N + |bins| × |partitions|)` counting sort = 50000 + 128 × 64 = 58,192 slots. Feasible.
- Segmented reduce: `O(N)` = 50000 ops × 4 features = 200k ops per iter. Negligible at AGX bandwidth.
- BUT: must be done **per feature** (sort key includes `bin_f`). Four independent sorts per iteration. Each sort is ~same cost as T2's current feature-0 sort (~2-3 ms empirically from D4's 7 ms T2 total minus ~4 ms accum).
- Total: ~4 × 2.5 ms = **10 ms mask-preparation cost**. Exceeds current T2 total by 50%.

**Rejected**: cost blows the budget by ~1.7×.

### §2.5 Mechanism E — doc-level partition-indexed scatter during accumulation

Abandon inversion proper: use 195 docs/thread shape, but scatter into `h[f][b][p]` directly (per-partition bin atomically). Buffer size: same as current histogram. TG count: 26. Docs/thread: 195.

- This is NOT inversion — it is a restored-shape T2 with per-partition accumulation kept. Equivalent to variant A + T2 accumulator, which is essentially what DEC-017 and DEC-018 attempted in different forms.
- Mechanism works only if 26-TG dispatch at 195 docs/thread is faster than 1664-TG at 3 docs/thread.
- D1-R2 variant A measurement says T2 at restored shape = 0.98 ms vs T2 at fragmented shape = 6.8 ms. Would be a ~6× savings if it transfers. But the per-bin atomic contention at 195 docs/thread × 64 partitions = 12,480 writers/bin vs 3 docs/thread × 64 parts = 192 writers/bin is **65× higher contention per bin**, which DEC-017 measured as the regression mechanism.
- **This is DEC-017 T3b revisited with a different label.** Already falsified empirically (+42.3% regression at gate).

**Rejected**: DEC-017 reincarnation. Same atomic-contention failure mode.

### §2.6 Summary

None of A-E produces a mask mechanism that is (a) cheap, (b) parity-safe, and (c) not algebraically equivalent to the current per-partition accumulation. The mask-mechanism design space is exhausted at the sketch level with no viable candidate.

---

## §3 Parity risk

### §3.1 Atomic contention under inversion

Current T2 features 1-3 path (production shipped, 1.90×):
- 1664 TGs × 256 threads × 3 docs/thread × 3 features = **per-bin atomic_fetch_add writers per iteration ≈ 50000 × 3 / 128 bins × 4 features = 2343 writers per `(feature, bin)` global slot** per iteration, spread over 64 partitions = 37 writers per `(feature, bin, partition)` slot.
- DEC-023 fires at N=10k config #8 (RMSE/128b): 10000 × 3 / 128 × 4 = 468 writers per `(feature, bin)` slot = 29 writers per per-partition slot. **29 contending atomic-float writers already produce bimodal parity drift.**

Inverted T2 (Mechanism E flavor, only shape that retains 1664→26 TG reduction):
- 26 TGs × 256 threads × 195 docs/thread × 3 features → **12,480 × 3 writers per `(feature, bin)` global slot**, distributed over 64 partitions = 585 writers per per-partition slot (16× worse than current, 20× worse than DEC-023 trigger).
- Or if truly inverted (no partition dimension), all 12,480 × 3 writers contend on 128 × 4 = 512 global slots = **73 writers per slot** (2.5× DEC-023 trigger).

### §3.2 DEC-008 envelope compliance

DEC-008 requires RMSE/Logloss ULP ≤ 4 (≈4.77e-7 FP32 relative). Current T2 satisfies at gate (100/100 deterministic per D4; DEC-020). DEC-023 documents 1 latent bimodal at N=10k.

Higham worst-case error for atomic-sum of `N` IID FP32 values is `γ_N = Nε/(1-Nε)` where `ε = 2^-24 ≈ 6e-8`. At 73 writers/slot: `γ_73 ≈ 4.4e-6` → **exceeds DEC-008 RMSE/Logloss ULP≤4 by ~9×**.

Per-partition accumulation variant (Mechanism E): 585 writers/slot → `γ_585 ≈ 3.5e-5` → **exceeds DEC-008 by ~73×** — same order as the DEC-017 worst-case estimate.

### §3.3 Kahan / fixed-point fallback (DEC-022 scope qualifier + DEC-023 Options 1-2)

- **Kahan**: probably insufficient at 500+ writers/slot; DEC-023 notes Kahan "mitigates but does NOT eliminate non-determinism" (DEC-023 Option 3). Also adds ~2× atomic cost per bin.
- **Threadgroup-local reduce + single-thread commit** (DEC-023 Option 1): structurally clean. Requires the inverted dispatch to have TGs that each own a doc range and commit one per-bin sum to the global slot without atomic contention — a second reduction pass. Adds ~1-2 ms per pass × 4 features = 4-8 ms. Consumes ~100% of the optimistic headroom.
- **Int-atomic fixed-point** (DEC-023 Option 2): deterministic by construction, but at inverted dispatch's ~73-585 writers per slot, the int-atomic contention still serializes. Same mechanism as DEC-017 T3b atomic-CAS, already falsified at production shape (+42.3%).

### §3.4 Parity-risk verdict

Structural. No single-sprint fix preserves DEC-008 envelope AND delivers the ~5.82 ms inversion budget win AND eliminates the per-bin contention that R2 creates. **Parity is a binary blocker for R2 in its current formulation.**

---

## §4 Mechanism gate spec (NOT ACHIEVABLE)

Per the Day-1 kill-switch: §§1-3 do not converge to a concrete testable design, therefore §4 should not attempt a falsifier for a design that doesn't exist.

For posterity, if a future researcher revives R2 with a novel mask mechanism not covered in §2, the mechanism gate MUST be:

1. **Isolate inverted-hist-only ms**: build a prototype that performs only the single-histogram accumulation (no per-partition output, no scoring integration). Measure in same-session T1/T2 harness at 50k gate config. Acceptable predicted: ≤ 0.98 ms (variant-A D1-R2 evidence).
2. **Isolate mask-application-only ms**: with a pre-computed inverted histogram and a synthetic partition membership, dispatch only the mask kernel. Measure at same gate config.
3. **Sum check**: `hist_ms_inverted + mask_ms < current T2 hist_ms = 6.796 ms` with margin ≥ 10% (i.e., ≤ 6.12 ms sum). Below 10% margin = conservative band (≤ 6.46 ms). Above 6.796 ms = falsified.
4. **Small-partition probe**: repeat at N=1000 gate-shape (where current T2 is 0.66× ratio per D4 §3, i.e. worst). If inversion is structurally correct, the N=1000 ratio should be strictly better than current T2 (1664/16 TGs of wasted capacity at N=1000 is the largest inversion win).
5. **Parity gate**: 18-config DEC-008 envelope sweep with 5 runs/non-gate config + 100 runs at gate (per the S23 D0 standing-order protocol). All 18 configs must be ≤ DEC-008 ULP envelope. Mechanism E/atomic variants will blow this by ≥ 9×.

This gate is **mechanism-direct** (does not test an amortization proxy — that was DEC-018's retired error). It isolates the two distinct cost components of the proposed lever and tests the mechanism sum directly against the current shipped total.

---

## §5 Feasibility verdict

**NO-GO.**

Justification:
1. **Day-1 kill-switch triggered** (§2): no concrete mask mechanism survives the cost budget from §1.4. A/B/C/D are algebraically or empirically rejected. E is DEC-017 T3b with a different name.
2. **Parity is a structural blocker** (§3): any inversion-class mechanism increases per-bin atomic contention by 2× to 20× over DEC-023's current trigger, pushing the Higham bound out of DEC-008 envelope by 1-2 orders of magnitude.
3. **Dispatch math at best is net-neutral** (§1.4, §2.3-2.5): the 5.82 ms headroom from fragmentation elimination is consumed by mask-application or deterministic-reduce costs.
4. **Distinction from DEC-017 is thin**: R2's Mechanism E (the only variant that retains the restored-shape win) is T3b without the CAS. The atomic contention failure mode is structurally identical — DEC-017's +42.3% regression is the predicted baseline outcome, not an aberration.

**Campaign window assessment**: Operation Verstappen R8 = 1.90× at S23 entry, already 40 pp above the 1.5× gate. R2 is not required to close the campaign. The incentive to pursue R2 further is minimal given the structural blockers above.

**Recommendation**: Do not revisit R2 in S24+ unless new evidence emerges of a mask mechanism that is (a) not equivalent to per-partition accumulation, (b) cheaper than 5 ms at gate config, and (c) parity-clean under DEC-008. Such evidence would likely come from a fundamentally different tree-search primitive (e.g., histogram-free split search, which is a different research program).

---

## §6 Sanity pass (Day 2)

### §6.1 Implicit AGX assumptions checked

1. **Threadgroup memory limit (32 KB)**: inversion uses same simdHist layout as T1, unaffected. ✓
2. **Atomic coherence on AGX**: `atomic_fetch_add_explicit(memory_order_relaxed)` on `device atomic_float*` is coherent but NOT deterministic across threadgroups — this is the DEC-023 root cause. R2 inherits and amplifies. ✓ (already accounted in §3)
3. **Dispatch overhead per TG**: measured ~1.3 µs/TG at S21 D0. 1664 → 26 TGs = ~2.1 ms savings bound. Correctly accounted in §1.5.
4. **L2 prefetcher hiding scatter-gather latency**: DEC-015/DEC-019 confirmed AGX out-of-order + prefetcher fully hides stats gather at production shape. R2's 195 docs/thread shape has LARGER per-thread prefetcher window and should inherit this property (makes inversion dispatch itself cheaper, not more expensive). This is good news for R2 on the accumulator side; the bottleneck remains §2 mask cost.
5. **`compressedIndex` line-size locality**: 50 features / 4 features-per-group = 13 ui32 per doc = 52 B/doc. Both T2 and inverted-T2 stride through the same compressed data. No locality difference. ✓

No AGX assumption violated. Falsification basis in §§2-3 is independent of AGX quirks.

### §6.2 Comparison to prior falsifications

| Falsified lever | Root failure mechanism | R2 parallel? |
|---|---|---|
| DEC-013 writeback-plurality | Writeback was 5% (0.79 ms), not 15 ms plurality | R2 fragmentation-tax IS ~5.8 ms, roughly real (§1.4). Not the same mis-attribution. |
| DEC-014 original gather / DEC-015 col-major | AGX prefetcher fully hides gather | R2 does NOT attack gather; it attacks fragmentation. Not a parallel. |
| DEC-017 T3b atomic-CAS | Toy-kernel 1-TG shape did not transfer to 1638-TG production shape; +42.3% regression | **R2 Mechanism E is T3b without the CAS.** Same shape-restoration mechanism, same atomic contention surface. Regression is the predicted outcome. |
| DEC-018 variant A | Specification error — gate tested T1 amortization (2.5%), not T3b shape restoration (>40%) | **R2 gate spec (§4) correctly isolates hist + mask.** Not a spec error — R2's §2 is a direct-mechanism design failure, not a gate-spec failure. |
| DEC-019 L2 stats pre-permute | Gather was fully hidden by prefetcher | Tangential; see §6.1.4. |

**What makes R2 different**: R2 tries to invert the dispatch without eliminating per-partition accumulation. The prior levers all targeted the wrong bottleneck (writeback, gather, stats locality). R2 targets the right bottleneck (dispatch fragmentation) but cannot solve the downstream reconstruction problem without (a) reintroducing the same contention, or (b) paying the same cost elsewhere.

### §6.3 "T3b revisited" distinction

**R2's claim to novelty**: T3b was a histogram-accumulator change at production dispatch shape. R2 is a dispatch-shape change holding the accumulator constant.

**Why they converge in practice**: T3b could only win at its "natural" 195 docs/thread shape, which is what R2 proposes to restore. The atomic contention that killed T3b was a function of writers-per-bin-slot at that restored shape. R2's Mechanism E has the SAME writers-per-bin-slot count (195 docs/thread × numPartitions), so inherits T3b's failure mode one-for-one.

**Distinction is thin**: the only variants of R2 that are TRULY distinct from T3b are Mechanisms A/B/C/D, all of which are algebraically or empirically rejected in §2.

---

## §7 Measurement plan for S24 — NOT APPLICABLE

Per §5 verdict. No S24 measurement plan is warranted.

If a future spike revives R2, the §4 gate is the template, and the plan would be:

- **Canonical configs**: gate (50k/RMSE/128b) + N=1000 small-partition probe (agreed with parent thread suggestion — small-partition is where current T2 underperforms, so inversion should show its largest relative win there).
- **Kill-switch**: `hist_ms_inverted + mask_ms > 6.12 ms` at gate = fail with >10% margin. **OR** any N=1000 config shows worse ratio than current T2 (current T2 N=1000 ratio 0.65-0.69× per D4 §3 — if inversion is the right theory, it should beat 0.50× at N=1000).
- **Expected outcome distribution**: Optimistic (mask_ms < 2 ms): ~2 ms hist + 2 ms mask = 4 ms total, 40% savings at gate. Unlikely given §2 analysis. Conservative: 5-6 ms total, neutral. Null: 7+ ms total (mask dominates), regression.
- **Budget**: 1 day ml-engineer time for prototype + 1 day for measurement + 0.5 day for document.

**This plan is filed for completeness only. §5 verdict supersedes.**

---

## §8 References

- `docs/sprint23/README.md §4.2` — R2 scope and Day-1 kill-switch authority.
- `docs/sprint21/d1r4_synthesis.md §3` Rank #2 — original dispatch-inversion thesis.
- `docs/sprint22/d4_perf_gate.md` — current T2 production numbers (0.317× ratio, 19.098 ms iter_total, 6.796 ms hist_ms).
- `docs/sprint22/d0_t2_production_shape.md` — D1-R2 variant A evidence (26 TGs × 195 docs/thread = 0.98 ms T2, 3.43 ms T1 — the shape-restoration ceiling).
- `catboost/mlx/kernels/kernel_sources.h` — `kT2SortSource`, `kT2AccumSource`, `kScoreSplitsLookupSource` (consumes per-partition histograms).
- `catboost/mlx/methods/histogram_t2_impl.cpp` — current T2 dispatch; grid `(256 * 13, 64, 2) = 1664 TGs` at gate.
- `catboost/mlx/methods/score_calcer.cpp:57-69` — per-partition-stats consumer; confirms scoring needs `h_p[f][b]`, not just `H[f][b]`.
- DEC-008 (parity envelope), DEC-017 (T3b retired — closest structural analog), DEC-018 (variant A retired — most similar shape-restoration attempt), DEC-020 (T2 shipped), DEC-022/023 (atomic-float race scope).
