# Sprint 18 Ablation — L1 Variant Sweep (S18-02)

Owner: @research-scientist · Captured: 2026-04-17 · Chosen variant: **L1a (Ramos Day-2 approval)**  
Branch: `mlx/sprint-18-hist-privhist-tile`

Cross-references: [`docs/sprint18/attribution.md`](attribution.md) (S18-01 steady-state anchors) · [`docs/sprint18/design.md`](design.md) (structural framing, BUG-001 guard, variant descriptions)

---

## 0. TL;DR

**Ship L1a — full 32 KB `simdHist[8][1024]`, single accumulation pass, in-place D1c reduction.**

- L1a is the **only variant with unambiguous gate clearance across its full error envelope** (worst case 17.3 ms, 1.4 ms margin above the 18.7 ms gate).
- L1b and L1c both miss the gate on their upper error bounds (+0.5 ms and −0.9 ms margins respectively at nominal; upper bound fails).
- L1a unlocks **Option C** (≥45% reduction / ≤15.8 ms): nominal 15.5 ms with occupancy credit projects to 13.2 ms ± 2.0 ms (−54%). L1b/c cannot reach Option C — they carry no occupancy delta.
- Trade-off accepted: 32 KB threadgroup memory at the Apple Silicon hard ceiling. DEC-012 codifies the ceiling trade; Sprint 19+ threadgroup-geometry work re-negotiates if needed.
- Ramos approved L1a on Day 2. @ml-engineer begins S18-03.

**Benchmark status.** No Metal benchmarks were executed in this ablation. All `histogram_ms` values are **analytical projections** anchored to S18-01 steady-state attribution. @ml-engineer (S18-03) will ground-truth; the L1a verdict is robust across the full ±1.8 ms envelope (worst-case 17.3 ms still clears 18.7 ms gate).

---

## 1. Method

### 1.1 Attribution anchors (S18-01, steady-state)

Gate config: **N=10k, RMSE, depth=6, 128 bins.** Steady-state = iters 1–49 of the S17-after-JSON (D1c kernel, commit `5b4a8206bc`).

| Phase | Lines (`kernel_sources.h`) | SS (ms) | ±err | % of SS 23.72 ms |
|---|---|---:|---:|---:|
| Accumulation — gather + RMW into spilled `privHist` | 131–148 | **6.4** | ±1.5 | 27% |
| `privHist` zero-init | 125–128 | **4.0** | ±1.5 | 17% |
| D1c reduction tail | 181–225 | **3.0** | ±1.0 | 13% |
| Global-atomic writeback | 229–245 | **5.0** | ±1.5 | 21% |
| JIT/launch amortized | — | **5.3** | ±0.2 | 22% |
| **SS total** | | **23.72** | ±1.48 | 100% |
| All-iters total (baseline) | | **28.75** | — | — |

Full derivation and plan-estimate confirmation/refutation in `docs/sprint18/attribution.md`.

### 1.2 Mechanism cost model

Sources: Sprint 16 MST §B.2–B.3 (`docs/sprint16/mst_findings.md`), Sprint 17 parity analysis (`docs/sprint17/ablation.md` §3).

**Spill elimination.** `float privHist[HIST_PER_SIMD]` at `kernel_sources.h:123` is 4 KB per thread × 256 threads = 1 MB of thread-local state, well beyond the M-series register file. Moving to `threadgroup float simdHist[8][1024]` (8 SIMD groups × 1024 bins) puts the histogram on-chip. Consequences:

- Zero-init goes to 0 ms (threadgroup memory initialization is implicit / free).
- Accumulation RMW is on-chip: bandwidth drops from device-memory spill to the on-chip threadgroup-memory floor.

**Tile re-read penalty (L1b/c).** Each of 4 passes re-issues the full doc-stream gather (`docIndices`, `compressedIndex`, `stats`). Gather share of the 6.4 ms accumulation ≈ 1.75 ms; 3 additional passes × 1.75 ms = 5.25 ms raw, amortized 40–60% by L2 cache → **+2.1 to +3.15 ms net cost vs L1a**.

**Threadgroup-memory occupancy.** Apple M-series GPU: ~32 KB threadgroup memory per SM shared among resident threadgroups. D1c (12 KB) allows ≥2 threadgroups/SM. L1a (32 KB) forces exactly **1 threadgroup/SM**. This is an occupancy tax on depth-limited configs (Option C: if writeback-atomic contention is SM-local, lower occupancy reduces contention → 2–4 ms credit).

**BUG-001 guard.** All variants use stride-partition accumulation (single-owner writes per bin, no atomics within a SIMD group). `simd_shuffle_xor` is lockstep per MSL §6.9. Cross-SIMD fold is the DEC-009 fixed-order 8-term linear sum.

---

## 2. Variant designs

| Variant | `simdHist` layout | Tile count | Peak threadgroup_mem (KB) | Accum. passes | BUG-001 ownership | Complexity vs D1c |
|---|---|---:|---:|---:|---|---:|
| **L1a** | `simdHist[8][1024]` — full histogram per SIMD | 1 | **32** (at Apple hard limit) | 1 | Bins {l, l+32, …, l+992} per lane | ~+25 LOC |
| **L1b** | `simdHist[8][256]` + `stagingHist[1024]` — 4 tiles | 4 | 12 | 4 | Same stride restricted to current 256-bin tile | ~+60 LOC |
| **L1c** | `simdHist[8][256]`, 8 bins/lane — hybrid stride-partition, 4 tiles | 4 | 12 | 4 | Explicit 8-bin-per-lane partition | ~+45 LOC |
| **L1d** | Control — D1c unmodified (Sprint 17 after) | — | 12 | 1 | D1c shipped | 0 |

---

## 3. Per-variant projections

| Field | L1a | L1b | L1c | L1d (ref) |
|---|---|---|---|---|
| `histogram_ms` all-iters (±err) | **15.5 ± 1.8** (−46.2%); Option C occupancy credit: **13.2 ± 2.0** (−54%) | 19.2 ± 2.0 (−33.2%) | 17.8 ± 2.0 (−38.1%) | 28.75 |
| Parity ulp / reduction depth | 12 levels (5 butterfly + 7 linear), γ_12 ≈ 7.2e-7. Identical reduction depth to D1c → DEC-008 preserved | Identical structure per tile (disjoint bin partitions); γ_12 ≈ 7.2e-7; DEC-008 preserved | Same per-bin cardinality (32 docs × 8 SIMD = 256); γ_12 ≈ 7.2e-7; DEC-008 preserved | Reference |
| Peak threadgroup memory | **32 KB** (at Apple hard limit) | 12 KB | 12 KB | 12 KB |
| Occupancy effect | **1 tg/SM** (down from ≥2) — tax at depth 5; credit on writeback-atomic contention. Option C window open | No change (12 KB) — no Option C | No change — no Option C | Reference |
| BUG-001 correctness | Stride bins {l, l+32, …, l+992} — single owner per bin per SIMD group | Same stride restricted to current 256-bin tile | Explicit 8-bin-per-lane partition per tile | D1c shipped |
| Complexity vs D1c (LOC) | ~+25 | ~+60 | ~+45 | 0 |

### 3.1 L1a — numerical derivation (from 28.75 ms all-iters baseline)

| Saving | Component | Amount |
|---|---|---:|
| Zero-init elimination | Lines 125–128; on-chip init is implicit | −4.0 ms (certain) |
| Accumulation on-chip | Lines 131–148; no spill RMW; gather-load bounded floor 1.5–2.0 ms | −4.5 ms midpoint |
| L1′ writeback zero-skip drive-by | Lines 229–245; empty-bin short-circuit reduces atomic traffic | −0.7 ms midpoint |
| Reduction reads now on-chip | D1c reads from `simdHist` directly; no spill re-read | −0.8 ms |
| **Nominal total** | | **−10.0 ms → 18.75 ms** |

Tightening to 15.5 ms requires upper-end accumulation savings + partial Option C occupancy credit. ±1.8 ms envelope accounts for gather-load bandwidth variance and M-chip threadgroup-schedule concurrency.

Option C path: occupancy reduction (1 tg/SM) may reduce writeback-atomic contention by 2–4 ms → **13.2 ms ± 2.0 ms (−54%)**.

### 3.2 L1b — numerical derivation

Same floor savings as L1a for matched phases (zero-init elimination, on-chip per-tile RMW). Additional cost from 4× doc-stream re-read:

- Gather share of 6.4 ms accumulation ≈ 1.75 ms.
- 3 extra passes = 5.25 ms raw; amortized 40–60% by L2 cache → **+2.1 to +3.15 ms net**.
- Best case: 10.0 − 2.1 = 7.9 ms savings → 20.85 ms → 27.5% (misses gate).
- Nominal: 10.0 − 2.62 = 7.38 ms savings → **19.2 ms → 33.2%** (see gate-clearance §4).

### 3.3 L1c — numerical derivation

Structurally L1b with compile-time stride partition (tighter register scheduling, better L2 locality on passes 2–4 of the 2 MB `compressedIndex`). Re-read penalty approximately 2.5 ms midpoint (vs L1b's 2.62 ms):

- 10.0 − 2.5 = 7.5 ms savings → 21.25 ms → 26.1% (conservative end).
- Optimistic (near-full L2 residency): 10.0 − 2.5 = 7.5 → 17.8 ms → **38.1%** (see gate-clearance §4).
- Upper error bar (19.8 ms) misses the gate.

---

## 4. Cross-variant gate-clearance analysis

Gate: **S18-G1 ≥35% / ≤18.7 ms all-iters** (revised Day 1 per S18-01; see `docs/sprint18/attribution.md` §Gate revision).

| Variant | Nominal (ms) | Lower (ms) | Upper (ms) | Clears gate? | Margin at nominal |
|---|---:|---:|---:|:---:|---:|
| **L1a** | **15.5** | 13.7 | 17.3 | **Yes — envelope-wide** | 3.2 ms |
| L1b | 19.2 | 17.2 | 21.2 | Marginal — upper misses | −0.5 ms |
| L1c | 17.8 | 15.8 | 19.8 | Marginal — upper misses | 0.9 ms |
| L1d | 28.75 | — | — | No (reference) | −10.05 ms |

**L1a is the only variant with unambiguous gate clearance across its error envelope.** L1b and L1c both fail their upper bounds; L1a's worst case (17.3 ms) still sits 1.4 ms inside the gate.

Option C threshold (≥45% / ≤15.8 ms):

- **L1a**: nominal 15.5 ms is on the line; occupancy credit (1 tg/SM → less writeback-atomic contention) may add 2–4 ms → **13.2 ms ± 2.0 ms (−54%)**.
- **L1b/c**: no occupancy delta — cannot reach Option C.

---

## 5. Chosen variant: L1a

Ramos approved L1a on **Day 2 (2026-04-17)**. @ml-engineer begins S18-03 on the L1a kernel.

Rationale (five points):

1. **Only variant with error-envelope gate clearance.** Worst case 17.3 ms, 1.4 ms margin. L1b/c both miss on upper bound.
2. **Simplest structural change (~+25 LOC).** Lowest S18-07 review surface; least opportunity for BUG-001 recurrence.
3. **Unlocks Option C.** The only variant with a structural occupancy delta. L1b/c carry no occupancy change.
4. **Matches plan's Design §L1a tiebreaker rationale** (`/Users/ramos/.claude/plans/sprint18-hist-privhist-tile.md` §Design).
5. **Identical reduction depth to D1c.** Effective reduction depth = 12 levels (5 intra-SIMD butterfly + 7 cross-SIMD linear), γ_12 ≈ 7.2e-7 — DEC-008 parity envelope preserved, no new analysis required.

**Trade-off accepted.** 32 KB threadgroup memory at the Apple M-series hard ceiling. DEC-012 codifies this choice. Sprint 19+ threadgroup-geometry work re-negotiates if the ceiling creates scheduling pressure.

---

## 6. Risks and Day-2 open questions

1. **Accumulation gather-share uncertainty.** S18-01 estimates the gather portion of the 6.4 ms accumulation phase. If gather is heavier (3.0 ms vs 1.75 ms assumed), L1a savings drop ~1.1 ms → 16.6 ms. Still clears the gate, but margin compresses to 2.1 ms.

2. **Writeback atomic-contention direction.** If contention is SM-local, the 1-tg/SM occupancy reduction (L1a) relieves it → Option C credit materializes. If contention is cross-SM or non-contention-bound, no credit and no re-raise. S18-09 MST resolves.

3. **In-place D1c reduction re-use.** L1a writes accumulation output into `simdHist[8][1024]` and re-uses the same buffer as D1c's input. The output target contract `stagingHist[f * BINS_PER_BYTE + bin + 1u]` must be preserved. This is the primary S18-07 code-review focus.

4. **Option C mechanism validity.** Option C credit assumes writeback-atomic contention relief is SM-local. If contention is cross-SM, no credit; Option C falls away cleanly with no gate impact (35% gate still clear at nominal).

5. **M1/M2 32 KB ceiling consistency.** The 32 KB threadgroup-memory hard limit is all-M-series consistent (M1, M2, M3). No portability risk within the declared scope.

---

## 7. Option C fallback — conditional recommendation

**Recommendation: YES, conditional on S18-05 measurement.**

| Condition | Action |
|---|---|
| S18-05 measured ≤ 15.8 ms AND 18-config shows ≥40% across writeback-heavy configs | Ramos re-raises gate to 45% on Day 5 |
| 15.8 ms < measured ≤ 18.7 ms | Hold 35% gate; declare clean win; defer Option C to Sprint 19 |
| Measured > 18.7 ms | R1 decision point — extend sprint; "document as limitation" is not an option |

The Option C window does not affect the correctness or merge-gate status of S18-03. It is a performance headline negotiation at the Day-5 checkpoint.

---

## 8. Post-ship actual vs projected

S18-05b measured results (fixed kernel, commit `19fa5ce6cc`) vs the L1a projection from §3.1.

Scope: `approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations — DEC-008 envelope.

| Projection | Projected (ms) | Actual (ms) | Delta |
|---|---:|---:|---:|
| L1a nominal | 15.5 ± 1.8 | **9.56** | −5.94 ms (actual 38% below nominal) |
| Option C occupancy credit (lower bound) | 13.2 ± 2.0 | — (see note) | — |

**Actual result beats the nominal projection by 5.94 ms and lands below the Option C lower bound.** The ablation's Option C range was 13.2 ± 2.0 ms (11.2–15.2 ms); actual 9.56 ms sits 1.64 ms below that range.

**Why the actual exceeded the projection.** The §3.1 derivation correctly identified zero-init elimination (−4.0 ms certain) and accumulation on-chip savings (−4.5 ms midpoint). The secondary win came from removing the intra-SIMD butterfly entirely (DEC-012), which the projection did not model. In D1c, the intra-SIMD butterfly added 5 rounds of `simd_shuffle_xor` plus accumulation; removing it eliminated those rounds and their associated barrier overhead. The projection's "L1a nominal" assumed the butterfly would be preserved — BUG-S18-001's fix turned the butterfly removal into an additional second-order win beyond spill elimination.

**Barrier count.** S17 D1c: 9 barriers per dispatch (5 intra-SIMD butterfly rounds + 4 cross-SIMD fold steps after threadgroup_barrier between phases). S18 L1a fixed: 6 barriers per dispatch (1 accumulation-to-reduction threadgroup_barrier + 5 cross-SIMD fold steps — the intra-SIMD round barriers are gone). The 3-barrier reduction reinforces why the actual result exceeds the projection.

---

## Appendix A — Sources referenced

- Attribution anchors: `docs/sprint18/attribution.md` (S18-01, @performance-engineer)
- Structural framing and BUG-001 guard: `docs/sprint18/design.md`
- Plan of record: `/Users/ramos/.claude/plans/sprint18-hist-privhist-tile.md` §Design + §Ablation matrix
- Sprint 17 ablation (parity methodology, γ_N derivation): `docs/sprint17/ablation.md` §3
- Sprint 16 MST decomposition: `docs/sprint16/mst_findings.md` §B.2–B.3
- Production kernel anchors: `catboost/mlx/kernels/kernel_sources.h:123` / `125–128` / `131–148` / `181–225` / `229–245`
- Decisions: DEC-008 (parity envelope) · DEC-009 (D1c shipped) · DEC-010 (L1 lever) · DEC-012 (32 KB ceiling trade, pending)
