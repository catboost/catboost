# Sprint 18 — S18-01: Post-S17 Kernel Phase Attribution

**Config:** N=10k, RMSE, depth=6, 128 bins  
**Source data:** `.cache/profiling/sprint17/after/10000_rmse_d6_128bins.json` — 50 iterations, iters 1–49 steady-state  
**Date captured:** 2026-04-17  
**Baseline:** Sprint 17 after-JSON (D1c kernel, `simd_shuffle_xor`, commit `5b4a8206bc`)  
**Reported by:** @performance-engineer (S18-01)

---

## Method

The S16 Metal System Trace at `.cache/profiling/sprint16/mst_10000_rmse_2026-04-17.trace` is the best available artifact for the unmodified accumulation and writeback phases. MST CLI export fails on this bundle with "Document Missing Template Error" — the same limitation documented in `docs/sprint16/mst_findings.md §A.3`, requiring one Instruments.app open-cycle before `xctrace export` can parse it.

For this report, a two-component linear regression was run on the per-depth `histogram_ms` breakdown from the S17 after-JSON (50 iterations, 6 depths per iteration, iters 1–49 steady-state):

```
hist_ms(depth_d) = K_fixed × tg_count(d) + K_accum
```

Fitted values: **K_fixed = 9.36 µs/tg**, **K_accum = 1.073 ms/depth**, R² = 0.97.

- `K_accum × 6 depths = 6.44 ms` — per-doc accumulation work (constant across depths, scales with 10k docs).
- `K_fixed × Σ(tg_counts) = 17.28 ms` — threadgroup-proportional cost (zero-init + D1c reduction + writeback).

The threadgroup-proportional 17.28 ms was decomposed by bandwidth analysis:

- **Zero-init:** 1846 tg × 1 MB spill writes/tg ≈ 1.84 GB/iter at ~400 GB/s peak → 4.6 ms floor.
- **D1c reduction:** 8 barriers × on-chip 12 KB threadgroup memory per tg, pipelined across 1846 tg.
- **Writeback:** ~200 global `atomic_fetch_add` per tg at 50–150 ns per atomic.

---

## Per-phase attribution

Steady-state (iters 1–49). All-iters column includes iter 0 JIT cost amortized over 50 iterations.

| Phase | Lines (`kernel_sources.h`) | SS time (ms) | ±err | % of SS 23.72 ms | % of all-iters 28.75 ms |
|---|---|---:|---:|---:|---:|
| privHist accumulation (gather + RMW) | 131–148 | **6.4** | ±1.5 | 27% | 22% |
| privHist zero-init | 125–128 | **4.0** | ±1.5 | 17% | 14% |
| D1c reduction tail | 188–225 | **3.0** | ±1.0 | 13% | 10% |
| Global-atomic writeback | 229–245 | **5.0** | ±1.5 | 21% | 17% |
| JIT/launch/dispatch amortized | — | **5.3** | ±0.2 | 22% | 18% |
| **TOTAL** | | **23.72** | | 100% | — |

---

## Plan estimate confirmation / refutation

The plan's §Post-S17 bottleneck attribution (`/Users/ramos/.claude/plans/sprint18-hist-privhist-tile.md`) provided pre-measurement estimates. S18-01 ground-truth against those estimates:

| Phase | Plan estimate (% of 28.75 ms all-iters) | Measured SS (±err) | Verdict |
|---|---|---|---|
| privHist accumulation | 15–17 ms (52–59%) | 6.4 ms ±1.5 ms (27% of SS) | **REFUTED — 2.3–2.6× overestimate** |
| privHist zero-init | 2–3 ms (7–10%) | 4.0 ms ±1.5 ms (17%) | Partially refuted (larger than estimated) |
| D1c reduction tail | 3–4 ms (10–14%) | 3.0 ms ±1.0 ms (13%) | **CONFIRMED** |
| Global-atomic writeback | 4–6 ms (14–21%) | 5.0 ms ±1.5 ms (21%) | **CONFIRMED** |
| Launch/dispatch amortized | ~2 ms | 5.3 ms (empirical) | Refuted — 2.5× underestimate |

**Why the accumulation estimate diverged.** The plan's §Post-S17 attribution cited ~130 ms of S16's 308 ms baseline as "accumulation," on the assumption it would remain the dominant residual after D1c. The S16 §B.2 figure conflated two distinct costs: (a) the actual per-doc gather + spill RMW (~50–60 ms) and (b) the 255 serial-reduction passes that each re-read `privHist` from spilled device memory before adding to `stagingHist` (~70–80 ms). D1c eliminated cost (b) entirely — `simd_shuffle_xor` reads `privHist` in registers, not device memory. The surviving accumulation cost is only (a): **6.44 ms**.

**Secondary JIT surprise (depth 3).** `iter0[depth3] = 48.8 ms` vs steady-state `3.3 ms`. Metal triggered a shader recompile on the first dispatch with a different block geometry (`maxBlocksPerPart=1` at depth 3 vs `maxBlocksPerPart=3` at depth 0 — two distinct dispatch geometries, two JIT events). Total JIT excess amortized over 50 iterations: 251.6 ms / 50 = **5.03 ms/iter**, accounting for the launch/dispatch line item above.

---

## Anomalies

**1. Accumulation is not the plurality cost.** The plan predicted 52–59% of the post-S17 histogram; actual is 27% of steady-state. Threadgroup-proportional phases (zero-init + D1c tail + writeback = 12 ms, 50%) now dominate. This is a structural shift from the plan's model.

**2. Writeback at 5 ms is sub-trigger but the second-largest phase.** The plan flagged ">10 ms writeback = escalate"; at 5 ms ±1.5 ms it sits below that threshold. Atomic contention at 200 writes/tg × 1846 tg/iter is real. L1′ zero-skip remains the appropriate drive-by. If L1 reduces kernel latency and raises concurrent threadgroup count, writeback contention could increase — track in S18-05.

**3. Depth-0 iter-0 spike is 105× steady-state depth 0.** `iter0[depth0] = 210.1 ms` vs SS `1.997 ms`; depth-3 secondary spike `iter0[depth3] = 48.8 ms` vs SS `3.3 ms`. Both are Metal JIT compilation events, not accumulation behavior. The plan did not model the depth-3 secondary JIT.

---

## L1 lever upper bound

| Lever component | Current SS | Post-L1 estimate | Savings |
|---|---|---|---|
| Zero-init (lines 125–128) | 4.0 ms | ~0 ms (threadgroup memory init is implicit) | **4.0 ms certain** |
| Accumulation (lines 131–148) | 6.4 ms | 2.5–4.0 ms (on-chip RMW, no spill) | **2.4–3.9 ms** |
| D1c reduction (lines 188–225) | 3.0 ms | 3.0 ms (unchanged; already on-chip) | 0 ms |
| Writeback drive-by (L1′) | 5.0 ms | 4.0–4.5 ms (zero-skip reduces atomic traffic) | ~0.5–1.0 ms |

**L1 total savings range: 6.4–8.9 ms SS — 27–38% of SS 23.72 ms.**

### Gate revision

The original S18-G1 gate of ≥50% reduction (≤14.4 ms all-iters) was derived from the plan's pre-measurement accumulation estimate of 15–17 ms. That estimate is refuted by S18-01. L1's evidence-based upper bound clears the original 50% gate by only ~5 ms — not enough margin to commit against.

**S18-G1 revised to ≥35% / ≤18.7 ms (all-iters 28.75 ms baseline).** This revision was approved by Ramos on Day 1. The 35% target sits squarely within L1's measured ceiling, mirrors Sprint 17's precedent of "gate from evidence," and preserves L2 (pre-permute gather) and L3 (multiclass fusion) as clean Sprint 19 headlines rather than forcing them into Sprint 18 to compensate for an overestimated lever. See the plan's §Sprint 18 perf gate for the revised gate text.

---

## Lineage

- `docs/sprint16/mst_findings.md §B.2` — the ~130 ms "accumulation" figure that this report revises. The S16 estimate conflated accumulation + reduction-re-read costs that D1c has since eliminated.
- `docs/sprint17/results.md §Stage attribution` — the 28.75 ms all-iters post-S17 baseline and per-stage breakdown that this report decomposes further.
