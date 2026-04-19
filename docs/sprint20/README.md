# Sprint 20 — T3b Atomic-CAS L1a Accumulator, Parity-Sweep Gated

**Branch**: `mlx/sprint-20-hist-atomic-cas` (to be cut from Sprint 19 tip)  
**Campaign**: Operation Verstappen — battle 5 of 9  
**Lever**: T3b threadgroup-atomic-CAS accumulator (DEC-017 DRAFT-S20)  
**Gate config**: 50k/RMSE/d6/128b (unchanged from Sprint 19)  
**Projected e2e**: ~2.0–2.2× if D1–D3 pass; R8 reset to ≥2.0× at D4

---

## Background

Sprint 19 closed with three analytical model falsifications:

- **DEC-013 writeback** — premise falsified by S19-01 (writeback = 0.79 ms / 5%, not ~15 ms)
- **DEC-014 original gather hypothesis** — falsified by S19-01c probe D (global loads hidden by AGX, kernel is shuffle-bound)
- **DEC-015 col-major layout** — measured 0.98× vs 2.13× projected; AGX hides scatter cost behind shuffle inner loop

The correct bottleneck is the `simd_shuffle` serial broadcast chain at 86.2% of single-TG accumulation time (`docs/sprint19/reattribution.md`). Sprint 19 ships T1 (DEC-016, −3.6% accumulation, ~1.03× e2e) and DEC-014 (A1) BATCH_DOCS=64 (~1.04× e2e), stacked to ~1.07× e2e (honest). The gate for genuine shuffle elimination is Sprint 20.

T3b (threadgroup atomic-CAS no-shuffle accumulator) measured −84.4% single-TG accumulation at gate config in toy-kernel isolation (`docs/sprint19/algorithmic_ablation.md`). It eliminates the shuffle chain entirely. The single remaining unknown is whether FP32 reduction-order non-determinism causes drift outside the DEC-008 parity envelope — a cheap question to answer before any integration work begins.

---

## Tasks

### D1 — Full DEC-008 parity sweep against T3b toy-kernel-equivalent

Run the T3b toy-kernel (from `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp`, T3b variant) as a standalone harness across the full 18-config DEC-008 grid. Compare output histograms against the T0 production kernel at each config. Verify 100-run determinism (T3b produces consistent results across runs — atomic order within one dispatch should be stable, but must be confirmed).

**Pass criterion**: All 18 configs within DEC-008 envelope (RMSE/Logloss ulp ≤ 4, MultiClass ulp ≤ 8) across 100 runs.

**If pass**: Proceed to D2 immediately.

**If fail**: Apply Kahan compensated summation. Each bin gets a second `atomic_uint` slot holding the running Kahan correction term. The correction term is updated atomically alongside the sum, using the standard Kahan scheme adapted for CAS-float. TG memory rises from 4 KB to 8 KB — still well within the 32 KB DEC-011 ceiling (and DEC-011 is being relaxed to 4 KB for T3b anyway; 8 KB Kahan is still an 8× reduction from 32 KB). Re-sweep parity after Kahan integration.

**Why D1 before D2**: Sprint 19 falsified two analytical models that led to ~2 days of wasted integration effort (DEC-015). Parity sweep is ~0.5 days and kills the integration entirely if it fails. The correct order is cheap kill before sunk cost.

### D2 — T3b integration into production kernel

Ground-up rewrite of the L1a accumulator section in `catboost/mlx/kernels/kernel_sources.h`. The accumulation loop (lines 175–209 at Sprint 19 tip) is replaced with the T3b per-thread per-doc CAS-float loop. The cross-SIMD fold section (lines 224–238) is eliminated — T3b writes directly to `simdHistU[0][bin]` which is the single shared accumulator. The writeback section reads `atomic_load_explicit` on `simdHistU` instead of `simdHist[0]`.

Corresponding changes in `catboost/mlx/methods/histogram.cpp`: update TG memory size assertion and DEC-011 note from 32 KB → 4 KB (or 8 KB if Kahan is applied). Document the DEC-011 relaxation in DECISIONS.md (amend DEC-011 status to note the T3b variant's new ceiling).

Per DEC-012: one structural change per commit. D2 ships as a single commit after D1 parity sweep confirms it is safe.

### D3 — Full-grid scaling validation

The T3b toy-kernel measured 1 TG × 256 threads processing all 50k docs in a single partition (depth-0 equivalent). Production dispatches 1575 TGs concurrently at the gate config (25 feature groups × 63 partitions at depth 6). Whether T3b's TG-atomic speedup holds under concurrent multi-TG dispatch is unmeasured.

Run `bench_boosting` with T3b kernel at gate config. Capture `histogram_ms` steady-state across 50 iterations. Confirm that the per-TG speedup from toy-kernel isolation (−84.4%) translates to a comparable reduction in steady-state `histogram_ms`. If the multi-TG dispatch introduces cross-TG TG-memory contention (unlikely — TG memory is private per TG on AGX) or scheduling overhead that compresses the speedup, characterize the compression factor.

Expected: TG-atomic is private within each TG (no cross-TG atomic contention at any point in T3b). Scaling to 1575 TGs should produce near-linear improvement. If unexpected regression appears, it is likely from Metal command-buffer scheduling overhead per TG, not from the accumulation itself.

**Pass criterion**: `histogram_ms` ≤ 4 ms at gate config (−74% from 15.43 ms baseline). This is the conservative target corresponding to ~80% of the toy-kernel speedup being realized.

### D4 — R8 reset and projection update

If D1–D3 all pass:

- Reset R8 to **≥2.0× e2e** at 50k/RMSE/d6/128b.
- Projected: `histogram_ms` ~2.5–4.0 ms (from 15.43 ms baseline, −74% to −84%) → `iter_total_ms` ~8–12 ms (from 21.03 ms baseline) → e2e 1.75–2.6×. Midpoint ~2.1×.
- Update `docs/sprint20/results.md` (to be created at D3 measurement) with measured gate values.
- Update DECISIONS.md DEC-017 status from DRAFT-S20 to ACTIVE.

If D1 fails and Kahan mitigation is applied, re-run D3 with Kahan T3b and reassess the speedup (Kahan adds ~2 CAS ops per doc per bin = ~25% overhead vs bare T3b, compressing the improvement to ~−70% accumulation). R8 target in the Kahan case: ≥1.7× e2e.

---

## Risks

**Third-iteration failure mode (sprint 19 pattern):** Sprint 19 falsified two analytical models before landing on a working intervention. If T3b fails parity (D1) and Kahan fails parity, the correct response is to escalate and re-attribute — not to add more compensating layers. The Kahan mitigation is one additional attempt with a solid theoretical basis (Kahan is guaranteed to reduce error for this summation structure). If it also fails, the bottleneck is likely a deeper numerical issue with FP32 atomic accumulation that needs a different approach (e.g. FP64 intermediate accumulation in TG memory, then cast to FP32 at writeback).

**MultiClass approxDim=3 parity drift is the highest-risk config.** MultiClass composes three independent reductions per gradient/hessian. The reduction-order non-determinism of T3b's CAS loop multiplies across dimensions. D1 must include all three MultiClass configs from the DEC-008 grid. If any MultiClass config fails while RMSE/Logloss passes, the mitigation scope is narrowed to MultiClass only (conditional Kahan per approxDim > 1).

**Contention at 32-bin and 16-bin configs.** The contention sweep showed T3b/T0 = 0.218 at 16 bins (3125 docs/bin). At 32 bins (1562 docs/bin) the ratio is 0.165. Both are still substantial speedups. However, the CAS retry rate rises with contention. At very high contention (e.g. future configs with N=1M and 16 bins) the CAS loop may serialize. Gate config is 128 bins — low contention. D3 validation is at gate config only; large-N extension is Sprint 23 scope.

**Full-grid scaling is the key unmeasured risk.** 1575 TGs concurrent dispatch at depth 5–6 with T3b may introduce Metal scheduler overhead not visible in single-TG probes. D3 is the measurement that proves or disproves this. Do not skip D3 or defer it — it is the cheapest way to discover a scheduling bottleneck before the PR review.

---

## Exit gate

| Gate | Criterion |
|------|-----------|
| G1 | `histogram_ms` ≤ 4 ms on 50k/RMSE/128b |
| G2 | No 18-config regression >5% |
| G3 | Parity 108/108 bit-exact across DEC-008 envelope (or Kahan-corrected within envelope) |
| G4 | `iter_total_ms` ≤ 10.5 ms on 50k/RMSE/128b (≥2.0× e2e) |
| G5 | No non-histogram stage regresses >10% |
| G6 | CI green |

---

## Carry-forward from Sprint 19

- DEC-014 (A1) BATCH_DOCS=64 ships Sprint 19 (~1.04× e2e). T3b replaces (not stacks on) the accumulation loop — (A1)'s wider-batch register state must be removed when T3b lands, or the two are composed (T3b with BATCH_DOCS=64 would give the same T3b speedup since T3b is independent of batch size, but the BATCH_DOCS=64 register doubling adds complexity for no gain in a CAS accumulator). Recommendation: T3b Sprint 20 ships without (A1) stacking — the shuffle chain is eliminated entirely so BATCH_DOCS has no effect on the dominant cost.
- DEC-016 (T1 fuse-valid) ships Sprint 19 and is orthogonal to T3b. T3b does not use `simd_shuffle` at all, so T1's MSB-sentinel fusion is irrelevant once T3b integrates. T1 can be removed in the T3b commit to reduce LOC, or retained as dead code that the compiler eliminates. Either approach is acceptable; document the choice.
- S19-11 (EvalAtBoundary readback removals in `structure_searcher.cpp`) carries forward if not completed in Sprint 19. Independent of the histogram kernel changes.
