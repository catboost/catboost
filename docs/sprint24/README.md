# Sprint 24 — DEC-023 Atomic-Float Race Fix + Championship Benchmark

**Branch**: TBD (cut from Sprint 23 tip)
**Campaign**: Operation Verstappen — battle 9 of 9
**Gate config**: 50k/RMSE/d6/128b (unchanged from S19–S23)
**Authority**: `DECISIONS.md DEC-023` (OPEN); `docs/sprint23/d0_bimodality_verification.md` (footprint + mechanism)

---

## §1 Background

Sprint 23 D0 (T2 scratch→production promotion) landed 4 commits and tripped the kill-switch at config #8 (N=10000/RMSE/128b): bimodal ~50/50 output between 0.48231599 and 0.48231912 (105 ULP gap). Verified pre-existing in S22 D2/D3 tip `73baadf445`. Root cause: features 1-3 `atomic_fetch_add_explicit(memory_order_relaxed)` on float is non-associative; Metal GPU thread scheduling produces 1-2 ULP histogram bin drift that cascades to 105 ULP at iters=50 for this dispatch shape.

Gate config #14 (N=50000/RMSE/128b) remains **100/100 deterministic** at 0.47740927. R8 1.90× record is unaffected.

Sprint 24 resolves DEC-023, then runs the championship benchmark suite.

---

## §2 D0 — DEC-023 Fix (Blocking)

**Goal**: Make all 18 DEC-008 configs deterministic across ≥5 runs (10/10 at config #8; 100/100 at gate config #14).

**Mechanism under repair**: T2-accum features 1-3 scatter path in `kT2AccumSource` (introduced S22, promoted S23):

```metal
// Current (non-deterministic): atomic float scatter
device atomic_float* dst = (device atomic_float*)(
    histogram + histBase + firstFold + b - 1u);
atomic_fetch_add_explicit(dst, s, memory_order_relaxed);
```

### Fix-option matrix

| Option | Mechanism | Deterministic | Perf risk | Notes |
|--------|-----------|:-------------:|:---------:|-------|
| **1 — TG-local reduce + single-thread commit** | Each TG accumulates feat 1-3 into threadgroup memory; single thread writes to global at kernel end | YES | Low — mirrors feat-0 design; no atomics in global | Preferred; known-clean pattern already proven at feat-0 |
| **2 — Int-atomic fixed-point** | `atomic_uint` with fixed-point encoding of gradient values; deterministic by integer arithmetic | YES | Medium — requires accuracy calibration; integer range must cover gradient magnitudes | CatBoost CPU uses uint64 fixed-point for this exact reason |
| **3 — Kahan/Neumaier compensated summation** | Running compensation term per bin | NO (reduces but does not eliminate non-determinism) | Low | NOT sufficient standalone; atomic order remains non-deterministic |

**Recommended**: Option 1. Rationale: feat-0's bin-range scan is 100/100 deterministic; replicating its structure for feats 1-3 unifies the accumulation design and eliminates atomics entirely from the inner loop.

### Kill-switch

Measure `hist_ms(T2_fixed) / hist_ms(T1)` at gate config post-fix. If ratio degrades below **0.45×** (optimistic band boundary), escalate to structural redesign rather than shipping a fix that degrades the 1.90× record. Current ratio is 0.317× — there is 13 pp of headroom before the 0.45× threshold.

### Acceptance criteria

1. Config #8: 10/10 deterministic (BENCH_FINAL_LOSS identical across 10 runs)
2. 18/18 DEC-008 parity sweep ULP=0
3. Gate config #14: 100/100 deterministic
4. `hist_ms` ratio ≥ gate (≤ 0.45× from above; expected near 0.317×)

### Budget

1-2 days. Per DEC-012: one atomic commit for the kernel fix, one for parity re-verify.

---

## §3 Championship Benchmark (Post-Fix)

Once DEC-023 is resolved and all 18 configs are deterministic, run the championship benchmark suite from `docs/operation-verstappen.md`:

- 10k/50k/500k regression (32-bin + 128-bin)
- 10k/50k binary classification
- 10k/50k multiclass K=3
- Compare vs CPU CatBoost, XGBoost, LightGBM
- Gate: MLX ≤ CPU CatBoost time on 50k+ benchmarks

Use the post-S23 production binary (tip `84529b47ed` + DEC-023 fix commit).

---

## §4 Carry-forward from Sprint 23

If S23-R1 (EvalAtBoundary readback elimination) or S23-R2 (dispatch inversion spike) are not completed in Sprint 23, they carry into Sprint 24 as secondary tasks after the DEC-023 fix:

- **S23-R1 carry**: Six `EvalAtBoundary` CPU readbacks in `structure_searcher.cpp` (~0.3 ms/iter compound gain). Bounded 0.5–1 day.
- **S23-R2 carry**: Dispatch inversion research spike (2-day timebox). If no concrete design surfaces, declare unreachable and close the campaign.

---

## §5 Exit Gates

| Gate | Criterion | Blocked on |
|------|-----------|-----------|
| S24-D0-G1 | Config #8: 10/10 deterministic post-fix | D0 fix commit |
| S24-D0-G2 | 18/18 ULP=0 parity sweep, ≥5 runs per config | D0 parity commit |
| S24-D0-G3 | Gate config #14: 100/100 deterministic | D0 parity commit |
| S24-D0-G4 | `hist_ms` ratio ≥ 0.45× (no kill-switch) | D0 parity commit |
| S24-BENCH-G1 | Championship suite complete; MLX ≤ CPU CatBoost on 50k+ | Post-fix benchmark |

**Parity sweep protocol** (standing order from S23 D0): minimum **5 runs per non-gate config**; gate config unconditionally **100 runs**. This is the new floor for all future sprint parity sweeps.

---

## §6 R8 Position

**Current (post-S22/S23)**: 1.90× cumulative. Verstappen ≥1.5× gate already cleared by 40 pp.

Sprint 24 DEC-023 fix is not expected to contribute additional R8 (the fix replaces non-deterministic atomics with an equivalent-cost deterministic path; Option 1 may be marginally faster due to fewer atomic operations). If Option 2 (int-atomic) is chosen, measure ratio before and after to confirm no regression.

**Do not inflate 1.90×.** This is the honest post-S22 position. Propagate unchanged into S24 closeout unless a new e2e measurement supersedes it.

---

## §7 D-Document Placeholders

| Doc | Status | Description |
|-----|--------|-------------|
| `d0_dec023_fix.md` | PENDING | DEC-023 kernel fix implementation + parity sweep results |
| `championship_benchmark.md` | PENDING | Full dominance suite results post-fix |
| `r1_evalatboundary.md` | PENDING (carry from S23 if not done) | EvalAtBoundary readback elimination |
| `r2_dispatch_inversion_spike.md` | PENDING (carry from S23 if not done) | Dispatch inversion research spike |
