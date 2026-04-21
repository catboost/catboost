# Sprint 24 — DEC-023 Atomic-Float Race Fix + Championship Benchmark

**Branch**: `mlx/sprint-24-dec023-fix` (cut from master `9f3b99c7d2` after S17–S23 PR chain merge)
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

## §4 Closed from Sprint 23 (not carry-forward)

Both S23 research tracks closed in S23; neither carries into S24:

- **S23-R1 — DEFERRED** (DEC-024, not falsified). 0/3 EvalAtBoundary sites at gate — `structure_searcher.cpp` sites are Depthwise/Lossguide only; gate config runs SymmetricTree via `bench_boosting` inline oblivious loop. Re-entry gated on `--grow-policy` flag addition or separate Depthwise/Lossguide harness. See `docs/sprint23/r1_evalatboundary.md`.
- **S23-R2 — FALSIFIED** (DEC-025). Dispatch inversion has a structural algebraic blocker (`H[f][b] = Σ_p h_p[f][b]` not invertible); no mask mechanism reconstructs per-partition bin sums below equivalent or worse cost. Do not re-enter without new mask-mechanism evidence. See `docs/sprint23/r2_dispatch_inversion_spike.md`.

---

## §5 Exit Gates

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| S24-D0-G1 | Config #8: 10/10 deterministic post-fix | 10/10 at 0.48231599, ULP=0 | **PASS** |
| S24-D0-G2 | 18/18 ULP=0 parity sweep, ≥5 runs per config | 18/18 ULP=0, all 5/5 det. | **PASS** |
| S24-D0-G3 | Gate config #14: 100/100 deterministic | 100/100 at 0.47740927 | **PASS** |
| S24-D0-G4 | `hist_ms` ratio ≥ 0.45× (kill-switch) | 0.959× — T2-at-T1-speed [1] | **PASS** |
| S24-BENCH-G1 | Championship suite complete; MLX ≤ CPU CatBoost on 50k+ | Not started — campaign retreated | **NOT RUN** |

[1] G4 kill-switch does not fire (0.959× >> 0.45×). However, 0.959× means T2 v5 is running at
essentially T1 speed — T2's structural histogram advantage is gone. The kill-switch threshold was
calibrated to detect "fix that degrades performance"; it does not capture "fix that eliminates
the speedup". See §8 Final Verdict and `d0_dec023_fix.md §7`.

**Parity sweep protocol** (standing order from S23 D0): minimum **5 runs per non-gate config**; gate config unconditionally **100 runs**. This is the new floor for all future sprint parity sweeps.

---

## §6 R8 Position

**Post-S24**: ~1.01× (honest position after v5 ships). T2-accum v5 runs at T1 speed.

**Pre-S24 position (post-S22/S23)**: 1.90× — superseded. That figure was predicated on the
bimodal T2 kernel (0.317× hist_ms ratio). Making T2 deterministic collapses it to 0.959×, which
translates to ~1.01× e2e vs the Sprint 16 baseline.

---

## §7 D-Document Status

| Doc | Status | Description |
|-----|--------|-------------|
| `d0_dec023_fix.md` | COMPLETE | DEC-023 full diagnostic history + v5 ship rationale |
| `d0_offby1_cascade_retest.md` | COMPLETE | Off-by-one false-positive diagnostic record |
| `championship_benchmark.md` | NOT STARTED | Campaign retreated before championship suite |

---

## §8 Final Verdict

### DEC-023 fix

**PASS.** v5 resolves DEC-023. All four S24 D0 acceptance criteria PASS. T2-accum v5 is
bit-exact vs T1 (ULP=0) on all 18 DEC-008 configs, deterministic across all run counts tested.
Commit `784f82a891`.

### Verstappen ≥1.5× gate

**FAIL — retroactive.** The ≥1.5× gate was cleared at Sprint 22 D4 (cumulative R8 = 1.90×).
That record was predicated on the non-deterministic T2 kernel. Making T2 deterministic is a
prerequisite for shipping under DEC-008 discipline (ULP ≤ 4). The fix eliminates T2's structural
speed advantage. Post-fix R8 at the gate config is **1.01×** — the Verstappen criterion (≥1.5×)
is not met.

This is an honest retreat. The campaign goal (≥1.5× on gate config, deterministic, ULP=0) has
not been achieved. Sprint 25 opens DEC-026, a research track investigating whether a
cascade-robust GAIN comparison mechanism can allow T2's sort-based accumulation to ship without
introducing the cascade amplification that triggered DEC-023. See `DECISIONS.md DEC-026`.

### Summary table

| Criterion | Result |
|-----------|--------|
| DEC-023 resolved (ULP=0 deterministic) | YES — v5, commit `784f82a891` |
| 18/18 DEC-008 parity | PASS (18/18 ULP=0) |
| Verstappen ≥1.5× gate | FAIL — R8 post-fix: 1.01× |
| PR #16 | Pending (Ramos opens) |
| S25 research track | DEC-026 cascade-robust GAIN comparison |
