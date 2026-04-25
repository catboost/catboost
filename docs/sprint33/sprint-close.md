# Sprint 33 Close — Iter≥2 Scaffold + DEC-036 Resolution (DEC-040 / DEC-042)

**Date**: 2026-04-25
**Branch**: `mlx/sprint-33-iter2-scaffold`
**Base**: mlx/sprint-32-cosine-gain-term-audit tip `9fcc9827d9`
**Tip at close**: `c511549eeb`
**Kernel md5**: `9edaef45b99b9db3e2717da93800e76f` (unchanged throughout S33)

---

## Executive Summary

Sprint 33 opened a layered L0–L4 scaffold (DEC-040) to close DEC-036: the 52.6% ST+Cosine
iter=50 RMSE drift that survived five prior sprints and two full precision-fix campaigns.
The sprint ran through a false-close, two invalidating probes, three further probes, the
correct fix, and guard removal — a complete investigation arc in a single branch.

**DEC-036 is fully resolved.** The mechanism is a one-line degenerate-child skip at
`csv_train.cpp:1980`: `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` drops
both sides of a partition from the joint Cosine numerator/denominator. CPU's reference path
(`UpdateScoreBinKernelPlain`) adds the non-empty side and zeros only the empty side. For any
split candidate whose tree ancestry creates degenerate children in a subset of partitions,
MLX under-scores the candidate by 20+ gain units — enough to flip depth=2 decisions from
signal features to noise at every iteration, producing runaway divergence.

The per-side mask fix in Commits 1 + 1.5 collapsed iter=50 ST+Cosine drift from **52.6%
to 0.027%** (1941× improvement). Both Cosine guards (#93 S28-ST-GUARD, #94 S28-LG-GUARD)
were removed in Commits 3a and 3b. DEC-032 is now fully closed; all three grow policies
support Cosine.

---

## Sprint Goals

From DEC-040 (ultrathink kickoff):

1. Execute L0–L4 layered scaffold to close DEC-036 structural divergence.
2. Identify the iter≥2 per-iteration mechanism driving 52.6% drift.
3. Implement the fix; validate with formal five-gate suite G4a–G4e.
4. Remove both Cosine guards (#93 ST-REMOVE, #94 LG-REMOVE) once parity passes.

---

## Investigation Arc

The sprint traversed a false summit before finding the real mechanism. The full arc is
preserved as written record; see section "RETRACTION (2026-04-24)" in the pre-retraction
file and the RETRACTION note appended to the original sprint-close.

### Phase 1: L0–L3 scaffold (DEC-040 tasks)

| Layer | Task | Verdict | Surviving? |
|-------|------|---------|------------|
| L0 | CONFIG AUDIT (#119) | NO-DIFF — Frame C falsified | yes |
| L1 | DETERMINISM (#120) | FALSIFIED — drift seed-independent (52.643%, 3 seeds) | yes |
| L2 | GRAFT (#121) | FRAME-B — graft ratio 0.974; per-iter mechanism confirmed | yes |
| L3 | ITER=2 INSTRUMENT (#122) | SPLIT — S1-grad bit-identical; S2-split divergent at depth=2 (not depth=0 as first read) | partially superseded |

L3's initial depth=0 divergence reading was a coordinate-system error: CPU's `split_index=3`
is a CBM-stored compressed index; MLX's `bin=64` is an upfront-grid index. Both physically
map to border 0.014169 (ULP-identical). The real divergence is at **depth=2** of iter=2.

### Phase 2: L4 false close (2026-04-24)

L4 concluded that the mechanism was "static vs dynamic quantization" in `csv_train.cpp`:
- Commit to "L4 CLOSED" state and sprint-close written.
- DEC-041 opened ("redesign quantization pipeline").

### Phase 3: Retraction via PROBE-A and PROBE-B

Two independent probes falsified the L4 conclusion within the same session:

**PROBE-A** (`c770ab6630`, `docs/sprint33/probe-a-borders/`): `Pool.quantize` on the
anchor dataset produces 128 borders × 20 features = 2560 — identical to csv_train.cpp's
static grid. The "95/71/0" figures cited in L4 are *stored-in-CBM* borders (serialization
prunes to thresholds referenced by the saved trees), not available borders. There is no
dynamic border accumulation mechanism in CatBoost. L4's primary claim is false.

**PROBE-B** (`600238f39f`, `docs/sprint33/probe-b-python/`): Tracing the nanobind Python
path (`core.py:1090 → train_api.cpp:14 #include csv_train.cpp → QuantizeFeatures`) shows
the Python path calls the same `QuantizeFeatures` as the CLI. Measured Python-path drift:
**52.64%** — matches cli-path drift to four significant figures. L4's "production path is
fine" rationale (Option 3) is structurally invalid.

**Reverted state**: DEC-036 OPEN, DEC-040 OPEN, DEC-041 INVALIDATED, S33 OPEN.

### Phase 4: PROBE-C — border and tree structure (2026-04-24)

`docs/sprint33/probe-c-borders/`

Stage 1: Per-feature border comparison. MLX's 127-border grid is a strict subset of CPU's
128 (ULP=1); each feature is missing exactly one CPU border (at index 6, e.g. feat=0
missing −1.65587). The deficit is the DEC-039 cap-127 truncation, not a value divergence.

Stage 2: Full tree[1] depth-by-depth comparison.
- Depth=0: AGREE — both pick feat=0, border 0.014169 (ULP-identical).
- Depth=1: AGREE — both pick feat=1 at the equivalent logical position (4.6e-6 absolute
  border drift from the 127 vs 128 grid, but same feature).
- **Depth=2: DIVERGE** — CPU picks feat=0 (border=−0.947), MLX picks feat=10 (border=+0.306).
  Given y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, feat=10 is pure noise; CPU stays on signal,
  MLX prefers noise. Mechanism is in the Cosine gain argmax at iter=2 **depth=2**.

### Phase 5: PROBE-D — precision class closure (2026-04-24)

`docs/sprint33/probe-d/FINDING.md`

fp32-vs-fp64 double-shadow gain dump at iter=2 d=0..5, 2540 (feature, bin) cells per depth.

- Max |gain_f32 − gain_f64| = **3.89e-5** across all depths and cells.
- At every depth: argmax(gain_f32) == argmax(gain_f64) bit-for-bit.
- No fp-widening change can flip the d=2 winner.

**Smoking gun — signal/noise inversion at d=2:**

| Feature class | d=2 gain (MLX fp32) |
|---|---|
| 18 noise features | ~101.95 (range 101.946–101.954) |
| feat=0 (signal, CPU's pick) | 81.89 |
| feat=1 (signal) | 77.77 |

At depth=0 (where CPU and MLX agree), feat=0 scores 87.18 and noise features score
0.58–2.30 — signal is 30–80× higher than noise. At depth=2, this is fully inverted.
Gap between MLX's winner (feat=10, 101.95) and CPU's pick (feat=0, 81.89) = 20.07 gain
units — three orders of magnitude larger than the fp32 noise floor. Precision class closed.

PROBE-D elevated the partition-state class to lead candidate: CPU's d=2 pick (feat=0,
bin=21) creates degenerate (empty-child) splits in 2 of 4 leaves because d=0 already
split on feat=0. Noise features split all 4 leaves non-trivially.

### Phase 6: PROBE-E — partition-state class confirmed, mechanism named (2026-04-24)

`docs/sprint33/probe-e/FINDING.md`, DEC-042 opened.

Per-(feat, bin, partition) capture of the joint cosNum/cosDen contribution under both
MLX's actual rule and CPU's counterfactual, at iter=2 d=0..5.

**Mechanism at `csv_train.cpp:1980`:**

```cpp
// MLX (wrong)
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
```

This skips the entire partition — both cosNum and cosDen contributions go to zero.

CPU's reference (`catboost/libs/helpers/short_vector_ops.h:155+`, SSE2
`UpdateScoreBinKernelPlain`) computes `average = mask · sum / (w + λ)` where the mask
zeros the empty side's average only. The non-empty side's contribution is always added.

**Per-partition smoking gun — (feat=0, bin=21) at iter=2 d=2:**

| part | wL | wR | skip | mlxTermN | cpuTermN |
|---|---|---|---|---|---|
| 0 | 4489 | 7854 | no  | 5718.05 | 5718.05 |
| 1 | **0** | 13828 | **YES** | 0.00 | 274.81 |
| 2 | 4104 | 6990 | no  | 973.74  | 973.74  |
| 3 | **0** | 12735 | **YES** | 0.00 | 4761.33 |

- MLX sum: cosNum = 6691.79, cosDen = 6687.72 → gain = **81.83**
- CPU sum: cosNum = 11727.93, cosDen = 11722.68 → gain = **108.32**
- Gap = +26.49 gain units — exactly enough to flip d=2 pick from feat=10 (noise, 101.79)
  to feat=0 (signal, 108.32).

Skip rate grows monotonically with depth: 0% / 2.5% / 5.0% / 7.6% / 10.6% / 14.6% at
d=0..5. All 127 non-trivial bins on feat=0 at d=2 have skips=2 — a structural consequence
of d=0 having already split on feat=0.

### Phase 7: S33-L4-FIX — commits 1, 1.5, 2, 3a, 3b

| Commit | SHA | Tag | Description |
|--------|-----|-----|-------------|
| 1 | `10c72b4e96` | S33-L4-FIX-1 | Per-side mask — Cosine path in FindBestSplit |
| 1.5 | `e98c6725cd` | S33-L4-FIX-1.5 | Per-side mask — L2 path (symmetric fix) |
| 2 | `dd778b0f7d` | S33-L4-FIX-2 | Four-gate parity validation (PASS) |
| 3a | `e1d72d64e8` | S33-L4-FIX-3a | S28-ST-GUARD removed (#93) |
| 3b | `d599e5b033` | S33-L4-FIX-3b | S28-LG-GUARD removed (#94) |

Also in scope:
| `f4664f0322` | PROBE-E | Partition-state class confirmed; DEC-042 opened |
| `fad1c3a08d` | PROBE-C-S3 | Stage 3 coordinate-system correction |
| `d246e00fae` | PROBE-D | Precision class closed |

---

## Gate Results

From `docs/sprint33/commit2-gates/REPORT.md`:

| Gate | Criterion | Pre-fix | Post-fix | Verdict |
|------|-----------|---------|---------|---------|
| G4a | iter=1 ST+Cosine drift ≤ 0.1% | ~0% (d=0 trivial) | **0.0001%** | **PASS** |
| G4b | iter=50 ST+Cosine drift ≤ 2% | **52.6%** | **0.027%** (1941× improvement) | **PASS** |
| G4c | v5 kernel ULP=0 | `0.48231599` | `0.48231599` | **PASS** |
| G4d | 18-config L2 parity [0.9800, 1.0200] | 18/18 [0.9991, 1.0008] | 18/18 [0.9991, 1.0008] | **PASS** |
| G4e | DW+Cosine sanity 5 seeds [0.98, 1.02] | 5/5 PASS | 5/5 PASS | **PASS** |

LG+Cosine post-fix measurement (Commit 3b validation): iter=1 drift = 0.0000%, iter=50
drift = **0.382%**. Below the 2% gate. Guard removal unblocked.

---

## What Shipped

| Item | Description |
|------|-------------|
| Per-side mask fix (Cosine) | `csv_train.cpp` FindBestSplit: replace whole-partition skip with per-side zero contribution; empty side contributes 0 but non-empty side is always added. |
| Per-side mask fix (L2) | Symmetric application of the same pattern to the L2 branch of FindBestSplit. |
| S28-ST-GUARD removed | `python/catboost_mlx/core.py` `_validate_params` guard for ST+Cosine replaced by acceptance. Guard tests inverted to positive assertions. |
| S28-LG-GUARD removed | Same for LG+Cosine. `train_api.cpp` and `csv_train.cpp` guard blocks also cleared. |
| DEC-042 opened and closed | Mechanism documented; fix plan recorded; all five gates PASS. |
| DEC-036 resolved | Root cause confirmed by PROBE-E smoking gun; fix closes four-sprint-old 52.6% drift. |
| DEC-040 closed | L0–L4 scaffold complete. |
| DEC-041 invalidated | Dead number; premise falsified by PROBE-A. Do not reuse. |
| State-update commit `c511549eeb` | HANDOFF, DECISIONS.md DEC-042 fully closed; S33 CLOSED. |

---

## What Did NOT Ship

Nothing material was deferred. The open carry-forward items below are independent of
DEC-036 and were never part of S33 scope.

---

## DEC Status After S33

| DEC | Status | Notes |
|-----|--------|-------|
| DEC-032 | **FULLY CLOSED** | All three grow policies now support Cosine; guards removed; S28-ST + S28-LG guards gone. |
| DEC-036 | **RESOLVED** | Mechanism: degenerate-child `continue` at `csv_train.cpp:1980`. Per-side mask fix closes drift 52.6% → 0.027%. |
| DEC-040 | **CLOSED** | L0–L4 scaffold complete. |
| DEC-041 | **INVALIDATED** | Static-vs-dynamic quantization premise falsified by PROBE-A. Number is dead; do not reuse. |
| DEC-042 | **FULLY CLOSED** | PROBE-E mechanism confirmed; fix + gates PASS; guards removed. |

---

## Carry-Forwards to S34

None of these items are load-bearing for DEC-036. All are independent.

| Item | Status | Notes |
|------|--------|-------|
| S31-T-LATENT-P11 | Carry | Hessian-vs-sampleWeight swap at `csv_train.cpp:3780, 3967`. Fires under Logloss/Poisson/Tweedie/Multiclass. Not blocking RMSE path. |
| #113 S31-T3-MEASURE re-run | Pending | iter=1 G3a/G3b gate matrix re-run against current binary. Blocked on clean branch. |
| #114 S31-T-CLEANUP | Carry | SA-I2 + S29 CR nits. |
| DEC-035 K4 / DEC-038 / DEC-039 | Standing | Independent precision items; remain correct and in v5. Not load-bearing for DEC-036. |
| LG+Cosine deep cells | No action | Post-fix 0.382% at iter=50 is within envelope. No deep-cell follow-up opened. |

---

## Lessons Learned

**1. Counterfactual capture as a debugging primitive.** PROBE-E ran MLX's actual
computation alongside a counterfactual CPU-rule shadow in the same binary pass — same
inputs, two implementations side-by-side. This produced a per-partition smoking gun with
exact gain deltas, requiring no interpretation. When a precision-class explanation is
exhausted (PROBE-D), the right next instrument is a counterfactual capture of the
alternative rule at the exact locus, not a new statistical measurement.

**2. Instrumentation gated under compile-time flags ships safely.** Every S33 probe binary
was built with `-DCOSINE_RESIDUAL_INSTRUMENT -DPROBE_E_INSTRUMENT -DPROBE_D_ARM_AT_ITER=1`
etc. The production binary and the kernel sources were untouched throughout — md5 invariant
held across all commits. Compile-time instrumentation gates mean probe complexity does not
touch any code path that ships.

**3. Sibling-commit pattern for symmetric fixes.** Commit 1 (Cosine path) and Commit 1.5
(L2 path) are separate atomic commits applying the same per-side mask pattern to two parallel
branches of FindBestSplit. This separates the causal fix (Commit 1) from the correctness
completeness sweep (Commit 1.5), makes each independently bisectable, and makes the
symmetry argument explicit in the commit log. When CPU uses a single shared reference path
for two score functions, both MLX paths must receive the same treatment — the sibling-commit
pattern enforces this without bundling.

**4. "Fix luck → fix correctness" framing for L2.** The L2 path had the same `continue`
line as the Cosine path. L2 parity was already 18/18 passing before Commit 1.5 because
the L2 gain formula does not depend on a joint multi-partition denominator the way Cosine
does — so the skip was harmless for L2 in practice. Commit 1.5 is not fixing a measured
regression; it is removing a latent correctness defect that would produce analogous divergence
for any L2 tree that encounters degenerate children in sufficient numbers. Shipping the fix
without a regression to point to is the right call when the structural argument is clear.

**5. The precision-class hypothesis chain was sequentially plausible, not obviously wrong.**
From S30 through S33, four independent precision mechanisms were each measurably real at
their measurement layer (cosDen residuals, gain cast ULP, fp64 widening, histogram N-scaling)
but none moved the trajectory-layer outcome. Each falsification required its own measurement.
PROBE-D's N-independent flat gain gap (20.07 units vs 3.89e-5 fp32 floor) was the decisive
oracle: when the gap between the two paths is five orders of magnitude larger than any
precision effect, the mechanism is structural by exclusion. Building that oracle earlier
(running a counterfactual-shadow probe after the first trajectory-gate FAIL) would have saved
several probe sprints.

---

## Files of Record

### Probe and verdict documents

| File | Contents |
|------|----------|
| `docs/sprint33/probe-a-borders/verdict.md` | PROBE-A: CatBoost Pool.quantize produces 2560 borders (falsifies L4) |
| `docs/sprint33/probe-b-python/verdict.md` | PROBE-B: Python path traces same QuantizeFeatures (falsifies L4 Option 3) |
| `docs/sprint33/probe-c-borders/FINDING.md` | PROBE-C Stage 1–2: border subset proof; depth=2 divergence confirmed |
| `docs/sprint33/probe-c-borders/STAGE3_FINDING.md` | PROBE-C Stage 3: coordinate-system correction for L3 depth-0 reading |
| `docs/sprint33/probe-d/FINDING.md` | PROBE-D: precision class closed; signal/noise gain inversion at d=2 |
| `docs/sprint33/probe-e/FINDING.md` | PROBE-E: per-partition smoking gun; mechanism fully specified |
| `docs/sprint33/commit2-gates/REPORT.md` | Four-gate validation report (G4a–G4e all PASS) |
| `docs/sprint33/l4-fix/verdict.md` | L4 original (retracted); preserved for audit trail |

### Original L0–L3 verdicts (retain — L0/L1/L2/L2 graft results survive retraction)

| File | Contents |
|------|----------|
| `docs/sprint33/l0-config/verdict.md` | L0 NO-DIFF — Frame C (config/RNG) falsified |
| `docs/sprint33/l1-determinism/verdict.md` | L1 FALSIFIED — 52.643% seed-independent |
| `docs/sprint33/l2-graft/verdict.md` | L2 FRAME-B — per-iter persistent mechanism confirmed |
| `docs/sprint33/l3-iter2/verdict.md` | L3 SPLIT — S1-grad bit-identical; S2-split divergent (L3 depth=0 reading retracted by PROBE-C Stage 3) |

### State files updated this sprint

| File | What changed |
|------|-------------|
| `.claude/state/DECISIONS.md` | DEC-040 outcome appended; DEC-041 invalidated; DEC-042 opened and closed |
| `.claude/state/HANDOFF.md` | S33 FULLY CLOSED; all probe results; DEC-032 closed |
| `docs/sprint33/sprint-close.md` | This file (replaces pre-retraction draft) |
