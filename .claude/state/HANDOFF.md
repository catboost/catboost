# Handoff — CatBoost-MLX

> Last updated: 2026-04-21 (Sprint 25 OPEN — DEC-026 cascade-robust GAIN research; stacked on S24; G1 ε-calibration in progress)

## Current state

- **Branch**: `mlx/sprint-25-dec026-cascade` (stacked on `mlx/sprint-24-dec023-fix`)
- **Tip commit**: S25 kickoff (post-scaffold); S24 tip at `3f4fff8a2d` (closeout) over `784f82a891` (v5)
- **Campaign**: Operation Verstappen — battle 9 CLOSED (S24). Post-campaign research: DEC-026 cascade-robust GAIN (S25) investigates R8 recovery 1.01× → ~1.85–1.90× via lexicographic ε-tiebreak. No guaranteed delivery.
- **Open PRs** (stacked on RR-AMATOK/catboost-mlx): #9 → #10 → #11 → #12 → #13 → #14 → #15 → **#16 OPEN** (Sprint 24) — S25 branch stacked on top; PR #17 deferred until S25 verdict
- **Known bugs**: BUG-T2-001 RESOLVED (`784f82a891`). Sibling S-1 (`kHistOneByte` writeback race) still latent, still guarded by NIT-4 CB_ENSURE `maxBlocksPerPart == 1`. BUG-007 and bench_boosting K=10 anchor mismatch still OPEN/unscheduled.

## Sprint 24 — CLOSED

### Verdict: D0 PASS on parity (DEC-023 RESOLVED); FAIL on R8 preservation (Verstappen retroactive retreat). R8: 1.01× post-fix.

**Branch tip at close**: `784f82a891`
**Date closed**: 2026-04-21

| Track | Verdict | Key finding |
|-------|---------|-------------|
| D0 — DEC-023 v5 fix | PASS — all 4 gates | ULP=0, 18/18 parity, 100/100 gate determinism |
| R8 preservation | FAIL — retroactive | 1.90× was predicated on non-deterministic T2; v5 collapses to 1.01× |
| S24-BENCH-G1 — championship suite | NOT RUN | Campaign retreated before suite started |

### D0 detail

**Problem**: DEC-023 — features 1-3 `atomic_fetch_add` on float in T2-accum; bimodal output
at config #8 (~50/50 between 0.48231599 / 0.48231912, 105 ULP gap).

**Path 5 falsified**: All T2-sort + int-fixed-point variants retaining feature-0's bin-range scan
over `sortedDocs` pinned to Value B (105 ULP off T1's Value A). Root cause: reduction topology
difference between sort-based scan and T1's SIMD fold. Integer accumulation made features 1-3
deterministic but did not change feature-0's incompatible topology.

**CPU anchor (Path X)**: CPU CatBoost at config #8 = 0.068 (~24M ULP from both A and B).
Inconclusive — bench_boosting is a GPU-kernel-speed harness, not a CatBoost conformance test.
T1 Value A (0.48231599) remains the declared parity anchor by construction.

**Off-by-one retest (false positive)**: Proposed off-by-one between scoring kernel ("bin ≥ b
right") and apply path ("bin > b right") was a coordinate-system labeling artifact. Both paths
encode `raw_bin > splitIdx`, consistent with CatBoost's `IsTrueHistogram`. No bug present.
Diagnostic at `docs/sprint24/d0_offby1_cascade_retest.md`.

**v5 fix**: All four features (0-3) in T2-accum rewritten to T1-style SIMD-shuffle accumulation
reading from `docIndices`. T2-sort kernel removed from dispatch. ULP=0 is structural — v5
executes the identical FP computation as T1.

**Acceptance criteria results** (all 4 gates PASS):

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| S24-D0-G1 | Config #8: 10/10 deterministic | 10/10 at 0.48231599, ULP=0 | PASS |
| S24-D0-G2 | 18/18 ULP=0, ≥5 runs | 18/18 ULP=0, all 5/5 | PASS |
| S24-D0-G3 | Gate config: 100/100 deterministic | 100/100 at 0.47740927 | PASS |
| S24-D0-G4 | hist_ms ratio ≥ 0.45× | 0.959× | PASS |

### R8 collapse

| Metric | Pre-fix (T2 v4, S22 D4) | Post-fix (T2 v5, S24 D0) |
|--------|:-----------------------:|:------------------------:|
| hist_ms (gate config) | ~6.85 ms (0.317× T1) | ~20.75 ms (0.959× T1) |
| e2e speedup vs S16 baseline | **1.90×** | **~1.01×** |
| Verstappen ≥1.5× | cleared by 40 pp | **FAILED retroactively** |

T2's 0.317× hist_ms ratio derived from the sort-based feature-0 bin-range scan — the same
mechanism that produced a different reduction topology from T1and caused DEC-023. These are not
separable: the speed came from the topological difference; fixing the topology eliminates the
speed.

### DEC-023 resolved

Commit `784f82a891`. v5 is the shipped kernel. The 1.90× record is superseded by 1.01×.

### DEC-026 opened

Research track for S25: cascade-robust GAIN comparison. Hypothesis: a lexicographic tiebreak at
near-tie GAIN comparisons (when `|GAIN_A - GAIN_B| < ε`) could block the cascade amplification
that makes config #8's 1-2 ULP/bin topology difference grow to 105 ULP at iters=50. If the
tiebreak succeeds, T2 Path 5 (sort + int-fixed-point) becomes shippable at R8 ≈ 1.85–1.90×.
This is research, not engineering. Falsification checkpoints at each gate. See
`DECISIONS.md DEC-026`.

## Next actions

1. **Ramos opens PR #16** for Sprint 24 (stacked on #15, branch `mlx/sprint-24-dec023-fix`).
2. **Sprint 25** opens DEC-026 cascade-robust GAIN research. Owner: @research-scientist.
   Entry point: epsilon calibration study (DEC-026-G1). Kill-switch: if no viable ε is
   identified, the research track is falsified and R8 stays at 1.01×.
3. **Standing orders** (unchanged): DEC-012 one-change-per-commit; no Co-Authored-By; RR-AMATOK
   only; parity sweep protocol ≥5 runs per non-gate + 100 runs at gate unconditionally.

## Standing orders (carried forward)

- **No `Co-Authored-By: Claude` trailer** in any commit message — global policy.
- **RR-AMATOK fork only** — do not push or PR to `catboost/catboost` upstream.
- **DEC-012 one-structural-change-per-commit** — still active.
- **Honest R8** — 1.01× is the new position. Do not round, inflate, or annotate "but we had X
  at some point". The 1.90× figure is documented as superseded in DECISIONS.md and CHANGELOG.
- **Parity sweep protocol**: ≥5 runs per non-gate config; gate config unconditionally 100 runs.
  Standing order from S23 D0, unchanged.

## Prior sprints — status

- **Sprint 24** — CLOSED. PR #16 pending (Ramos opens). DEC-023 RESOLVED via v5 (T1 accumulation topology). R8 1.90× → 1.01× retroactive. Verstappen ≥1.5× gate failed. DEC-026 cascade-robust GAIN research opens S25.
- **Sprint 23** — CLOSED. PR #15 pending (Ramos opens). T2 promoted to production (8 commits). R8 1.90× unchanged. D0 PASS (pre-existing bug). R1 DEFERRED. R2 FALSIFIED. DEC-023/024/025 opened.
- **Sprint 22** — CLOSED. PR #14 pending (Ramos opens). T2 SHIPPED, R8 1.90× (record stands at time of close; superseded by S24). Verstappen gate cleared at S22. S22 D3 verdict corrected to 17/18 (see DEC-020 footnote + DEC-023).
- **Sprint 21** — CLOSED. PR #13 pending (Ramos opens). 0× perf, A1 measurement record.
- **Sprint 20** — CLOSED. PR #12 OPEN stacked on #11. T3b DEC-017 RETIRED.
- **Sprint 19** — CLOSED. PR #11 OPEN stacked on #10. T1 DEC-016 SHIPPED (−2.3% e2e).
- **Sprint 18** — CLOSED. PR #10 OPEN stacked on #9. L1a DEC-011 SHIPPED (−66.8% histogram_ms).
- **Sprint 17** — CLOSED. PR #9 OPEN. D1c DEC-009 SHIPPED (−89–93% histogram_ms).
- **Sprints 0–16** — merged to master.
