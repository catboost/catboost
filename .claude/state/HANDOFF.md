# Handoff — CatBoost-MLX

> Last updated: 2026-04-22 (PR #20 merged — latent-bugs backlog cleared; no open PRs; no active sprint)

## Current state

- **Branch**: `master` is the current working base. No active sprint branch.
- **Tip commit**: `71aabaa842` (merge of #20 — latent-bugs cleanup). v5 (`784f82a891`) remains the shipped production kernel.
- **Campaign**: Operation Verstappen — battle 9 CLOSED (S24). Post-campaign research (S25 DEC-026) **FALSIFIED** 2026-04-21 at G1: ε-threading impossible by 21,091× (ε_min = 2.200e-03 vs ε_max⁺ = 1.043e-07). DEC-027 (alternative accumulation) **deferred** — acknowledged; not open; reserved for a dedicated future research sprint.
- **Open PRs**: none. Recent merges: #16 (`1385e056ca`) → #17 (`5caa6e64cf`) → #18 (`9b0c03fec2`, CI unblock) → #19 (`1afd0a35b2`, state refresh) → #20 (`71aabaa842`, latent-bugs cleanup).
- **Known bugs**:
    - BUG-T2-001 RESOLVED (`784f82a891`).
    - BUG-007 MITIGATED 2026-04-22 (`71aabaa842`) — Python wrapper at `core.py:1131-1137` pre-sorts `group_ids`; C++ `BuildDatasetFromArrays` now throws `std::runtime_error` on unsorted input. Two-layer defense.
    - K=10 anchor mismatch RESOLVED Sprint 8 (TODO-022, `CHANGELOG.md:27`); ledger alignment landed in `71aabaa842`.
    - Sibling S-1 (`kHistOneByte` writeback race) still latent; now guarded by `constexpr` + `static_assert(maxBlocksPerPart == 1, ...)` at `histogram.cpp:126` — compile-time failure with pointer to `KNOWN_BUGS.md` if anyone raises the literal.

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

## Sprint 25 — CLOSED (FALSIFIED)

### Verdict: FALSIFIED at G1 on day 1. R8 stays at 1.01×. v5 is final production kernel.

**Branch tip at close**: (S25 FALSIFIED closeout commit)
**Date closed**: 2026-04-21

### G1 empirical result

| Quantity | Value | Source |
|---|---|---|
| Sweep runs | 180 (18 configs × 5 runs × 2 kernels) | 5 min 4 s wall |
| Determinism | 5/5 per (config, kernel) | all 180 bit-identical |
| T1 vs DEC-008 reference | 18/18 exact | T1 reproduces every reference loss |
| T1 vs Path 5 agreement | 17/18 bit-exact | only config #8 diverges (T1=A, Path5=B) |
| Flip events (earliest-per-iter) | 35 total, 7 unique × 5 runs | all at config #8 |
| ε_min (required to gate flips) | 2.200e-03 | config #8 iter 45 depth 0 |
| ε_max (incl. zero-gain ties) | 0.0 | configs 1/2/8/14 pure nodes |
| ε_max⁺ (positive floor) | 1.043e-07 | config #1 iter 40 depth 3 |
| Safety ratio (positive) | 4.74e-05 (target ≥ 2.0) | **21,091× below threshold** |

**Structural cause**: Path 5's flip gaps span 5.96e-08 to 2.2e-03 — the full range of legitimate
top-2 separations at non-#8 configs. No ε discriminates "ambiguous split" from "clear split"
when both share the same gain separation. Kill-switch fired cleanly.

**Implication**: R8 stays at 1.01× (post-S24 honest position, unchanged). Verstappen ≥1.5×
gate remains retroactively failed from S24 D0. v5 (`784f82a891`) is the final production
kernel. DEC-027 (alternative accumulation paths such as XGBoost-style per-feature
deterministic radix-sum) is acknowledged for future research but **not opened** as part of
S25 closure — Ramos dedicates dedicated time for it later.

See `docs/sprint25/g1_epsilon_calibration.md` for the full verdict doc including §9 forward
paths, and `benchmarks/sprint25/g1/results/` for raw artifacts.

## Next actions

1. **Latent bugs triage — DONE** (PR #20, `71aabaa842`). BUG-007 mitigated with two-layer
   defense; K=10 anchor closed in ledger; Sibling S-1 guard promoted to `static_assert`.
   Nothing else on the backlog.
2. **DEC-027 — deferred (unchanged)**. Not opened. Ramos to revisit when dedicating time for
   alternative accumulation research (e.g., XGBoost-style per-feature deterministic radix-sum)
   as a separate future sprint. All scaffolding assumptions (kill-switch, gate criteria, parity
   protocol) will be designed at open-time, not pre-committed here.
3. **No active sprint.** Next session starts cold — pick a direction (new sprint kickoff, bug
   report, or DEC-027 when ready) before delegating work.
4. **Standing orders** (unchanged): DEC-012 one-change-per-commit; no Co-Authored-By; RR-AMATOK
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

- **Latent-bugs cleanup (PR #20)** — merged 2026-04-22 as `71aabaa842`. Three commits under DEC-012: ledger hygiene (close K=10 + BUG-007, reframe S-1), `BuildDatasetFromArrays` groupIds sortedness CB_ENSURE, and `histogram.cpp` S-1 `static_assert`. No production behavior change.
- **State refresh (PR #19)** — merged 2026-04-22 as `1afd0a35b2`. Docs-only alignment of `HANDOFF.md` / `TODOS.md` / `CHANGELOG-DEV.md` with post-stack-merge reality.
- **CI fix (PR #18)** — merged 2026-04-22 as `9b0c03fec2`. Three commits unblocking the stack: MLX 0.31+ CLI breakage in `mlx-build.yaml`, stale `0.3.0`/`minor==3` version pins in `test_qa_round13_sprint10.py`, and overly-broad BUG-001 MAE sentinel in `test_qa_round8_sprint3_losses.py` (narrowed to SIGABRT-only). No production code changes.
- **Sprint 25** — CLOSED, FALSIFIED. Merged 2026-04-22 as `5caa6e64cf` (PR #17). DEC-026 FALSIFIED at G1: ε-threading impossible (safety ratio 4.74e-05 vs 2.0 target). R8 stays at 1.01×. DEC-027 deferred. No production code changes; shipped as empirical falsification evidence.
- **Sprint 24** — CLOSED. Merged 2026-04-22 as `1385e056ca` (PR #16). DEC-023 RESOLVED via v5 (T1 accumulation topology). R8 1.90× → 1.01× retroactive. Verstappen ≥1.5× gate failed. DEC-026 cascade-robust GAIN research opened S25.
- **Sprints 0–23** — merged to master.
