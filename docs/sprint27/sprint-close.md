# Sprint 27 — Correctness Closeout — Close Report

**Sprint**: S27 — Correctness Closeout
**Branch**: `mlx/sprint-27-correctness-closeout`
**Date closed**: 2026-04-22
**Base SHA**: `20d14564e8` (master at branch cut)
**Tip SHA (pre-close)**: `44bb9ee74b` (S27-CR; 26 commits on branch)
**PR status**: OPEN-PENDING-RAMOS
**Verdict**: ALL TRACKS CLOSED. G1-FU1 6/6 PASS. G2-AA 0 class-b. G3-FU3 5/5 PASS (conditional). CR APPROVE. SA PASS-WITH-NOTES.

---

## Sprint summary

Sprint 27 closed three correctness debts carried from S26 before any R8 performance work resumes.
Track A fixed a two-bug sequence in `ComputeLeafIndicesDepthwise` that produced wrong validation
RMSE tracking (not training correctness). Track B audited all 18 committed numeric anchors,
found 0 class-b regressions, and codified a standing anchor hygiene protocol (DEC-031). Track C
triaged the DW N=1000 parity asymmetry from S26-FU-2 and identified it as a score-function
fidelity gap rather than a parity edge case: MLX hardcodes L2 Newton gain while CPU defaults to
Cosine. The honest scope split — conditional gate pass on `score_function='L2'`, full Cosine port
to S28 — is codified in DEC-032. No kernel sources were modified. R8 remains 1.01× (unchanged).

---

## Tracks summary

| Track | Scope | Outcome | Key commits | Key DEC |
|-------|-------|---------|-------------|---------|
| A (FU-1) | DW leaf-index fix — `ComputeLeafIndicesDepthwise` encoding + split-lookup bugs | G1-FU1 6/6 PASS; ratios 0.9988–1.0027 | `34f62b32c9`, `eca086e4dd`, `c7c09451e2`, `fb7eb59b5f`, `88cbe6d067`, `13c7ac9b2b` | DEC-030 |
| B (AA) | Anchor audit — 18 anchors classified; hygiene protocol authored | 0 class-b; 4 class-a updated; 2 class-c; 3 class-d resolved; 2 anchors now live-enforced | `d4e2d7cf88`, `800fdc8fce`, `9be26b91c0`, 9 × anchor-landing commits, `884147774b` | DEC-031 |
| C (FU-3) | DW N=1000 parity-asymmetry triage — fidelity gap identified | G3-FU3 5/5 PASS (CPU `score_function='L2'`); Cosine port deferred to S28 | `0931ad6e9c`, `ff053fa3ac`, `fc44bfc936`, `591f4ce3e6` | DEC-032 |
| D (Quality) | CR + SA + close | CR: APPROVE (`44bb9ee74b`); SA: PASS-WITH-NOTES (`24e80dde45`); CLOSE: this commit | `44bb9ee74b`, `24e80dde45`, this commit | — |

---

## Exit gates

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| G0 | Root-cause docs complete (DEC-030, DEC-031, DEC-032) | All three landed: DEC-030 at `c7c09451e2`, DEC-031 at `884147774b`, DEC-032 at `ff053fa3ac` | PASS |
| G1-FU1 | DW validation RMSE (rs=0) ratio ∈ [0.98, 1.02], 3 seeds × {N=10k, N=50k} | 6/6 PASS; ratios 0.9988–1.0027. Evidence: `docs/sprint27/fu1/t4-gate-report.md`, commit `88cbe6d067` | PASS |
| G2-AA | All 18 anchors classified; 0 class-b uninvestigated | 18/18 classified. 0 class-b. 4 class-a updated; 2 class-c; 3 class-d resolved. Evidence: `docs/sprint27/scratch/` anchor inventory + T3 classification | PASS |
| G3-FU3 | DW N=1000 ratios ∈ [0.98, 1.02] with `score_function='L2'` on CPU side | 5/5 PASS; ratios 0.9956–1.0011. Conditional on CPU explicit L2. Unconditional (Cosine) is S28 scope. Evidence: `docs/sprint27/fu3/t4-gate-report.md`, commit `591f4ce3e6` | PASS (conditional) |
| G4 | bench_boosting v5 ULP=0 preserved (18-cell sweep, kernels unchanged) | NOT RUN — no kernel source was modified in S27. No code path changed that would affect ULP record. Prior gate (S24 D0-G2, S26-FU-2 reproducing 17/18 → G4) stands. | NOT RUN — N/A by DEC-032 scope |
| G5 | `tests/test_python_path_parity.py` 8/8 PASS | The 8/8 suite is the same suite extended by FU-3-T4 (`fc44bfc936`). S27 DW parity tests now pass with explicit `score_function='L2'` on CPU side. No nanobind or training path was modified by Track A or Track C beyond the validation-path fix. Evidence: T4 gate artifacts at `591f4ce3e6`. | PASS |
| G6 | R8 bench_boosting e2e ratio drift ≤ 2% vs master baseline | NOT RUN — S27 is zero-perf-work. No production kernel, training, or leaf-estimation code was modified. Prior R8 position of 1.01× (post-S24) is unchanged. | NOT RUN — no perf-path changes |
| G7 | Determinism: gate config 100 runs max−min ≤ 1e-6 | NOT RUN — no kernel sources modified. Kernel is still v5 (`784f82a891`). Prior determinism gate (S24 D0-G3: 100/100 at gate config; S26-FU-2 G5: max−min 1.49e-08) stands. | NOT RUN — no kernel changes |

---

## Carried tech-debt

### From CR nits

**1. G1-FU1 regression guard not CI-wired (medium priority)**

The 6-cell DW validation RMSE check exists in the gate script but is not a live pytest assertion.
This creates tension with DEC-031 Rule-1 ("no docs-only canonical values"). Recommended S28-QA
task: port the 6-cell DW val-RMSE ratio check into `tests/test_python_path_parity.py` as a live
assertion with named constants. Non-blocking for S27 close.

**2. `exit(1)` in `ComputeLeafIndicesDepthwise` invariant guard (low priority)**

The guard added by FU-1 calls `exit(1)` on invariant violation. `std::runtime_error` (or
CatBoost's `CB_ENSURE`) is more idiomatic and integrates with the test harness. Stylistic only;
no correctness impact. Can fold into the next S28 refactor pass that touches `csv_train.cpp`.

### From SA notes

**3. Hard-coded `/Users/ramos/...` paths in 3 scratch scripts (low priority)**

`docs/sprint27/fu3/fu3_t4_gate.py`, `docs/sprint27/scratch/algorithmic/step1_step2_sweep.py`,
and `docs/sprint27/scratch/algorithmic/step3_score_function_analysis.py` contain absolute paths
tied to the developer's username. Low-risk (docs/scripts only, no production path), but leaks
username into the committed tree. Fold into S28 opener or a dedicated debt-sweep task.

**4. Theoretical bit-shift UB at depth ≥ 32 (low priority)**

`partBits |= (goRight << lvl)` in the FU-1 fix could shift into undefined behavior if `lvl ≥ 32`
on a 32-bit integer. This is pre-existing (the pattern appears elsewhere in the codebase) and is
not widened by S27. CatBoost's default `MaxDepth = 16` provides an effective ceiling. Add an
explicit bounds check (`CB_ENSURE(depth < 32, ...)`) in S28 scope, low priority.

### From anchor audit (AA)

**5. AN-015 dead tests: 5 SKIP markers with DEC-031 pointer, never wired (low priority)**

The `TestCIBenchWorkflow` tests in `mlx-test.yaml` were corrected for the filename typo but the
underlying `BENCH_FINAL_LOSS` / baseline embed never landed. The 5 dead tests remain marked SKIP
with a DEC-031 pointer. Either retire them (confirm they test a feature that does not exist) or
implement Item H. Tracked in DEC-031 §class-d carry-forwards. S28 scope or dedicated debt sweep.

**6. AN-013 / AN-014 DEAD markers left in place (low priority)**

DEC-031 codifies the "remove or wire" policy for class-d anchors. AN-013 and AN-014 have DEAD
markers committed but have not been removed or wired to live assertions. Follow-up decision: remove
in S28 (clean closure) or keep as permanent historical markers with explicit rationale. S28 scope.

**7. AN-017 deferred-a: confirmed FU-1-immune, no action required**

AN-017's re-capture was deferred until the DW leaf-index fix (DEC-030) merged. Post-FU-1
triage confirmed `valDocs = 0` in the anchor-generating config: the validation path was never
exercised in this anchor, so the fix produces no value change. No re-capture needed. Logged here
for completeness; no S28 task required.

### From DEC-032

**8. Score function fidelity gap — full scope is S28**

MLX `FindBestSplitPerPartition` hardcodes L2 Newton gain. CPU CatBoost defaults to Cosine. This
is not S27 debt — DEC-032 correctly assigns the full Cosine port (audit dispatch plumbing →
implement Cosine → re-bless aggregate parity claims → optional Newton variants) to S28. S28 is
the debt container for this item. See DEC-032 and `HANDOFF.md §Sprint 28`.

---

## Known limitations

- **DW parity tests require `score_function='L2'` on CPU side.** Unconditional DW parity (any
  score function, including CPU default Cosine) is S28 scope. The S27 gate is structural and
  honest but conditional; do not treat G3-FU3's 5/5 as evidence of full algorithm parity.
- **`_core.so` / nanobind build is not CI-asserted across the S27 changes.** Relies on developer
  rebuild. This is an existing known issue (pre-S27) and was not widened by any S27 change.

---

## Code review outcome

**S27-CR** (@code-reviewer, commit `44bb9ee74b`): **APPROVE** — 0 blockers. CR nits recorded as
tech-debt items 1 and 2 above.

**S27-SA** (@security-auditor, commit `24e80dde45`): **PASS-WITH-NOTES** — 0 CRITICAL, 0 HIGH.
SA notes recorded as tech-debt items 3 and 4 above.

---

## Metrics

| Metric | Value |
|--------|-------|
| Commits on branch | 26 (+ this close commit = 27) |
| Files changed vs master | 33 |
| Insertions | 3,991 |
| Deletions | 59 |
| Kernel source changes | 0 |
| DECs authored | 3 (DEC-030, DEC-031, DEC-032) |
| Anchors audited | 18 |
| Class-b regressions | 0 |
| Gate cells: G1-FU1 | 6/6 PASS |
| Gate cells: G3-FU3 | 5/5 PASS (conditional) |

---

## Next sprint pointer

**S28 — Score function fidelity** — see DEC-032 and `HANDOFF.md §Sprint 28`.
Branch `mlx/sprint-28-score-function-fidelity` cuts from master after S27 PR merges.
Blocked on S27 PR merge.

**S28 minimum scope**: audit `score_function` dispatch plumbing → implement Cosine gain in
`FindBestSplitPerPartition` → make L2 explicit via enum/dispatch → re-bless aggregate parity
claims with explicit `score_function` annotation → re-validate FU-3's 5 DW N=1000 cells with
Cosine both sides (structural proof of gap closure).

---

## Files of record

| File | Role |
|------|------|
| `docs/sprint27/fu1/t4-gate-report.md` | G1-FU1 gate report (6/6 cells; ratios 0.9988–1.0027) |
| `docs/sprint27/fu3/t4-gate-report.md` | G3-FU3 gate report (5/5 cells; ratios 0.9956–1.0011; conditional L2) |
| `docs/sprint27/reviews/` | CR and SA reports |
| `docs/sprint27/scratch/` | AA inventory, classification, and FU-3 triage artifacts |
| `.claude/state/DECISIONS.md §DEC-030` | DEC-030 — DW leaf-index BFS fix |
| `.claude/state/DECISIONS.md §DEC-031` | DEC-031 — Anchor hygiene protocol |
| `.claude/state/DECISIONS.md §DEC-032` | DEC-032 — Score function fidelity gap |
| `tests/test_python_path_parity.py` | Extended parity harness (CPU `score_function='L2'` explicit from FU-3-T4) |
| `catboost/mlx/train_lib/csv_train.cpp` | `ComputeLeafIndicesDepthwise` fix (Track A) |
