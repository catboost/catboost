# Sprint 28 Close — Score Function Fidelity (DEC-032)

**Sprint:** 28  |  **Status:** CLOSED  |  **Date:** 2026-04-23
**Branch:** mlx/sprint-19-hist-writeback  |  **Tip:** e0b0b1b527

---

## Mission

DEC-032 (adopted S27-FU-3) established that MLX `FindBestSplitPerPartition` hardcodes L2 Newton
gain while CPU CatBoost defaults to Cosine — a structural algorithmic gap, not noise. Sprint 28
was opened to close that gap: implement `EScoreFunction` enum dispatch in all MLX grow-policy
paths, make L2 explicit, implement Cosine gain from the CPU reference, re-bless all parity cells
with explicit `score_function` labels, and guard combinations that are not yet numerically safe.
The sprint produced a fully dispatched enum surface on the Python canonical path with quantitative
evidence for each guarded combination's drift envelope.

---

## Outcomes

### What shipped

- **`0409e632fa` S28-00 kickoff** — branch cut from master (`4b3711f82b`, post PR #25), state
  files updated with fleshed-out acceptance criteria.

- **`da02da0259` S28-AUDIT** — formal grep confirmation: zero `score_function` references in
  `catboost/mlx/` prior to this sprint. Hardcoded L2 call site confirmed at
  `csv_train.cpp:~L1281`. Documents zero-plumbing baseline.

- **`83f30c3677` S28-COSINE** — `ComputeCosineGainKDim` helper ported from CPU
  `catboost/private/libs/algo/score_calcers.cpp` (`TCosineScoreCalcer`). Implements the
  numerator `(Σg)² / (Σh + reg)` and denominator `sqrt((Σg²/Σh) × (Σh + reg))` structure,
  matching CPU semantics. No dispatch wired yet at this commit.

- **`0ea86bde21` S28-L2-EXPLICIT** — `EScoreFunction` enum + `ParseScoreFunction` added.
  Dispatch wired into `FindBestSplitPerPartition` (Depthwise and Lossguide paths). Nanobind
  binding exposes `score_function` to Python. `_validate_params` in `python/catboost_mlx/core.py`
  rejects `NewtonL2` and `NewtonCosine` with `ValueError`.

- **`4083add248` S28-OBLIV-DISPATCH** — dispatch mirrored into `FindBestSplit`
  (SymmetricTree / oblivious path). All three grow policies now dispatch on
  `EScoreFunction`.

- **`c07e895f7c` S28-REBLESS** — 8 parity cells in `tests/test_python_path_parity.py`
  relabeled with explicit `score_function` per DEC-031 Rule 3. No cell may now rely on a
  silent CPU default. AN-017 re-captured.

- **`dca62f0d72` S28-FU3-REVALIDATE** — DW force-L2 branch removed; DW+Cosine revalidation
  gates pass. LG force-L2 retained pending S29 root-cause (LG+Cosine drift is unacceptable;
  see Guarded Combinations below).

- **`b9577067ef` S28-{LG,ST}-GUARD** — two Python `ValueError` rejections added in
  `core.py`: `Cosine + Lossguide` and `Cosine + SymmetricTree`. Both guarded pending S29
  work.

- **`e0b0b1b527` S28-CR-S1** — dead `ComputeCosineGain` scalar helper removed (code-review
  cleanup, CR nit S1 resolved). No behavior change. Parity state unchanged at 28/28.

### What's guarded

| Combination | Evidence | Guard |
|-------------|----------|-------|
| `Cosine + Lossguide` | ~unacceptable per-partition gain drift; LG priority-queue magnitude interaction with joint-Cosine gain not yet root-caused | `ValueError` in `core.py:634` |
| `Cosine + SymmetricTree` | 0.77% drift at 1 iter → ~47% at 50 iter (float32 joint-denominator compounding); pending Kahan/Neumaier port | `ValueError` in `core.py:644` |

`Cosine + Depthwise` ships without a guard. Measured drift: 1.6% at N=1000/50k/50-iter — within
the DEC-032 accepted envelope. No guard placed.

### What was re-blessed

All 8 parity cells in `tests/test_python_path_parity.py` now carry explicit `score_function`
annotations per DEC-031 Rule 3 (commit `c07e895f7c`). AN-017 re-captured as a class-a update.
No cell may henceforth silently fall through to the CPU default.

---

## Scope Evolution

| Decision point | Direction chosen | Rationale |
|---------------|-----------------|-----------|
| S28 open (2026-04-23) | Small-sprint shape: audit → Cosine port → dispatch → rebless → CR/SA/close. NewtonL2/NewtonCosine deferred. | Per Ramos: get the enum surface right first; Newton variants are a separate tranche. |
| S28-FU3-REVALIDATE discovery | DW force-L2 lifted (parity passes); LG force-L2 retained (drift unacceptable) | Quantitative gate evidence distinguishes safe vs unsafe combinations rather than treating all non-oblivious paths equally. |
| Hybrid-D scope decision (after LG evidence) | ST+Cosine also guarded when ~47% 50-iter drift measured | Same principle: gate on evidence, not grow-policy taxonomy. |
| S28-{LG,ST}-GUARD commit | Two `ValueError` guards added | Explicit rejection preferred over silent wrong results; S29 carries the proper fix. |

---

## Gates

| Gate | Verdict | Report path | Follow-ups |
|------|---------|-------------|------------|
| G2a — Cosine gain unit: DW N=1000, rs=0, 5 seeds | PASS | `docs/sprint28/fu-cosine/t2-gate-report.md` | None |
| G2b — Cosine vs CPU Cosine per-partition ratio ∈ [0.98, 1.02] | PASS | `docs/sprint28/fu-cosine/t2-gate-report.md` | None |
| G3a — L2 non-regression: DW/LG/ST rs=0 after dispatch refactor | PASS | `docs/sprint28/fu-l2-explicit/t3-gate-report.md` | None |
| G3b — Enum dispatch wired: `grep score_function` returns dispatch table (no bare call) | PASS | `docs/sprint28/fu-l2-explicit/t3-gate-report.md` | None |
| G3c — `NewtonL2`/`NewtonCosine` rejected at Python API with `ValueError` | PASS | `docs/sprint28/fu-l2-explicit/t3-gate-report.md` | S29-CLI-GUARD: port into C++ `TrainConfigToInternal` + CLI |
| G5a–G5d — REBLESS: 8/8 cells carry explicit `score_function`; zero silent-default cells | PASS | `docs/sprint28/fu-rebless/t4-rebless-report.md` | None |
| G6a–G6d — FU3-REVALIDATE: DW 5/5 PASS under Cosine both sides; LG force-L2 justified | PASS | `docs/sprint28/fu-fu3-revalidate/t5-gate-report.md` | S29-LG-COSINE-RCA |
| G7 — OBLIV-DISPATCH: ST dispatch present; ST+Cosine guard fires correctly | PASS | `docs/sprint28/fu-obliv-dispatch/t7-gate-report.md` | S29-ST-COSINE-KAHAN |
| T6-CR — Code review | PASS-WITH-NITS | `docs/sprint28/fu-cr/t6-cr-report.md` | CR-S1 resolved by `e0b0b1b527`; CR-N1, CR-N2, CR-N3 remain nits (non-blocking) |
| T6-SA — Security audit | PASS-WITH-FINDINGS | `docs/sprint28/fu-sa/t6-sa-report.md` | SA-H1 deferred to S29-CLI-GUARD (non-blocking) |

---

## Parity Suite State

28/28 tests passing at `b9577067ef`. Unchanged at `e0b0b1b527` (dead-code removal, no
behavior change). No new test failures introduced this sprint. No parity regression on v5
kernel ULP=0 record (kernel sources untouched throughout S28).

---

## Open Items (carry to S29)

- **S29-CLI-GUARD** (SA-H1): Port `Cosine+Lossguide` and `Cosine+SymmetricTree` combination
  rejections into `catboost/mlx/train_api.cpp:TrainConfigToInternal` (C++ entry point) and
  `catboost/mlx/tests/csv_train.cpp:ParseArgs` (CLI entry point). Currently only the Python
  API surface rejects these combinations; C++ direct calls and the `csv_train` binary bypass
  the guard. Verification: unit test `_core.train()` with forbidden combo throws
  `std::invalid_argument`; CLI test exits non-zero on same combo.

- **S29-LG-COSINE-RCA**: Root-cause investigation of Lossguide × Cosine priority-queue leaf
  ordering × joint-gain magnitude interaction producing unacceptable per-partition gain drift.
  Deliverable: triage doc with mechanism identified + fix plan. Referenced by
  `python/catboost_mlx/core.py:634`.

- **S29-ST-COSINE-KAHAN**: Port Kahan/Neumaier compensated summation into the joint-Cosine
  denominator accumulator in `catboost/mlx/tests/csv_train.cpp` (`ComputeCosineGainKDim`
  callers). Deliverable: 50-iter ST+Cosine drift ≤ 1% at N=50k. Referenced by
  `python/catboost_mlx/core.py:644`.

- **CR-N3 (out-of-scope visibility, non-blocking)**: CPU Lossguide accumulates a single Cosine
  score across all leaves of a given split candidate; MLX computes per-(split × partition).
  Pre-existing DW/LG design divergence, not introduced by S28. Not DEC-032 scope. Noted in
  T6-CR report for future audit.

---

## Key Numbers

| Combination | Drift measured | Status |
|-------------|---------------|--------|
| DW + Cosine | 1.6% at N=1000/50k/50-iter | Ships — within envelope |
| LG + Cosine | ~unacceptable (not yet quantified precisely) | Guarded pending S29-LG-COSINE-RCA |
| ST + Cosine | 0.77% @ 1 iter → ~47% @ 50 iter | Guarded pending S29-ST-COSINE-KAHAN |

All `score_function=L2` cells: zero regression across 28/28 test suite at `b9577067ef`.

---

## DEC-032 Status

**PARTIALLY CLOSED.** The Python canonical surface is fully dispatched and guarded. All four
`EScoreFunction` enum values are wired (`L2` and `Cosine` implemented; `NewtonL2` and
`NewtonCosine` explicitly rejected). Parity suite re-blessed with explicit labels. The gap
between what ships (`DW+Cosine` in-envelope, `L2` non-regressed) and what is deferred
(`LG+Cosine` RCA, `ST+Cosine` Kahan, CLI-level guards) is tracked under S29-CLI-GUARD,
S29-LG-COSINE-RCA, and S29-ST-COSINE-KAHAN. DEC-032 promotion to fully CLOSED is queued for
S29-close once SA-H1 (CLI-GUARD) is resolved.
