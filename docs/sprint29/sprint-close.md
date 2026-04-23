# Sprint 29 Close — DEC-032 Closeout + LG Mechanism Spike

**Sprint:** 29  |  **Status:** CLOSED  |  **Date:** 2026-04-23
**Branch:** mlx/sprint-29-dec032-closeout  |  **Base:** master `987da0e7d5`  |  **Tip:** `fa7f9b55fc`

---

## Mission

Complete DEC-032 closeout via two parallel tracks:

1. **S29-CLI-GUARD** — Port `Cosine+{LG,ST}` rejection guards from the Python `_validate_params`
   layer into `catboost/mlx/train_api.cpp:TrainConfigToInternal` (nanobind entry) and
   `catboost/mlx/tests/csv_train.cpp:ParseArgs` (CLI entry). This closes SA-H1 from the S28
   security audit.

2. **S29-LG-MECHANISM-SPIKE** — 1-session-capped investigation of whether LG+Cosine drift
   shares the float32 joint-Cosine denominator compounding mechanism with ST+Cosine (outcome A)
   or is driven by priority-queue leaf-ordering sensitivity (outcome B). Capped to one session
   per scope refinement (E) agreed at kickoff; LG/ST Kahan work carries to S30 regardless.

DEC-034 (PENDING-SPIKE) resolves to outcome A at verdict commit `64a8d9076b`.

---

## Scope (E) Recap

Scope (E) is: CLI-GUARD + LG spike + 1-session cap. No Kahan/Neumaier implementation this
sprint. S29-ST-COSINE-KAHAN and S29-LG-COSINE-RCA are both absorbed into a single
S30-COSINE-KAHAN task following the outcome-A verdict.

| Track | Scope | Outcome |
|-------|-------|---------|
| CLI-GUARD-T1 | Port guards to C++ + CLI | DONE (`73e9460a31`) |
| CLI-GUARD-T2 | pytest coverage for guards | DONE (`c73f5073af`) |
| LG-SPIKE-T1 | Iter-1 drift measurement | DONE (`503ebacdb2`) |
| LG-SPIKE-T2 | Outcome A/B/C verdict | DONE — **Outcome A** (`64a8d9076b`) |
| CR | Code review | PASS-WITH-NITS (`3f87b85e38`) |
| SA | Security audit / SA-H1 | PASS (`3f87b85e38`) |

---

## Commits

| SHA | Tag | Description |
|-----|-----|-------------|
| `33ce5f1d66` | S29-00 | Branch kickoff; scope (E), DEC-032 closeout + LG spike; state files updated |
| `73e9460a31` | S29-CLI-GUARD-T1 | Port `Cosine+{LG,ST}` rejection to `train_api.cpp:TrainConfigToInternal` + `csv_train.cpp:ParseArgs` |
| `c73f5073af` | S29-CLI-GUARD-T2 | pytest coverage for C++ and CLI guards (4 test cases) |
| `503ebacdb2` | S29-LG-SPIKE-T1 | LG+Cosine iter-1 drift measurement harness + data artifacts |
| `64a8d9076b` | S29-LG-SPIKE-T2 | DEC-034 verdict — outcome A (shared compounding mechanism) |
| `3f87b85e38` | S29-CR + S29-SA | CR gate report (PASS-WITH-NITS) + SA gate report (PASS); SA-H1 CLOSED |
| `fa7f9b55fc` | S29-CR SF-1 | Verdict wording tightened per CR SF-1 (honest "<0.2%" language) |

7 commits. No production code changes after `c73f5073af`.

---

## Gate Results

### Code Review (T1-CR) — PASS-WITH-NITS

Report: `docs/sprint29/fu-cr/t1-cr-report.md`

| Count | Category |
|-------|----------|
| 0 | Must Fix |
| 3 | Should Fix |
| 3 | Nits |
| 5 | Praise |

All Must Fix: 0. Sprint proceeds to close.

Gates passed: G-CR1 (guard correctness), G-CR2 (dispatch plumbing), G-CR3 (DEC-012 atomicity),
G-CR4 (test quality), G-CR5 (verdict-doc integrity), G-CR6 (project conventions),
G-CR7 (dead code / debt).

### Security Audit (T1-SA) — PASS

Report: `docs/sprint29/fu-sa/t1-sa-report.md`

| Count | Category |
|-------|----------|
| 0 | Critical |
| 0 | High |
| 0 | Medium |
| 0 | Low |
| 2 | Informational |

SA-H1 (C++/CLI guard bypass, opened S28): **CLOSED.**

Gates passed: G-SA1 (SA-H1 closure — all three entry points guarded), G-SA2 (guard-layer
correctness), G-SA3 (error message info-disclosure — PASS-INFO), G-SA4 (numerical bounds
re-check), G-SA5 (CLI exit behavior — PASS-INFO).

### Parity

28/28 Python parity suite at all S29 commits. No kernel sources touched. v5 ULP=0 record
unaffected throughout.

---

## DEC-032 Status

**PARTIALLY CLOSED** (unchanged from S28). The status does not advance to CLOSED this sprint
because the two guarded combinations (`Cosine+Lossguide`, `Cosine+SymmetricTree`) remain
guarded pending the S30 Kahan fix. SA-H1 is closed — all three entry points (Python, nanobind,
CLI) now reject forbidden combinations — but lifting the guards requires the shared Kahan fix to
land and gate first.

Progress made in S29:
- SA-H1 closed: guards ported to C++ nanobind entry and CLI entry.
- DEC-034 resolved (outcome A): LG+Cosine and ST+Cosine share the same float32 joint-denominator
  compounding mechanism. A single Kahan fix addresses both.

DEC-032 will promote to **CLOSED** when S30-COSINE-KAHAN lands and both parity gates pass.

---

## DEC-034 Resolved — Outcome A

**Verdict: Outcome A (shared compounding mechanism). Confidence: moderate.**

| Signal | LG+Cosine (spike, post-S28) | ST+Cosine anchor (S28) |
|--------|-----------------------------|------------------------|
| iter-1 mean drift (%) | 0.0024 (mean of 3 seeds) | 0.77 |
| iter-1 per-seed drift (%) | 0.0046 / 0.0015 / 0.0010 | — |
| 50-iter drift (%) | 0.0029 – 0.197 per seed | ~47 (aggregate) |
| iter=1 BFS split seq (seed=0) | `[0,0,0,1,1,1,1]` — identical CPU vs MLX | n/a |

Drift is ~300× smaller than the ST+Cosine anchor but structurally the same compounding pattern.
At iter=1 the priority-queue BFS sequences are bit-identical (CPU vs MLX), ruling out outcome B
at this cell. 50-iter peak is 0.197% (seed=1) — within outcome-A envelope.

Confidence is moderate, not high: 3 seeds, one shallow cell (depth=3, max_leaves=8). Deep LG
cells with `max_leaves>8` are not covered; priority-queue ordering sensitivity cannot be ruled
out there.

Source: `docs/sprint29/lg-mechanism-spike/verdict.md`; data at
`docs/sprint29/lg-mechanism-spike/data/`.

Cell-mismatch disclosure: the t5-gate-report's 14% LG figure
(`docs/sprint28/fu-fu3-revalidate/t5-gate-report.md`) was pre-S28 algorithmic divergence (MLX
hardcoded L2 vs CPU Cosine, closed in `0ea86bde21`), not float-precision drift. The 0.0024%
figure is the first honest measurement of LG+Cosine float-precision drift with matching gain
functions on both sides.

---

## Carries to S30

### S30-COSINE-KAHAN (primary)

Merges S29-ST-COSINE-KAHAN carry + S29-LG-COSINE-RCA follow-up into a single task.

**Scope**: Apply Kahan/Neumaier compensated summation to the shared float32 joint-Cosine
denominator accumulator in `ComputeCosineGainKDim`. Gate both `ST+Cosine` and `LG+Cosine`
parity behind a single post-fix check. Remove both Python and C++ guards atomically upon
parity pass.

**Cleanup grep**: `grep -rn 'TODO-S29-(LG|ST)-COSINE'` returns exactly four guard sites
(Python `core.py:628-647`, C++ `train_api.cpp:25-51`, C++ `csv_train.cpp:241-267`, and
`tests/test_cli_guards.py`). All four must be removed in one commit.

**Gate**: 50-iter `ST+Cosine` drift ≤ 1% at N=50k; 50-iter `LG+Cosine` drift ≤ 1% at the
spike cell (N=1000, depth=3, max_leaves=8).

### S30-CLI-EXIT-WRAP (secondary, SA-I2-S29 / CR pattern watch)

Add top-level `try { ... } catch (const std::invalid_argument& e) { fprintf(stderr, "%s\n",
e.what()); return 1; }` in `csv_train.cpp:main()`. Replaces current SIGABRT(134) path with
graceful `exit(1)` on guard fires. Tests already assert `returncode != 0` and will pass
without changes.

### S31-LG-DEEP-RESIDUAL (conditional — blocked by S30)

Open a dedicated spike in S31 **only if** post-S30-COSINE-KAHAN drift persists on deep/wide
LG cells (`depth>3`, `max_leaves>8`). If Kahan closes LG+Cosine at the spike cell and deep
cells are not tested, do not open S31. Re-open outcome B only on evidence.

---

## Nits Not Addressed (with rationale)

| Item | Source | Rationale for deferral |
|------|--------|------------------------|
| N-1: stale `#include <stdexcept>` comment in `train_api.cpp:18` | CR | Cosmetic; one-liner; will be touched anyway during S30 guard removal |
| N-2: binary-path env override in `test_cli_guards.py:30-37` | CR | Non-blocking; current layout matches `setup.py` output; address in S30 guard-removal commit |
| N-3: `/dev/null` ordering-luck in `test_cli_guards.py:192-229` | CR | Cosmetic; guard-before-file-open ordering is documented and stable; will update in S30 with guard removal |
| SF-3: dead `run_secondary()` in `harness.py:228-244` | CR | One-shot spike harness in `docs/`; not production; non-blocking |
| SA-I1-S29: sprint markers in error messages | SA | Intentional cross-language grep anchors for S30 single-point removal; accepted in S28 G-SA3 |

Should-Fix items SF-1 and SF-2 were addressed before close:
- **SF-1** resolved in `fa7f9b55fc`: verdict wording changed to "reaches 0.197% at iter=50 seed=1,
  and stays below 0.2% on all measured points."
- **SF-2** noted: downstream docs citing tree structure should use "BFS feature-index sequence
  bit-identical" rather than "trees bit-identical"; commit message already landed.

---

## Sprint Retrospective

CLI-GUARD went cleanly: verbatim guard porting across three languages (Python → C++ nanobind →
C++ CLI) with byte-identical error messages and `TODO-S29-*` grep anchors keeps the S30
guard-removal to a single `grep -rn` sweep. The spike-plus-verdict two-commit pattern
(`503ebacdb2` data, `64a8d9076b` interpretation) worked well — reruns can overwrite data
without losing the verdict, and the outcome-A classification is now traceable to a specific
evidence snapshot. The 1-session cap on the spike was the right call: outcome A was clear
from iter-1 data alone, and deeper LG cell coverage can wait until after Kahan confirms or
falsifies the shared-mechanism hypothesis.
