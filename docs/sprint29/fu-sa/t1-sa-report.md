# S29 Sprint-Close Security Audit Report

**Date:** 2026-04-23
**Auditor:** @security-auditor
**Branch:** mlx/sprint-29-dec032-closeout (off master)
**Scope:** 5 commits — SA-H1 remediation + LG-mechanism spike + S29 kickoff

**Note:** report persisted by orchestrator; audit agent returned findings inline due to Write-tool restriction.

## Commits Audited

```
64a8d9076b S29-LG-SPIKE-T2 verdict — outcome A (shared mechanism) [docs only]
503ebacdb2 S29-LG-SPIKE-T1 LG+Cosine iter-1 drift measurement + harness [docs only]
c73f5073af S29-CLI-GUARD-T2 pytest coverage for C++/CLI guards
73e9460a31 S29-CLI-GUARD-T1 port Cosine+{LG,ST} rejection to C++ and CLI
33ce5f1d66 S29-00 kickoff — scope (E), DEC-032 closeout + LG spike [state only]
```

Production-code surface (≈55 LoC):
- `catboost/mlx/train_api.cpp` — `TrainConfigToInternal` (27 lines added as 2 guard blocks)
- `catboost/mlx/tests/csv_train.cpp` — `ParseArgs` (28 lines: 1 include + 2 guard blocks)

Test surface (250 LoC):
- `tests/test_cli_guards.py` — 4 test cases (2 nanobind + 2 CLI subprocess)

Out of scope (per task brief):
- `docs/sprint29/lg-mechanism-spike/harness.py` (spike harness, doc directory, not production)
- Cosine numerical bounds (already passed G-SA1/2 in S28-SA, spot-checked only)

---

## Summary

**Verdict:** PASS

**Finding counts:**
- Critical: 0
- High:     0
- Medium:   0
- Low:      0
- Info:     2 (sprint-marker leak, CLI SIGABRT exit — both pre-known, non-blocking)

S29 closes SA-H1 from the S28 security audit. All three entry points to `TrainFromArrays` (Python `_validate_params`, nanobind `_core.train()`, `csv_train` CLI) now reject the two drift-degraded Cosine combinations (`Cosine+Lossguide`, `Cosine+SymmetricTree`) before any expensive work begins. Guards are correctly placed, message-consistent across languages, and test-covered.

---

## Gate-by-gate findings

### G-SA1 SA-H1 closure — **PASS**

Three callers traced end-to-end to their guards. All three now fail-closed on forbidden combinations:

| Entry point | Guard location | Throws | Translated to |
|---|---|---|---|
| Python `CatBoostMLX.fit()` / `cross_val_score` → `_validate_params` | `python/catboost_mlx/core.py:628-647` (unchanged in S29, carried from S28) | `ValueError` | `ValueError` |
| Nanobind `_core.train()` direct → `TrainFromArrays` → `TrainConfigToInternal` | `catboost/mlx/train_api.cpp:25-51` (NEW in S29) | `std::invalid_argument` | `ValueError` (nanobind auto-translation) |
| `csv_train` CLI → `ParseArgs` | `catboost/mlx/tests/csv_train.cpp:241-267` (NEW in S29) | `std::invalid_argument` | Non-zero process exit (SIGABRT via `libc++abi`) |

Critical trace point: `_core.train()` (bindings.cpp:129) → `TrainFromArrays` (train_api.cpp:217) → `TrainConfigToInternal` on the very first statement of the function body (train_api.cpp:233). Guards at lines 25-51 fire before `BuildDatasetFromArrays` (line 236) and any downstream processing. No bypass remains.

Message text is byte-identical across Python / C++ nanobind / C++ CLI guards (verified by reading all three sources). Grepping `TODO-S29-LG-COSINE-RCA` and `TODO-S29-ST-COSINE-KAHAN` returns matches in exactly the expected production sources plus the test file — a single-point-of-removal invariant for S30.

Residual attack surface considered:
- `TTrainConfig` struct can be constructed directly in C++ (not via nanobind), but no such caller exists in-tree.
- `BuildDatasetFromArrays` is called after `TrainConfigToInternal`, so there is no alternate path that bypasses the conversion.
- `csv_train` `main()` only calls `ParseArgs` once (line 4473); guards cover all CLI flows.

### G-SA2 Guard-layer correctness — **PASS**

Both C++ guards fire at the structural beginning of their functions, before any side-effecting work.

- **nanobind path**: `TrainConfigToInternal` first statement (line 25) is the LG guard; second statement (line 38) is the ST guard. No I/O, no allocation, no Metal context, no dataset parsing before the check. The `TConfig c; c.NumIterations = ...` copy begins only at line 52.
- **CLI path**: guards fire at `ParseArgs:241-267`, immediately after argument parsing completes (line 240 closes the argv loop). `main()` does not call `LoadCSV` (line 4484) / `IsBinaryFormat` (line 4482) until `ParseArgs` returns — forbidden combinations terminate before any filesystem touch. The test's use of `/dev/null` as the CSV path argument confirms this: `/dev/null` would fail to parse as CSV, but the guard fires first.

GIL note: `_core.train()` acquires `nb::gil_scoped_release` at bindings.cpp:128 before calling `TrainFromArrays`. The guard `throw` unwinds through `gil_scoped_release`'s destructor (which re-acquires the GIL) before nanobind's exception translator raises `ValueError`. This is the standard nanobind pattern; if broken, the pytest cases `test_core_train_rejects_cosine_lossguide` / `test_core_train_rejects_cosine_symmetric_tree` would fail.

### G-SA3 Error message info-disclosure — **PASS-INFO**

C++ error messages are verbatim copies of the S28 Python strings (diffed by reading both) and contain the same internal sprint markers:

- `TODO-S29-LG-COSINE-RCA`
- `TODO-S29-ST-COSINE-KAHAN`
- References to `S28-OBLIV-DISPATCH gate`, `~47% aggregate-metric drift`, `priority-queue leaf ordering`, `float32 joint-Cosine denominator accumulates precision drift`

**Assessment:** Informational only. Open-source research project; all referenced material is already in committed CHANGELOG/DECISIONS/KNOWN_BUGS docs. Markers provide greppability for the S30 cleanup (single-point removal). Inheritance from the S28-flagged Python messages is intentional and was already accepted as INFORMATIONAL in S28-SA gate G-SA3. No new disclosure introduced.

No stack traces, no internal file paths, no user input reflected. `std::invalid_argument` carries only the literal string — no `argv[i]`, no config dump.

No `TODO-S30` markers present in production code (verified via grep).

### G-SA4 Numerical bounds re-check — **PASS**

Spot-check of `csv_train.cpp` Cosine denominator guard:
- Line 1044-1045: documentation comment preserved (`den guard of 1e-20f prevents sqrt(0) on empty partitions. CPU uses double 1e-100; 1e-20f is safely above FP32 subnormal range.`)
- Line 1134: `float cosDen = 1e-20f;  // guard against sqrt(0)` intact
- Line 1264: second `cosDen = 1e-20f` seed (Depthwise path) intact
- Line 1174-1175: `cosDen += sumLeft * sumLeft * weightLeft * invL * invL + ...` accumulator unchanged

Weight pre-skip (`weightLeft * invL * invL`) and K-dim summation order unchanged vs S28 baseline. `git diff 987da0e7d5..64a8d9076b -- catboost/mlx/` confirms the ONLY line-ranges touched in `csv_train.cpp` are 69-69 (include) and 241-267 (guards). No regressions to S28 numerical work.

### G-SA5 CLI exit behavior — **PASS-INFO**

`main()` (csv_train.cpp:4472) has no top-level try/catch around `ParseArgs`. Uncaught `std::invalid_argument` → `std::terminate()` → `libc++abi` → SIGABRT → process exit code 134 (128 + signal 6).

**Risk classification:** Informational, not High.

Rationale:
1. The exit is non-zero and strictly fail-safe. No partial model is written, no dataset loaded, no Metal buffer allocated. CI/scripting layers that check `$?` / `returncode != 0` correctly detect failure.
2. The exception message is written to stderr by `libc++abi`'s terminate handler, so the `TODO-S29-*` marker is still observable for the caller.
3. Pytest cases assert `returncode != 0` (not `== 1`) specifically so assertions survive the planned S29-CR / #87 cleanup that will replace SIGABRT with a graceful `try { ... } catch (const std::invalid_argument& e) { fprintf(stderr, "%s\n", e.what()); return 1; }`.
4. Documented as carry to #87 — known technical debt, not a security finding.

No security impact: attacker cannot influence what `ParseArgs` does before the guard fires (only argv bounds-checked `strcmp` comparisons). SIGABRT does not produce a core file by default on macOS for non-setuid binaries, so no exploitable crash artifact.

---

## Critical findings

None.

## High findings

None.

## Medium findings

None.

## Low findings

None.

## Informational

### [SA-I1-S29] Sprint markers retained in error messages
- **Location:** `catboost/mlx/train_api.cpp:34,47`; `catboost/mlx/tests/csv_train.cpp:250,263`
- **Description:** Error strings contain `TODO-S29-LG-COSINE-RCA` and `TODO-S29-ST-COSINE-KAHAN`, plus references to internal gates (S28-OBLIV-DISPATCH, ~47% drift figure).
- **Rationale for retention:** Intentional cross-language grep anchors for S30 single-point removal. Already accepted as INFORMATIONAL in S28-SA G-SA3. Open-source context; all information public via committed docs.
- **Disposition:** Accept. Remove when S30-COSINE-KAHAN lands.

### [SA-I2-S29] CLI guard exits via SIGABRT (134) rather than graceful exit(1)
- **Location:** `catboost/mlx/tests/csv_train.cpp:4472` (`main` — no top-level try/catch)
- **Description:** Uncaught `std::invalid_argument` from `ParseArgs` terminates via `libc++abi` with SIGABRT. Exit code 134 is non-zero (fail-safe) but non-canonical for config errors.
- **Rationale for non-High:** Strictly fail-safe — no side effects occur before the throw. Exception message reaches stderr. Tests assert `returncode != 0` to remain green after the planned cleanup.
- **Follow-up:** Issue #87 — S29-CR wrap-up adds top-level try/catch in `main()` to normalize to `exit(1)`. No security impact in the interim.

---

## SA-H1 Closure Verdict

**CLOSED.**

SA-H1 (from S28-SA report, `docs/sprint28/fu-sa/t6-sa-report.md` line 66) identified that combination guards on Cosine+{Lossguide, SymmetricTree} were enforced only at the Python `_validate_params` layer, leaving two bypasses:
1. nanobind `_core.train()` direct invocation
2. `csv_train` CLI direct invocation

Both bypasses are now closed:
1. **nanobind bypass** — closed by `TrainConfigToInternal` guards at `train_api.cpp:25-51`, firing at the first statement of the first function invoked by `_core.train()`, before any GPU/Metal context or dataset allocation. Pytest `test_core_train_rejects_cosine_lossguide` and `test_core_train_rejects_cosine_symmetric_tree` cover this path and assert the `TODO-S29-*` marker is present in the raised `ValueError`.
2. **CLI bypass** — closed by `ParseArgs` guards at `csv_train.cpp:241-267`, firing after argv parsing, before `LoadCSV`. Pytest `test_csv_train_cli_rejects_cosine_lossguide` and `test_csv_train_cli_rejects_cosine_symmetric_tree` cover this path with `returncode != 0` + stderr-marker assertions.

Defense-in-depth achieved: a user can now reach the forbidden combination only by:
- Hand-constructing a `TConfig` in C++ and calling `Train` directly (no such caller in-tree, would require modifying MLX backend source)
- Modifying the guard sources and rebuilding

Neither is a remote- or authenticated-attacker surface; both require local source-code write access and a fresh build, at which point the attacker is already trusted.

The sprint-close artifact meets the remediation specification written in S28-SA line 71-72: *"port the two `_validate_params` guards (core.py:628-647) into `TrainConfigToInternal` (train_api.cpp:24) and into `ParseArgs` after csv_train.cpp:239"* — exact file-and-location match.

---

## S30 Follow-ups Confirmed

1. **#87 — CLI graceful exit wrap** (SA-I2-S29 remediation): add top-level try/catch in `main()` to replace SIGABRT(134) with `exit(1)`. Tests already tolerant (`returncode != 0`).
2. **#86 — S29-BRANCH-DECISION**: human checkpoint on merging `S29-ST-COSINE-KAHAN` + `S29-LG-COSINE-RCA` into a single `S30-COSINE-KAHAN` task (per spike verdict, commit 64a8d9076b). When Kahan/Neumaier lands and parity is re-gated, remove all three guard blocks (Python + C++ nanobind + C++ CLI) and the 4 pytest cases in a single atomic commit.
3. Sprint-marker removal (SA-I1-S29): tied to the S30 guard removal above — the `TODO-S29-*` tokens exist precisely to make the S30 cleanup a grep-and-delete operation.

---

## Positive Findings

1. **Message-level verbatim parity across three languages.** Python / C++ nanobind / C++ CLI error strings are byte-identical (read & diffed). No drift. Minimizes risk of partial remediation in S30.
2. **Grep-anchored single-point-of-removal invariant.** `TODO-S29-LG-COSINE-RCA` and `TODO-S29-ST-COSINE-KAHAN` tokens appear in exactly the expected production sources plus the test file. `grep -rn 'TODO-S29-LG-COSINE-RCA'` gives the S30 removal checklist for free.
3. **Test-forward-compatibility design.** `returncode != 0` (not `== 1`) assertion already anticipates the S29-CR/#87 graceful-exit cleanup without assertion churn. Evidence of process maturity.
4. **Guards placed at the structural top of entry functions.** No risk of a future refactor inserting work above them — the `TConfig c;` initialization at train_api.cpp:52 comes AFTER the guards, and argv parsing at csv_train.cpp:178-240 comes BEFORE. Both natural code-structure seams protect the invariant.
5. **nanobind exception translation used correctly.** `std::invalid_argument` is a standard exception nanobind auto-translates to `ValueError`; no custom translator needed. GIL release/acquire is handled by RAII via `gil_scoped_release`'s destructor on unwind.
6. **No Python-layer regression.** `python/catboost_mlx/core.py:628-647` unchanged from S28 baseline (verified via `git diff 987da0e7d5..64a8d9076b -- python/catboost_mlx/core.py` returning empty). Defense-in-depth preserved; C++ layer adds belt-and-suspenders.
7. **No numerical-guard regressions.** S28 Cosine `cosDen = 1e-20f` seeds and weight pre-skip logic unchanged in `csv_train.cpp`. S29 diff scope limited to additive guard blocks + 1 `<stdexcept>` include.
8. **Spike code quarantined.** `docs/sprint29/lg-mechanism-spike/harness.py` lives under `docs/`, is not compiled, is not imported from production, and is not on any installation path — correctly isolated from the audit surface.

---

## Recommendations

1. **Merge and close S29.** SA-H1 is closed; no blocking security findings. Recommend proceeding to #86 (S29-BRANCH-DECISION) and #87 (CLI exit cleanup) as already planned.
2. **In S30, remove guards atomically.** When `S30-COSINE-KAHAN` lands and parity is re-gated, remove all three guard blocks (Python + nanobind + CLI) plus the 4 pytest cases in one commit. Use `grep -rn 'TODO-S29-(LG|ST)-COSINE'` as the removal checklist. Do NOT remove one without the others — a Python-only removal would recreate SA-H1.
3. **Do not re-audit numerical bounds in S30 unless the Kahan port touches them.** S28-SA gates G-SA1/2 covered this surface; S29 did not regress it. Focus next SA on the Kahan compensated-summation correctness and the removal of the guards.
4. **No action needed on SA-I1-S29 / SA-I2-S29.** Both are tracked follow-ups; neither impacts deployment.
