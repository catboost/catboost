# S29 Sprint-Close Code Review — T1 CR Report

**Date:** 2026-04-23
**Reviewer:** @code-reviewer
**Branch:** `mlx/sprint-29-dec032-closeout` (off `master` @ `987da0e7d5`)
**Commits reviewed (oldest → newest):**

| SHA | Title |
|-----|-------|
| `33ce5f1d66` | S29-00 kickoff — scope (E), DEC-032 closeout + LG spike |
| `73e9460a31` | S29-CLI-GUARD-T1 port Cosine+{LG,ST} rejection to C++ and CLI |
| `c73f5073af` | S29-CLI-GUARD-T2 pytest coverage for C++/CLI guards |
| `503ebacdb2` | S29-LG-SPIKE-T1 LG+Cosine iter-1 drift measurement + harness |
| `64a8d9076b` | S29-LG-SPIKE-T2 verdict — outcome A (shared mechanism) |

**Parity state:** Python parity suite 28/28 at all S29 commits (prior evidence; not re-run by this review).

**Note:** report persisted by orchestrator; reviewer returned findings inline due to Write-tool restriction.

---

## Summary

**Verdict: PASS-WITH-NITS.**

The sprint lands its declared scope (E) cleanly. CLI-GUARD port is a faithful verbatim mirror of the Python `_validate_params` guards, placed at the correct layer in both the nanobind entry (`TrainConfigToInternal`) and the CLI entry (`ParseArgs`), with pytest coverage that is purposely resilient to a future `main()` try/catch cleanup. The LG mechanism spike produces data that the verdict reads accurately, and the verdict is honest about its moderate-confidence limitations. Commits are atomically scoped. No must-fix issues; three should-fix items and three nits, all cosmetic or doc-level.

**Counts:** Must Fix: 0 | Should Fix: 3 | Nits: 3 | Praise: 5.

---

## Gate-by-gate findings

### G-CR1 Guard correctness — PASS

The LG and ST guard bodies in `catboost/mlx/train_api.cpp:25-51` and `catboost/mlx/tests/csv_train.cpp:241-267` are **byte-for-byte identical** to the Python `_validate_params` guards at `python/catboost_mlx/core.py:627-647`, including:

- Combination logic (`ScoreFunction == "Cosine" && GrowPolicy == "Lossguide"` / `"SymmetricTree"`).
- Exact error-message text, quoting style, and `TODO-S29-*` markers.
- Exception type: `std::invalid_argument` (nanobind maps this to Python `ValueError`, matching the Python layer's `ValueError` raise).
- Comment tag scheme (`S28-LG-GUARD` / `S28-ST-GUARD`) consistent with the sprint-28 Python commit `b9577067ef`.

The `#include <stdexcept>` was already pulled in for `std::runtime_error` in `train_api.cpp:18`; adding the new `<stdexcept>` include in `csv_train.cpp:69` is correct and explicitly commented.

### G-CR2 Dispatch plumbing — PASS

Guards are placed at the top of each entry function, **before any `ScoreFunction` consumer is invoked**:

- `TrainConfigToInternal` guards fire at lines 28/41, before the `TConfig c; c.X = tc.X;` copy block at line 52+. Nothing reads `tc.ScoreFunction` downstream before the guard.
- `ParseArgs` guards fire at lines 244/257, after the full arg-parse loop (so both `--grow-policy` and `--score-function` have been read) but before `ParseArgs` returns, and therefore before `main()` calls `ParseLossType` or `LoadCSV`. The guard fires **before the CSV is opened** — so passing `/dev/null` in the pytest is safe.

There is no alternative entry point that bypasses either guard: the nanobind module funnels all train calls through `TrainConfigToInternal`, and the CLI binary has only one `main()`. The Python `_validate_params` guard is preserved as the first-line defense; the C++ guards are correctly framed as "defense-in-depth" in the commit message and in-code comments. No bypass path exists.

### G-CR3 Commit atomicity (DEC-012) — PASS

Each commit touches one orthogonal concern:

| Commit | Files | Verdict |
|--------|-------|---------|
| `33ce5f1d66` | 4 state files only | atomic — pure kickoff |
| `73e9460a31` | 2 C++ source files | atomic — guard port only |
| `c73f5073af` | 1 new test file | atomic — test coverage only |
| `503ebacdb2` | 5 doc/data files, no prod code | atomic — spike artifacts only |
| `64a8d9076b` | 1 doc file (verdict) | atomic — verdict write-up |

No stray edits leaked across boundaries. Order is the correct "kickoff → implement → test → measure → verdict" narrative.

### G-CR4 Test quality — PASS-WITH-NITS

`tests/test_cli_guards.py` covers exactly the four required combinations:

| Test | Path | Assertion strategy |
|------|------|---------------------|
| `test_core_train_rejects_cosine_lossguide` | nanobind `_core.train` | `pytest.raises(ValueError)` + `TODO-S29-LG-COSINE-RCA` substring |
| `test_core_train_rejects_cosine_symmetric_tree` | nanobind `_core.train` | `pytest.raises(ValueError)` + `TODO-S29-ST-COSINE-KAHAN` substring |
| `test_csv_train_cli_rejects_cosine_lossguide` | subprocess(`bin/csv_train`) | `returncode != 0` + `TODO-S29-LG-COSINE-RCA` in stderr |
| `test_csv_train_cli_rejects_cosine_symmetric_tree` | subprocess(`bin/csv_train`) | `returncode != 0` + `TODO-S29-ST-COSINE-KAHAN` in stderr |

Strengths:
- Binary resolved at `python/catboost_mlx/bin/csv_train` (the post-#82 build), skipped if missing via `@_SKIP_CLI` — avoids false-red when the binary hasn't been rebuilt. Explicit docstring explains *why* the repo-root `csv_train` is not used.
- `returncode != 0` (not `== 1`) is the right call: `main()` (`csv_train.cpp:4472`) has no top-level `try/catch`, so the uncaught `std::invalid_argument` currently raises SIGABRT (-6 / 134 on macOS). The `!= 0` assertion survives both current behavior and a future cleanup that replaces the abort with `return 1`. That reasoning is explicitly documented in the test file header.
- Assertion anchors on the `TODO-S29-*` marker rather than full error text — minimizes coupling to wording changes while still proving the guard fired (not some other early failure).

Nits (see below): binary-path coupling could be centralized; `/dev/null` as positional arg works by accident of guard-before-file-open ordering.

### G-CR5 Verdict-doc integrity — PASS-WITH-NITS

Numerical claims in `verdict.md` reconcile with `data/iter1_drift.json` and `data/iter_curve.csv`:

- "iter-1 mean drift 0.0024%" — matches `"mean_drift_pct": 0.0024` (computed: `(0.0046+0.0015+0.0010)/3 = 0.00237`). ✓
- Per-seed iter-1: "0.0046 / 0.0015 / 0.0010" — matches JSON per-seed. ✓
- "50-iter drift 0.0029 – 0.1970 per seed" — CSV seed=0→0.0921, seed=1→0.1970, seed=2→0.0029; range cited is correct. ✓
- "iter curve stays <0.2% through iter=50" — max is 0.197 (seed=1 at iter=50). Technically below 0.2%, but only by 3 basis points. Worth a word of honesty ("up to 0.197%") — see SF-1.
- "~300× smaller" — `0.77 / 0.0024 ≈ 320`. OK as a round figure.
- "iter=1 trees bit-identical CPU-vs-MLX" (from commit message `64a8d9076b`) — the verdict body itself says "bit-identical BFS split sequences `[0,0,0,1,1,1,1]`" and "identical root feature." `tree_structure_iter1.json` confirms feature sequences match exactly, but the two sides record *different domain* thresholds (CPU float `border` vs MLX integer `bin_threshold`), so "bit-identical trees" is strictly stronger than what was measured. See SF-2.
- Cell-mismatch disclosure correctly reconciles the apparent contradiction with `t5-gate-report`'s 14% figure (pre-S28 algorithmic divergence vs post-S28 float drift) — exactly the kind of audit-trail honesty DEC-031 asks for. Praise item below.

### G-CR6 Project conventions — PASS

- No `Co-Authored-By:` trailers in any of the five commit messages. ✓
- Commit message format `[mlx] sprint-29: <task-id> <description>` matches the S28 precedent. ✓
- C++ naming: PascalCase fields (`ScoreFunction`, `GrowPolicy`), `T`-prefixed types (`TConfig`, `TTrainConfig`) — matches CatBoost/MLX style. ✓
- Inline comments explain **why** (nanobind exception translation, cross-language grep markers) rather than restating the code. ✓
- Python test file uses `pytest.raises(ValueError)` idiom, `from pathlib import Path`, descriptive docstrings with DEC/task refs. ✓

### G-CR7 Dead code / debt — PASS-WITH-NITS

One dead function: `docs/sprint29/lg-mechanism-spike/harness.py:228 run_secondary(rows)` is defined but never called — `main()` inlines the loop at lines 455-469 to share data with the primary pass rather than recomputing iter=1 rows. The inline version is the one that actually ran. See SF-3.

No undocumented TODOs introduced. The `TODO-S29-LG-COSINE-RCA` and `TODO-S29-ST-COSINE-KAHAN` markers are intentional grep anchors, tracked in DEC-034 / #86 (branch decision) and folded into the proposed S30-COSINE-KAHAN task per the verdict's recommendation.

The comment on `train_api.cpp:18` (`#include <stdexcept>   // std::runtime_error — BUG-007 groupIds sortedness contract`) is now stale — the same include also services the new `std::invalid_argument` throws in `TrainConfigToInternal`. Minor nit (see N-1).

---

## Must Fix

*(None.)*

---

## Should Fix

### SF-1. `verdict.md:8` — "<0.2%" wording undercounts seed=1 iter=50

**Problem:** Verdict says "iter curve stays <0.2% through iter=50." Actual max is `0.1970%` at iter=50 seed=1 (per `iter_curve.csv` line 18). The claim is technically true but only 3 basis points below the round threshold, and for a DEC-032-closing audit doc an honest "up to 0.197%" framing is preferable.

**Suggestion:** Replace "stays <0.2% through iter=50" with "reaches 0.197% at iter=50 seed=1, and stays below 0.2% on all measured points." No numerical claim changes; the narrative survives a post-Kahan re-read.

### SF-2. `verdict.md` / commit `64a8d9076b` message — "iter=1 trees bit-identical" is stronger than the measurement

**Problem:** The commit message states "iter=1 trees bit-identical CPU-vs-MLX." The verdict body more carefully says "bit-identical BFS split sequences" and "identical root feature" — both supported by `tree_structure_iter1.json`. But leaf values and thresholds weren't compared bit-for-bit (MLX stores quantized integer `bin_threshold`, CPU stores dequantized float `border`; the harness itself acknowledges this at `harness.py:287-290`).

**Suggestion:** In any future doc/decision that quotes this result (DEC-034 filing, S30 RCA), use "BFS feature-index sequence bit-identical" or "structural skeleton identical" rather than "trees bit-identical." The commit message is already landed; the risk is downstream claim inflation.

### SF-3. `harness.py:228-244` — dead `run_secondary(rows)` function

**Problem:** `run_secondary` is defined but unused; `main()` inlines an equivalent loop at lines 455-469 to dedupe iter=1 rows with `primary["per_seed"]`. The inline version is the authoritative one.

**Suggestion:** Delete `run_secondary`, or refactor `main()` to call it (and have it accept the primary rows to skip). Either is acceptable; current state is confusing to a future re-runner. Non-blocking (harness is a one-shot measurement tool).

---

## Nits

### N-1. `catboost/mlx/train_api.cpp:18` — stale include-comment

Comment reads `// std::runtime_error — BUG-007 groupIds sortedness contract`. After this sprint, `<stdexcept>` also backs the two `std::invalid_argument` throws in `TrainConfigToInternal`. Trivial update: `// std::runtime_error (BUG-007 groupIds) + std::invalid_argument (S28-LG/ST-GUARD)`.

### N-2. `test_cli_guards.py:30-37` — binary path is correct but brittle

Constant `_CSV_TRAIN_BIN = _REPO_ROOT / "python" / "catboost_mlx" / "bin" / "csv_train"` is hard-coded. If CI or a future packaging change relocates the binary (wheel install, alternate build dir), the test silently skips rather than failing. Consider an env override (`CATBOOST_MLX_CSV_TRAIN_BIN`) before the default. Not a sprint blocker — current layout matches what `setup.py` produces.

### N-3. `test_cli_guards.py:192-196, 225-229` — `/dev/null` positional CSV arg works by ordering luck

The test relies on guards firing inside `ParseArgs` *before* `LoadCSV` is called. That's true today (guards at 244/257, `LoadCSV` at 4484), and the test file even calls this out ("Guard fires in ParseArgs … before file open — so /dev/null as the CSV path argument is safe"). Future-proofing: if someone reorders guards below file open or adds an early `config.CsvPath` read, these tests will fail with a file error rather than a guard error. A tiny nonexistent-file path like `/nonexistent-csv-for-cli-guard-test.csv` would be equally safe and slightly clearer about intent.

---

## S30 Follow-ups (confirmed from verdict)

1. **S30-COSINE-KAHAN** — merged task per verdict recommendation. Applies Kahan/Neumaier compensated summation to the shared joint-Cosine denominator accumulator; gates both ST+Cosine and LG+Cosine behind a single post-fix parity check; removes both Python and C++ guards together.
2. **Deep-LG outcome-B residual watch (S31 if triggered)** — verdict explicitly keeps outcome B open for deep LG cells (`depth > 3`, `max_leaves > 8`, priority-queue-stressed configs) not exercised by the spike. Escalate to a dedicated spike in S31 only if post-Kahan drift persists on those configs.
3. **`main()` try/catch cleanup in `csv_train.cpp`** — deferred from this sprint. Replaces current `std::terminate` / SIGABRT path with `catch (const std::invalid_argument&) { fprintf(stderr, …); return 1; }` so `csv_train` errors with a clean `exit(1)` rather than a `-6`. Tests already accommodate this (`returncode != 0`), so the cleanup is a no-op for test signal.

---

## Praise

- **`73e9460a31` verbatim guard port.** Matching the Python message text character-for-character (quotes, spacing, hyphenation, TODO markers) is the right call for a defense-in-depth mirror: it makes the C++ layer greppable from a single token and will make the post-Kahan cleanup a find-and-delete across exactly four well-tagged call sites. The `S28-LG-GUARD` / `S28-ST-GUARD` comment tags propagate the audit trail cleanly.
- **`c73f5073af` `returncode != 0` forward-compat assertion.** Most reviewers would write `== 1` and produce test churn on the next `main()` cleanup. The rationale is even explained in the test docstring — saves the next on-call a head-scratch.
- **`503ebacdb2` README transparency on guard-bypass requirement.** The README is explicit that reproduction requires a local (uncommitted) guard bypass and that production code ships no flag to disable the guard. That is exactly the right discipline for a spike that requires forbidden-config runs — no `--allow-unsafe-cosine-lg` back door leaks into the CLI or nanobind surface.
- **`503ebacdb2` data/verdict separation.** The primary JSON explicitly refuses to classify outcome A/B/C and points at task #85 as authoritative. Keeps the harness replayable without a silent data-drift into the verdict.
- **`64a8d9076b` cell-mismatch disclosure.** Proactively calling out that the t5-gate-report's 14% LG figure was pre-S28 algorithmic divergence (MLX hardcoded L2 Newton vs CPU Cosine, closed in commit `0ea86bde21`), not precision drift, prevents a future reader from concluding LG+Cosine precision is 300× worse than it actually is. That kind of post-hoc honesty is what makes DEC docs trustable years later.

---

## Pattern watch

- **Cross-language-grep marker discipline (`TODO-S29-*`) is paying off.** The markers now sit in 4 files (Python core, C++ train_api, C++ csv_train, Python tests) and future code-archaeologists will find all guard sites from any one token. Keep this pattern on follow-ups.
- **Defense-in-depth at API boundaries.** "Python-layer guard as first defense + C++ boundary guard as second defense" is a good template for any future forbidden-config rejections (e.g. when ordered boosting rolls out with restricted loss types).
- **Spike-plus-verdict as two commits, not one.** `503ebacdb2` (data) + `64a8d9076b` (interpretation) is the right granularity — reruns overwrite the data but not the verdict, and a verdict update later can cite a replaced data snapshot without losing commit history.
