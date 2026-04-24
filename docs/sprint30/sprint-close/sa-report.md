# Sprint 30 Close-out — Security Audit Report

**Branch:** `mlx/sprint-30-cosine-kahan`
**Tip:** `24a0e829b8`
**Auditor:** security-auditor
**Date:** 2026-04-24

## Verdict: **PASS-WITH-FINDINGS**

No CRITICAL or HIGH findings. No remotely-exploitable vulnerabilities introduced by Sprint 30. SA-H1 defence-in-depth (Python -> nanobind -> CLI) is intact and un-weakened by the sprint. K4 fp64 widening and Fix 2 fp64 gain scalar introduce no unsafe cast or TOCTOU. Instrumentation is airtight behind compile-time macros. No secrets, PII, or credentials in sprint30/ artifacts. Three LOW and three INFO findings, plus one outstanding minor (SA-I2, carried from S29), are listed below. Sprint 30 may close and the branch may be merged.

---

## Scope reviewed

Sprint 30 was a correctness sprint. Code-change footprint vs `master`:

| File | Lines (ins/del) | Nature |
|---|---|---|
| `catboost/mlx/tests/csv_train.cpp` | +507 / -62 | K4 fp64 widening; Fix 2 gain->double; T1/D2/T3 instrumentation and gates |

Verified **unchanged** vs `master`:

- `catboost/mlx/train_api.cpp` — nanobind entry-point guards intact.
- `python/catboost_mlx/core.py` — Python `_validate_params` guards intact.
- All Metal kernel sources (`catboost/mlx/kernels/*.metal`, dispatcher, launcher) — unchanged.
- `tests/test_cli_guards.py` — unchanged; still asserts runtime guard fire.
- No dependency additions; no `package.json`, `pyproject.toml`, vendored library, or `environment.yml` change.

Threat model: an attacker who can influence (a) the `--score-function` / `--grow-policy` CLI flags or `TrainConfig` fields, (b) the CSV payload, or (c) the environment variables of a privileged training host. All surfaces pre-existed S30; the sprint did not add new surfaces.

---

## Executive Summary

Sprint 30 was correctness-only (cosine + Kahan / fp64 numerical audit) but landed two narrow code changes in the CLI driver (`csv_train.cpp` only):

1. **K4:** widen cosine-joint accumulators (`cosNum`, `cosDen`) from `float` to `double` so per-term products are computed in fp64 throughout every partition / dim loop.
2. **Fix 2:** widen the per-candidate gain scalar, the argmax accumulator, and the persisted `TBestSplitProperties::Gain` from `float` to `double`. The priority-queue key in Lossguide (`TLeafCandidate::Gain`) was widened in lock-step.

Instrumentation behind `COSINE_RESIDUAL_INSTRUMENT` / `COSINE_D2_INSTRUMENT` and a T3-measure bypass gate behind `COSINE_T3_MEASURE` were added. All three macros are compile-time; none is defined by any CMakeLists.txt; all instrumentation state (`g_cosInstr`, `<filesystem>` include, writer functions) is gated; the default (production) build has zero footprint.

The SA-H1 closure from Sprint 29 (Python -> nanobind -> CLI defence-in-depth rejection of `Cosine + {Lossguide, SymmetricTree}`) is preserved verbatim in all three layers. The S30 bypass macros (`COSINE_T3_MEASURE`, `COSINE_RESIDUAL_INSTRUMENT`) relax the **CLI** guard only, and only at compile time — the production CLI binary at `python/catboost_mlx/bin/csv_train` has neither macro and thus still fires the guard. `tests/test_cli_guards.py` is specifically wired to that production binary and verifies the runtime behaviour (see G1-CLI gate).

---

## Confirmation on each deliverable

### Point 1 — SA-H1 surface remains closed: **CONFIRMED**

All three guard sites present and un-weakened. Greps:

| Marker | Expected sites | Observed sites |
|---|---|---|
| `TODO-S29-LG-COSINE-RCA` | Python, nanobind, CLI | `python/catboost_mlx/core.py:634` ; `catboost/mlx/train_api.cpp:34` ; `catboost/mlx/tests/csv_train.cpp:387` |
| `TODO-S29-ST-COSINE-KAHAN` | Python, nanobind, CLI | `python/catboost_mlx/core.py:644` ; `catboost/mlx/train_api.cpp:47` ; `catboost/mlx/tests/csv_train.cpp:405` |

Plus identical markers in `tests/test_cli_guards.py` (test-substring assertions). `git diff master..HEAD -- python/catboost_mlx/core.py catboost/mlx/train_api.cpp` shows **no edits** — only the CLI file changed. Guard text is byte-identical across layers (single-point-of-removal grep invariant preserved).

The S30 decision explicitly stated **guards stay in place — mechanism not fixed** (`DEC-035` PARTIALLY CLOSED, `DEC-036` OPEN). The CLI gating additions wrap the guard in `#ifndef COSINE_T3_MEASURE` and `#if !defined(COSINE_RESIDUAL_INSTRUMENT) && !defined(COSINE_T3_MEASURE)` blocks — default builds (no `-D` flag) still fire the guard.

Runtime evidence: the production binary at `python/catboost_mlx/bin/csv_train` (used by pytest) is built without any of the three macros and exits non-zero with the marker on stderr for both rejected configs. `test_cli_guards.py` re-verifies this on every test run (S29 G1-CLI gate).

### Point 2 — K4 + Fix 2 widening introduces no unsafe cast or TOCTOU: **CONFIRMED**

Per-site audit (six FindBestSplit call-sites for Cosine: OneHot/Ordinal x three functions — `FindBestSplit` oblivious, `FindBestSplitPerPartition` depthwise OneHot & ordinal):

- All `cosNum_d`, `cosDen_d` accumulators are `double`.
- All inputs widened once at the top of each term via explicit `static_cast<double>` (line 1325-1334, 1497-1506, 1723-1732, 1807-1816, and equivalents in the instrumentation-only shadow).
- All `totalGain` / `gain` / `perturbedGain` locals widened to `double`.
- Argmax comparators (`perturbedGain > bestGain`, `perturbedGain > bestGains[p]`) operate on `double` vs `double` — no `float`-`double` mixing.
- The priority-queue comparator `TLeafCandidate::operator<` (line 3729) compares two `double` fields.
- `TBestSplitProperties::Gain` widened to `double` (line 229). The residual `Score` member is kept as `float` **for legacy-output compatibility only**; verified via grep that no downstream code compares or argmaxes on `Score`.
- Noise perturbation `noiseScale * noiseDist(*rng)` is computed in `float` then explicitly promoted with `static_cast<double>` before accumulating into `perturbedGain` (line 1355, 1565, 1749, 1831). No silent truncation.

TOCTOU: the K4 widening is pure per-thread arithmetic inside a function with no shared mutable state between the widening cast and its use; instrumentation is guarded by a boolean (`g_cosInstr.dumpCosDen`) set before the call and cleared after, always on the same thread and always under a macro gate — no race window.

One minor precision observation (not a security issue): `catboost/mlx/tests/csv_train.cpp:3104` defines `extractPropField` with return type `float`, so the snapshot-restore path truncates a JSON-parsed gain through `float` before assigning to `prop.Gain` (now `double`). This is a residual ~1 ULP drop only applied to snapshot metadata (feature importance roll-up at `tree.SplitProps[level].Gain`), never to live argmax decisions (which are recomputed each iteration from gradients). Recorded as **SA-I3** below.

### Point 3 — Instrumentation gate is airtight: **CONFIRMED**

`COSINE_RESIDUAL_INSTRUMENT` appearances (15 sites in `csv_train.cpp`):

- All definitions/declarations of instrumentation types (`TCosineResidualInstrument`, `g_cosInstr`, `WriteCosAccumCSV`, `WriteGainScalarCSV`, `PrintResidualSummary`) are inside `#ifdef COSINE_RESIDUAL_INSTRUMENT` (lines 110-201).
- `#include <filesystem>` and `#include <cassert>` (pulled in by the instrumentation) are inside the gate (lines 111-112).
- All instrumentation **reads/writes** of global state, shadow accumulator variables (`cosNum_f32_shadow`, `cosDen_f32_shadow`), record buffers, and stderr summary prints are `#ifdef`-gated (line 1431, 1507, 1532, 4178, 4202, 4330, 4353, 4594, 4886).
- `COSINE_D2_INSTRUMENT` implies `COSINE_RESIDUAL_INSTRUMENT` via a clean `#ifndef / #define` forwarding block (lines 103-108).
- `COSINE_T3_MEASURE` relaxes only the two `csv_train.cpp` guards; it never reaches production code (`train_api.cpp`, `core.py`).
- No CMakeLists.txt in the repo defines any of the three macros. Grep across `**/CMakeLists.txt` returns zero hits. Release builds (via `build_binaries.py`, per-sprint `python/catboost_mlx/bin/csv_train` build) cannot turn them on.

One precision note (SA-L3 below): when `COSINE_RESIDUAL_INSTRUMENT` is defined for research, the CLI accepts a `COSINE_RESIDUAL_OUTDIR` env-var path (line 4892). In instrumented builds an attacker with control of env-vars could redirect instrumentation CSV output to an arbitrary directory — **not a concern for production** (the macro is never set in release builds). Flagged for future hardening of research harnesses only.

### Point 4 — No secrets, PII, or credentials in sprint30 artefacts: **CONFIRMED**

Ran pattern scans across the 13 verdict docs + `.csv` / `.json` / `.py` artifacts (~419k lines, ~200 files):

- `password|secret|api[_-]?key|credential|private[_-]?key`, `AKIA[0-9A-Z]{16}`, `sk-[A-Za-z0-9]{20,}`, `ghp_[A-Za-z0-9]{36}`, `xox[baprs]-...`, `BEGIN (?:RSA|PRIVATE|OPENSSH)`, `ssh-rsa|ssh-ed25519`, `Bearer ...`: **0 hits** on real-credential patterns.
- `token` matches (168) resolve to the CatBoost model JSON schema (`end_of_sentence_token_policy`, `start_token_id`, `token_level_type` — text-feature tokenizer fields from CatBoost 1.2.10). Benign.
- Email pattern: a single public upstream maintainer address `akhropov@yandex-team.com` appears in `catboost_version_info` embedded by the upstream 1.2.10 CLI when it writes the reference-CPU-CatBoost JSON models; same with `/Users/runner/work/catboost/...` (GitHub Actions runner build path). Both are upstream provenance metadata, already public on the CatBoost GitHub, and not user-controllable. See SA-I1 below.
- Local macOS username `ramos` appears in `docs/sprint30/d3-lg-outcome-ab/{run_discriminator.py:48, verdict.md:253}` as an absolute repro path. Already disclosed by git author `RR-AMATOK` and present in prior-sprint docs; no new surface. See SA-L2 below.

### Point 5 — SA-I2 status: **UNRESOLVED (carried to S31)**

`csv_train:main()` (lines 4883-5437) has **no top-level try/catch**. `ParseArgs` (and guard throws inside it), and any `std::invalid_argument` from downstream, terminate via `std::terminate` -> `SIGABRT` (exit code 134 / -6) rather than a graceful `exit(1)`. This remains the S29-SA recorded minor finding `SA-I2-S29` tracked as TODOS.md #95. **Sprint 30 did not touch this pattern** — the changes inside `main()` are limited to the `COSINE_RESIDUAL_INSTRUMENT` initialisation block at the top (line 4886-4918). `grep -n "try\\s*{" / "catch\\s*(" csv_train.cpp` shows no addition of a top-level handler; the three pre-existing `try/catch` sites remain scoped to local I/O paths (CSV load, snapshot dir creation, float parsing).

SA-I2 carries to S31 unchanged. Severity remains LOW (no crash-dump exfil, no information leak beyond what `std::invalid_argument::what()` already sends to stderr; `test_cli_guards.py` is already wired with `returncode != 0` rather than `== 1`, so the future fix is a drop-in).

---

## Findings

### [SA-H1-S30] SA-H1 guard layer verification
- **Severity:** INFO (no finding — confirmation)
- **Category:** OWASP A04 / defence-in-depth
- **Description:** SA-H1 closure from Sprint 29 (reject Cosine + {Lossguide, SymmetricTree} at Python, nanobind, and CLI layers) is intact and un-weakened. See Point 1 above for evidence.
- **Action:** None. Verified intact.

### [SA-L1-S30] Sprint-30 bypass binaries are not gitignored
- **Severity:** LOW
- **Category:** OWASP A08 Data-Integrity / CI-CD hygiene
- **Location:** `.gitignore:110-119`; working-tree untracked `csv_train_t3`, `csv_train_d2`, `csv_train_d2_redux`, `csv_train_instrument`, `csv_train_t3_k4_orig`.
- **Description:** `.gitignore` enumerates known-good CLI binary names (`/csv_train`, `/csv_train_profiled`, `/csv_train_ref`, ...) but does not use a `/csv_train_*` glob. The S30 measurement scripts compile to names (`csv_train_t3`, `csv_train_instrument`, `csv_train_d2`, ...) that are not covered by any existing pattern. These are **guard-bypassed** binaries — they disable the SA-H1 CLI guard at compile time.
- **Attack scenario:** A developer runs `git add .` in a frustrated-commit moment and stages a guard-bypassed binary into the tree; a downstream user `git clone`s the repo and runs the binary at the repo root, now producing numerically-invalid Cosine + ST / Cosine + LG models without any error. Outcome is silent data corruption, not system compromise — hence LOW.
- **Evidence:** `git status --porcelain` reports the five binaries as `??` (untracked but not ignored). `git check-ignore` on any of them returns empty.
- **Remediation:** Replace the explicit per-name entries with a catch-all glob:

```
/csv_train
/csv_train_*
/csv_predict
/csv_predict_*
```

- **Verification:** After patching, re-run `git check-ignore csv_train_t3 csv_train_d2 csv_train_d2_redux csv_train_instrument csv_train_t3_k4_orig` and confirm each matches a pattern line.

### [SA-L2-S30] Username in absolute-path repro
- **Severity:** LOW
- **Category:** OWASP A05 / information disclosure
- **Location:** `docs/sprint30/d3-lg-outcome-ab/run_discriminator.py:48` ; `docs/sprint30/d3-lg-outcome-ab/verdict.md:253`
- **Description:** Two files hard-code the absolute macOS + iCloud-sync path under `/Users/ramos/Library/Mobile Documents/...`. Discloses the macOS local account username.
- **Attack scenario:** Pure disclosure; username `ramos` is already visible in `user.name = RR-AMATOK` history and in several prior sprints. No new surface.
- **Remediation:** Either (a) replace with `$(git rev-parse --show-toplevel)` / `Path(__file__).resolve().parents[N]`, or (b) ignore per policy.
- **Verification:** re-run `grep -rn "/Users/ramos" docs/sprint30/` and confirm 0 hits.

### [SA-L3-S30] `COSINE_RESIDUAL_OUTDIR` env-var writable path (instrumentation builds only)
- **Severity:** LOW
- **Category:** OWASP A04 / insecure design (research harness only)
- **Location:** `catboost/mlx/tests/csv_train.cpp:4892`
- **Description:** Under `COSINE_RESIDUAL_INSTRUMENT`, the CLI reads `$COSINE_RESIDUAL_OUTDIR` and writes instrumentation CSV files to that directory via `std::filesystem::create_directories` + `fopen("w")`. No path validation, no canonicalisation, no restriction to a sandbox root. A malicious env-var setter could redirect writes to `/etc/` or similar.
- **Attack scenario:** Attacker who can set env-vars on a privileged host runs the instrumented binary and overwrites adjacent files. **Not reachable in release builds** (macro never defined); valid only on explicit research invocations. Hence LOW.
- **Remediation:** (a) validate `envDir` against a canonical `docs/sprint30/**/data/` prefix, or (b) refuse to run if EUID != UID (instrumentation is a developer tool). Defer to S31 if a research harness is still needed.
- **Verification:** add a unit-test that sets `COSINE_RESIDUAL_OUTDIR=/etc/passwd.bogus` and asserts creation is refused.

### [SA-I1-S30] CatBoost upstream maintainer email and runner path in reference model JSONs
- **Severity:** INFO
- **Category:** OWASP A05 / information disclosure (third-party provenance)
- **Location:** `docs/sprint30/d3-lg-outcome-ab/data/cpu_ml{8,16,31,64}_s{0,1,2}.json` (12 files, in `model_info.catboost_version_info`)
- **Description:** The upstream `catboost` 1.2.10 CLI, when saving a JSON model, embeds its own build metadata — commit hash, author (`akhropov <akhropov@yandex-team.com>`), GitHub-Actions runner absolute path (`/Users/runner/work/catboost/...`). The sprint30 reference JSONs therefore carry this upstream string. Public (same string in every CatBoost 1.2.10 JSON model worldwide); not local PII.
- **Remediation:** None required. If future sprints archive model JSONs for redistribution, consider a post-processing pass that strips `model_info.catboost_version_info`.
- **Verification:** `grep -l yandex-team.com docs/sprint30/` and confirm only upstream-produced JSON artefacts match.

### [SA-I2-S29] csv_train:main() graceful exit (carried from S29)
- **Severity:** LOW (unchanged from S29)
- **Category:** OWASP A09 / failure resilience
- **Location:** `catboost/mlx/tests/csv_train.cpp:4883-5437`
- **Description:** `main()` has no top-level `try { ... } catch (const std::exception&) { ... return 1; }`. The SA-H1 `throw std::invalid_argument` inside `ParseArgs`, snapshot parser throws, and miscellaneous `std::runtime_error`s propagate to `std::terminate` -> `SIGABRT` (134 / -6). Aesthetic, not a security issue — no crash-dump leak beyond `what()` to stderr. `tests/test_cli_guards.py` is already wired for this (`assert result.returncode != 0`).
- **Status:** Tracked as TODOS.md `#95 S30-T5-CLI-EXIT-WRAP`. Not addressed in S30. Carried to S31 unchanged.

### [SA-I3-S30] Snapshot-restore precision truncation (not security-relevant)
- **Severity:** INFO
- **Location:** `catboost/mlx/tests/csv_train.cpp:3104,3112`
- **Description:** `extractPropField` is declared `-> float`; `std::atof` (double-producing) is truncated at the return boundary, then widened to `double` on assign to the (now fp64) `prop.Gain`. A ~1 ULP drop on snapshot metadata only; never reaches argmax comparisons (which recompute gain from gradients each iteration).
- **Attack scenario:** None — numerical roll-up drift in feature-importance reporting, no security impact.
- **Remediation:** Widen `extractPropField` lambda to `-> double` when SA-I2 or a future snapshot-fidelity sprint is in flight. Not blocking.

---

## Dependency audit

No dependency changes in S30. No `package.json`, `pyproject.toml`, `environment.yml`, or vendored-library-version bump. `git diff --stat master..HEAD -- "**/package*.json" "**/pyproject.toml" "**/environment.yml" "**/requirements*.txt"` returns empty.

Cached S29 audit against MLX (0.31.1, Homebrew pin), nanobind (vendored via MLX), and Python deps stands — no re-audit required.

---

## Positive findings

1. **Compile-time instrumentation gating is exemplary.** 15 `#ifdef COSINE_RESIDUAL_INSTRUMENT` occurrences all cleanly scoped; `<filesystem>` include itself is gated. Zero release-build footprint. `COSINE_D2_INSTRUMENT` implies the base flag via a forwarding `#ifndef`/`#define` block (no independent duplicate state). Release-build verification is a greppable invariant.
2. **Single-point-of-removal grep anchors.** The `TODO-S29-LG-COSINE-RCA` / `TODO-S29-ST-COSINE-KAHAN` marker strategy makes "is the guard still in place at all three layers?" a one-liner grep. A removal attempt anywhere would trip the CI grep gate from S29 (DEC-032) — verified still effective against the S30 tree.
3. **Runtime test coverage preserved.** `test_cli_guards.py` exercises the production binary (`python/catboost_mlx/bin/csv_train`) — the one built without S30 bypass macros — and asserts both non-zero exit and marker string in stderr. `returncode != 0` rather than `== 1` keeps the test resilient to a future SA-I2 fix. No regressions in the pytest suite.
4. **Defence-in-depth mirrors guards across languages verbatim.** The nanobind guard (`train_api.cpp:28-51`) and CLI guard (`csv_train.cpp:381-410`) are byte-identical copies of the Python text — grep-maintainable.
5. **Bypass macros are self-documenting.** Comments above each gate explicitly read "MUST NOT be used in production builds" and name the scripts that set them. No hidden doors.

---

## Recommendations (priority-ordered)

1. **LOW — fix SA-L1 now.** Replace the per-name `.gitignore` entries with `/csv_train_*` / `/csv_predict_*` globs. Prevents an accidental `git add .` from committing a guard-bypassed binary. One-line change, zero risk.
2. **LOW — close SA-L2 on next doc pass.** Technical writer can swap `/Users/ramos/Library/...` for `$(git rev-parse --show-toplevel)` in the two affected doc/script files.
3. **LOW — harden `COSINE_RESIDUAL_OUTDIR` handling (S31).** Validate the env-var path against a hard-coded `docs/sprint30/**/data/` prefix, or drop support for the override now that the sprint is over.
4. **LOW — carry SA-I2 (#95) as S31 scope.** Close the graceful-exit wrap when `@ml-engineer` is next in the loop on `csv_train.cpp`. Not blocking; already tracked.
5. **INFO — future sprints archiving CPU-CatBoost reference JSONs.** Consider a post-processing pass that strips `model_info.catboost_version_info` from redistributable JSON fixtures. Not blocking S30.

---

## Outstanding items carried to S31

| Ticket | Title | Source | Severity |
|---|---|---|---|
| `#95` | S30-T5-CLI-EXIT-WRAP (SA-I2-S29) | S29-SA t1-sa-report.md | LOW |
| `SA-L1-S30` | `.gitignore` glob for `csv_train_*` binaries | this report | LOW |
| `SA-L2-S30` | Absolute `/Users/ramos/...` paths in repro docs | this report | LOW |
| `SA-L3-S30` | `COSINE_RESIDUAL_OUTDIR` research-only env-var hardening | this report | LOW |

---

## Deployment guidance

**No CRITICAL findings — deployment is not blocked.**

Sprint 30 may merge to `master`. Pre-merge checklist:

- [x] SA-H1 guards intact (Python, nanobind, CLI).
- [x] K4 + Fix 2 do not introduce silent truncation or TOCTOU.
- [x] Instrumentation is compile-time-gated; release builds have zero footprint.
- [x] `test_cli_guards.py` still exercises the production binary at `python/catboost_mlx/bin/csv_train`.
- [x] No dependency changes.
- [x] No secrets or credentials in sprint30/ artefacts.
- [ ] (Optional, recommended) patch `.gitignore` per SA-L1 before the merge to close a paper-cut.

Finding counts: **CRITICAL 0 / HIGH 0 / MEDIUM 0 / LOW 3 / INFO 3 + 1 carried (SA-I2-S29)**.
