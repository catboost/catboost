# Sprint 33 Sprint-Close — Security Audit

**Branch**: `mlx/sprint-33-iter2-scaffold`
**Tip**: `c511549eeb`
**Merge-base**: `17451f4780`
**Commits in scope**: 44 since master (12 PROBE + 9 verdict-state + 5 fix + others)
**Date**: 2026-04-25

## Verdict: PASS-WITH-NOTES

Finding counts: **CRITICAL 0 / HIGH 0 / MEDIUM 0 / LOW 1 / INFO 4**.

No remotely-exploitable vulnerabilities introduced by Sprint 33. The two guard removals (S28-ST-GUARD, S28-LG-GUARD) are behavioral correctness gates, not security boundaries — their removal does not expand attack surface. The per-side mask numerical fix introduces no divide-by-zero risk under valid input ranges (`l2_reg_lambda >= 0`, validated at `core.py:572-573`). All new instrumentation (PROBE-D, PROBE-E, ITER1_AUDIT, COSINE_TERM_AUDIT) is compile-time `#ifdef`-gated and absent from the production build flags in `python/build_binaries.py`. No secrets, credentials, or PII in committed artifacts. No dependency or CMakeLists changes. SA-L1-S30 has been remediated (`/csv_train_*` glob now in `.gitignore:120`). Sprint 33 may merge to master.

## Top-3 Risks / Notes

### SA-N1-S33 (Info, code-consistency)

Per-side mask asymmetry between `FindBestSplit` and `FindBestSplitPerPartition`. The S33-L4-FIX per-side mask landed only in `FindBestSplit` (oblivious / SymmetricTree path). `FindBestSplitPerPartition` (Depthwise + Lossguide path) at `csv_train.cpp:2293` and `:2377` still uses the original joint-skip pattern:

```cpp
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
```

The LG-guard removal validation showed 0.382% drift at iter=50 for LG+Cosine, well within the 2% threshold, so this is not currently a correctness regression. However, the fix that resolves DEC-036 for ST has not been propagated to DW/LG; if a future workload triggers similar partition-state-class divergence in DW or LG, the mask code in `FindBestSplitPerPartition` will still be wrong. Not a security issue. Recommend tracking as a follow-up consistency item.

### SA-L1-S33 (Low, info-disclosure)

Hard-coded `/Users/ramos/...` paths in three new sprint scripts and probe-b output (same class as SA-L2-S30):

- `docs/sprint33/commit2-gates/run_g4a_g4b.py:22`
- `docs/sprint33/commit2-gates/run_g4d.py:32`
- `docs/sprint33/commit2-gates/run_g4e.py:26`
- `docs/sprint33/probe-b-python/data/run_log.txt:1,16` (committed log capture)
- `docs/sprint33/probe-b-python/data/probe_b_results.json:3` (committed output capture)

These leak the macOS local username `ramos` and the absolute iCloud-sync path. Pure disclosure; same surface as SA-L2-S30 (carried open in S30 sprint-close). Recommend a single doc-pass to swap absolute paths for `Path(__file__).resolve().parents[N]` in the runner scripts and consider whether the captured `run_log.txt` / `probe_b_results.json` need redaction. Not blocking.

### SA-N2-S33 (Info, research-build only)

Instrumentation env-var write paths expanded. Sprint 33 carries forward the SA-L3-S30 pattern in two additional macros: `ITER1_AUDIT_OUTDIR` (`csv_train.cpp:5612`) and `COSINE_TERM_AUDIT_OUTDIR` (`csv_train.cpp:5636`), in addition to `COSINE_RESIDUAL_OUTDIR` (existing). All three accept an arbitrary directory from environment, then `std::filesystem::create_directories` + `fopen("w")` without canonicalisation or sandbox-root validation.

Not reachable in release builds — `build_binaries.py` defines none of `ITER1_AUDIT`, `COSINE_TERM_AUDIT`, `COSINE_RESIDUAL_INSTRUMENT`, `PROBE_E_INSTRUMENT`, `PROBE_D_ARM_AT_ITER`, `COSINE_T3_MEASURE`. Same recommendation as SA-L3-S30: validate against a hard-coded prefix or refuse if `EUID != UID`, applied uniformly across all four research-instrumentation env-vars.

## Affirmations

- **Guard removal is byte-clean across all three layers.** `grep -rn "TODO-S29-LG-COSINE-RCA\|TODO-S29-ST-COSINE-KAHAN" python/ catboost/` returns zero hits. Markers remain only in `tests/test_cli_guards.py:235,293` as negative-presence assertions in stderr — defense-in-depth that the guard wasn't accidentally re-introduced.
- **Removed test cases were strictly the four guard tests.** No other security-relevant assertions were collateral damage. Inverted versions still exercise both nanobind and CLI subprocess paths.
- **Per-side mask divide-by-zero is provably absent under documented input ranges.** `weightLeft + l2RegLambda` is reached only when `wL_pos == (weightLeft > 1e-15f)`, so divisor ≥ `1e-15f` even when `l2_reg_lambda == 0`. Cosine `sqrt(cosDen_d)` is safe because `cosDen_d` is seeded to `1e-20` outside any partition skip. The L2 path's parent subtraction is reached only when at least one side has weight `> 1e-15f`. No new TOCTOU.
- **Instrumentation gate is layered.** `PROBE_E_INSTRUMENT` is nested inside `COSINE_RESIDUAL_INSTRUMENT` (`csv_train.cpp:141`). `COSINE_D2_INSTRUMENT` implies `COSINE_RESIDUAL_INSTRUMENT` (`csv_train.cpp:104-108`). `PROBE_D_ARM_AT_ITER` defaults to 0.
- **Production build excludes all instrumentation macros.** Release CLI binary at `python/catboost_mlx/bin/csv_train` has zero instrumentation footprint.
- **Synthetic-only data in committed artifacts.** All `.csv` and `.json` files under `docs/sprint33/{probe-c,probe-d,probe-e,commit2-gates,l*}/data/` derive from `np.random.default_rng(42|43|44)` synthetic anchors. Pattern scan for `password|secret|api_key|credential|private_key|AKIA…|sk-…|ghp_…|xox[baprs]-` returned zero hits.
- **Upstream provenance file isolated.** `docs/sprint33/probe-a-borders/data/cpu_anchor.json` is the sole file containing `akhropov@yandex-team.com` / `/Users/runner/work/...` — these are upstream CatBoost 1.2.10 build metadata, identical to SA-I1-S30. Public, not local.
- **No dependency or build-config changes.** `git diff --stat` against `environment.yml`, `requirements*.txt`, `pyproject.toml`, `package*.json`, `**/CMakeLists.txt`, `python/build_binaries.py` returns empty.
- **SA-L1-S30 remediated.** `.gitignore:120-121` now uses `/csv_train_*` and `/csv_predict_*` globs. Working tree is clean; no untracked guard-bypass binaries.

## Recommendations

1. **Info — propagate per-side mask to `FindBestSplitPerPartition`** (DW/LG path, `csv_train.cpp:2293,2377`). DEC-042 closure measured DW/LG drift empirically and found it acceptable, but the mathematical fix is structurally incomplete. Recommend a follow-up task to apply the same `wL_pos`/`wR_pos` per-side accumulation to both branches for code-symmetry and forward robustness. Not blocking.
2. **Low — close SA-L1-S33 in next doc pass.** Replace absolute `/Users/ramos/...` paths with `Path(__file__).resolve().parents[N]` or `$(git rev-parse --show-toplevel)` in the three `docs/sprint33/commit2-gates/run_g4*.py` runner scripts. Re-capture or trim `probe-b-python/data/run_log.txt` and `probe_b_results.json` if redistribution becomes a concern.
3. **Low — schedule a unified instrumentation env-var hardening (carries SA-L3-S30 + SA-N2-S33).** A small helper that validates research-instrumentation output paths against a hard-coded `docs/sprint*/**/data/` prefix would close `COSINE_RESIDUAL_OUTDIR`, `ITER1_AUDIT_OUTDIR`, `COSINE_TERM_AUDIT_OUTDIR` simultaneously. Research-only surfaces; defer to whichever sprint next touches the instrumentation harness.
4. **Carry forward SA-I2-S29 (`#95` graceful exit wrap).** Still unfixed in `csv_train` `main()`. Not regressed by S33; no new exception sites added.

## Outstanding items carried from prior sprints

| Ticket | Title | Source | Severity |
|---|---|---|---|
| `#95` | S30-T5-CLI-EXIT-WRAP (SA-I2-S29) | S29-SA t1-sa-report.md | LOW |
| `SA-L2-S30` | Absolute `/Users/ramos/...` paths in repro docs (now also S33) | S30 sa-report.md | LOW |
| `SA-L3-S30` | `COSINE_RESIDUAL_OUTDIR` research-only env-var hardening (S33 expands to two more vars) | S30 sa-report.md | LOW |

## Deployment guidance

**No CRITICAL or HIGH findings — deployment is not blocked. Sprint 33 may merge to master.**

Pre-merge invariants verified:

- [x] No production-source references to the removed guard markers.
- [x] Per-side mask math safe under documented input ranges (`l2_reg_lambda >= 0`).
- [x] All instrumentation `#ifdef`-gated; release binary build excludes all macros.
- [x] No secrets, credentials, or new PII in `docs/sprint33/**/data/`.
- [x] No dependency, environment, or build-config changes.
- [x] Test surface preserved: `tests/test_cli_guards.py` still exercises both nanobind and CLI subprocess paths (now as positive acceptance tests, four cases).
