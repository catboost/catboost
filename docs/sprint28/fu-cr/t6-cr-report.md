# S28-CR — Sprint 28 Code Review Report

**Date:** 2026-04-23
**Reviewer:** @code-reviewer
**Commits reviewed:** `83f30c3677`, `0ea86bde21`, `4083add248`, `c07e895f7c`, `dca62f0d72`, `b9577067ef`
**Parity suite state:** 28/28 at `b9577067ef` (not re-run)

**Note:** report persisted by orchestrator; reviewer returned findings inline due to Write-tool restriction.

---

## Summary

**Verdict: PASS-WITH-NITS.** Counts: Must-fix 0 / Should-fix 1 / Nits 3.

All seven gates pass. Cosine gain formula matches CPU `TCosineScoreCalcer` semantics exactly. Enum dispatch is complete across all four call sites (FindBestSplit × {one-hot, ordinal}, FindBestSplitPerPartition × {one-hot, ordinal}). Combination guards mirror the NewtonL2/NewtonCosine rejection pattern. One non-blocking should-fix: a scalar `ComputeCosineGain` helper is present but never called.

---

## Gate-by-gate findings

### G-CR1 Formula correctness — **PASS**

`ComputeCosineGainKDim(num, den) = num / sqrt(den)` with per-(p,k) accumulation:

- `num += G² · invL + G² · invR`
- `den += G² · W · invL² + G² · W · invR²`

Exactly matches CPU `TCosineScoreCalcer::AddLeafPlain` expanded (cross-checked `score_calcers.h:63-66` + `short_vector_ops.h:155-175`). FP32 guards (`weight < 1e-15f` skip; `cosDen` init `1e-20f`) justified inline.

### G-CR2 Enum dispatch plumbing — **PASS**

Four switch sites: `FindBestSplit` / `FindBestSplitPerPartition` × `{one-hot, ordinal}`. Each has `default: throw std::logic_error`. `ParseScoreFunction` rejects unknown strings + `NewtonL2`/`NewtonCosine` explicitly. All three call sites use `ParseScoreFunction(config.ScoreFunction)`.

### G-CR3 DEC-012 atomicity — **PASS with nit**

`b9577067ef` bundles LG-GUARD + ST-GUARD. Symmetric, same function, same validation pattern — defensible; strict DEC-012 would split. Documented as nit CR-N1.

### G-CR4 Guard consistency — **PASS**

Mirrors NewtonL2/NewtonCosine rejection style. Errors name the mechanism + quantitative evidence + S29 marker + concrete workaround. Error messages are actionable.

### G-CR5 Test coverage adequacy — **PASS**

Every parity cell has explicit `score_function`. LG force-L2 correctly scoped (only LG retains the override; DW force-L2 was correctly removed per G5a 5/5).

### G-CR6 Project conventions — **PASS**

`EScoreFunction` matches CPU enum exactly; no `Co-Authored-By` trailers; PascalCase/ui32 consistent with codebase. Comments only where non-obvious.

### G-CR7 Dead code / debt — **SHOULD-FIX (CR-S1)**

- **CR-S1:** `ComputeCosineGain` (non-KDim scalar) at `csv_train.cpp:1026-1046` is dead code. No dispatch path calls it; only `ComputeCosineGainKDim` is used. Either remove or annotate as derivation-only.
- `TODO-S29-LG-COSINE-RCA` and `TODO-S29-ST-COSINE-KAHAN` markers in `core.py` have no backing entry in `.claude/state/TODOS.md` yet (Nit CR-N2).

---

## Must-fix

None.

## Should-fix

### [CR-S1] Remove or annotate unused scalar `ComputeCosineGain` helper

`catboost/mlx/tests/csv_train.cpp:1026-1046`. No call site. Future-reader ambiguity about which Cosine helper is authoritative. Remove, or annotate as `// [derivation-only; see ComputeCosineGainKDim for production path]`.

## Nits

- **CR-N1:** `b9577067ef` bundles two guards in one commit. Symmetric enough to be defensible; split would have been cleaner DEC-012.
- **CR-N2:** `TODO-S29-*` markers in `core.py` need backing entries in `.claude/state/TODOS.md` or Sprint 29 skeleton.
- **CR-N3:** CPU leafwise scoring accumulates one Cosine score per split across ALL leaves via one `scoreCalcer` (`leafwise_scoring.cpp:477` + `calcStatsScores`); MLX computes per-(split × partition). Pre-existing DW/LG design, NOT introduced by S28. Out of DEC-032 scope — flagged for future fidelity consideration only.

## S29 follow-ups confirmed

Aligned with S28-SA: SA-H1 / S29-CLI-GUARD, TODO-S29-LG-COSINE-RCA, TODO-S29-ST-COSINE-KAHAN.
