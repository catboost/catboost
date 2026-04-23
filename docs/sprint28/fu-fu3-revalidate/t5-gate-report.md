# S28-FU3-REVALIDATE Gate Report

**Branch**: `mlx/sprint-28-score-function-fidelity`
**Date**: 2026-04-23
**Authored by**: @ml-engineer (S28-FU3-REVALIDATE, task #74)
**Files edited**: `tests/test_python_path_parity.py`

---

## Context

S28-REBLESS (`c07e895f7c`) left a `TODO-S28-FU3-REVALIDATE` marker on the
conditional force-L2 blocks in `_cpu_fit_nonoblivious` (DW and LG). The
condition for removal was: 5/5 DW cells pass with `score_function='Cosine'`
on BOTH sides as structural proof of gap closure, run through the Python-path
CI harness rather than just the sprint gate harness.

This task provides that proof for DW and records the LG outcome honestly.

---

## PROPOSE / CRITIQUE (pre-implementation)

### Test cells that hit `_cpu_fit_nonoblivious`

| Function | grow_policy | parametrized over | cell count |
|---|---|---|---|
| `test_nonoblivious_python_path_parity` | DW + LG | seed × {1337,42}, rs × {0.0,1.0} | 8 |
| `test_nonoblivious_pred_std_ratio` | DW + LG | seed × {1337,42}, rs × {0.0,1.0} | 8 |

Total: **16 cells** call `_cpu_fit_nonoblivious`. `test_nonoblivious_monotone_convergence`
calls only `_mlx_fit_nonoblivious` (no CPU model).

### Expected outcome

S28-L2-EXPLICIT G3b showed DW Cosine both-sides ratios [0.9950, 1.0160] at N=1000
(seeds 42–46). That was sprint-gate harness only. DW removal was conditioned on
reproducing that evidence through the CI harness — same seeds, same N.

LG had no prior Cosine-both-sides evidence. This is a real open question.

### Decision tree applied

See decision tree in task spec. LG result determines commit scope.

---

## Gate Harness Results

**Harness**: `docs/sprint28/fu-fu3-revalidate/t5-gate-harness.py`
**Raw data**: `docs/sprint28/fu-fu3-revalidate/t5-gate-results.json`
**N=1000**, depth=6, iters=50, rs=0, seeds={42,43,44,45,46}

---

## Gate G5a: DW-NATIVE-COSINE

**Mechanism**: 5 DW N=1000 cells, `score_function='Cosine'` on BOTH sides,
no forced config. Ratio must be in [0.98, 1.02].

| seed | MLX_RMSE | CPU_Cosine | ratio | G5a |
|------|----------|------------|-------|-----|
| 42 | 0.214025 | 0.210677 | 1.0159 | PASS |
| 43 | 0.212311 | 0.208968 | 1.0160 | PASS |
| 44 | 0.210635 | 0.210156 | 1.0023 | PASS |
| 45 | 0.214788 | 0.213174 | 1.0076 | PASS |
| 46 | 0.218467 | 0.219571 | 0.9950 | PASS |

**Max deviation**: 1.60% (seeds 42–43). All ∈ [0.98, 1.02].

**End-to-end vs kernel-level snapshot match**: DW ratios are identical to the
S28-L2-EXPLICIT G3b snapshot (same data generator, same N, same seeds). No
expansion. The Python-path end-to-end numbers faithfully reflect the kernel-level
measurement.

**G5a verdict: PASS (5/5)**

---

## Gate G5b: NO-FORCE-DW

**Mechanism**: grep of `tests/test_python_path_parity.py` for DW-context
force-L2 code returns zero hits.

Post-edit: the only `score_function = "L2"` assignment in `_cpu_fit_nonoblivious`
is in the `if grow_policy == "Lossguide":` branch. The `if grow_policy == "Depthwise":`
branch assigns `score_function = "Cosine"`. No force-L2 remains for DW in either
`_cpu_fit_nonoblivious` or `_mlx_fit_nonoblivious`.

**G5b verdict: PASS**

---

## Gate G5c: LG-OUTCOME

**Mechanism**: 5 LG N=1000 cells, `score_function='Cosine'` on BOTH sides.
Outcome recorded (PASS or FAIL-DOCUMENTED).

| seed | MLX_RMSE | CPU_Cosine | ratio | G5c |
|------|----------|------------|-------|-----|
| 42 | 0.212485 | 0.185999 | 1.1424 | FAIL |
| 43 | 0.213166 | 0.186804 | 1.1411 | FAIL |
| 44 | 0.215828 | 0.188677 | 1.1439 | FAIL |
| 45 | 0.213382 | 0.187135 | 1.1403 | FAIL |
| 46 | 0.215872 | 0.187741 | 1.1498 | FAIL |

**Ratio range**: [1.1403, 1.1498] — MLX LG Cosine is ~14% worse than CPU LG
Cosine across all 5 seeds. The gap is consistent and reproducible, not noise.
Direction and magnitude are analogous to the pre-S28 DW gap (0.82–0.87 on the
CPU/MLX ratio, or 14–17% divergence).

**Interpretation**: MLX `FindBestSplitPerPartition` in the Lossguide path
also hardcodes L2 Newton gain. When CPU is given `score_function='Cosine'`, the
two sides are computing different gain functions — exactly the DEC-032 pattern.
The S28 Cosine port wired DW dispatch but did not reach the LG code path, or
the LG code path has a separate dispatch gap.

**G5c verdict: FAIL-DOCUMENTED** (0/5 seeds in [0.98, 1.02])

**LG force-L2 retained**: `_cpu_fit_nonoblivious` LG branch keeps
`kwargs["score_function"] = "L2"`. `_mlx_fit_nonoblivious` LG branch passes
`score_function="L2"`. The harness tests L2-vs-L2 parity for LG, which passes
(28/28 full suite).

**Proposed followup**: **S29-FU-LG** — port Cosine dispatch to the MLX
Lossguide code path in `FindBestSplitPerPartition`. Scope: audit whether the
LG branch shares the same `switch (scoreFunction)` added in S28-L2-EXPLICIT
for DW, or has a separate hardcoded gain call that was missed. Evidence to
collect: per-partition LG gain instrumentation (same as DEC-032 FU-3 T1 did
for DW). Gate: 5/5 LG seeds in [0.98, 1.02] with Cosine both sides.

---

## Gate G5d: FULL-SUITE-REGRESSION

**Mechanism**: pytest on full `tests/test_python_path_parity.py` after edits.
Same pass/fail set as pre-commit; no unrelated regressions.

```
28 passed in 55.58s
```

| Pre-commit | Post-commit | Delta |
|---|---|---|
| 28 passed | 28 passed | 0 |

DW cells now test Cosine-both-sides parity (structural change). LG cells still
test L2-both-sides parity (unchanged algorithm, updated structural form —
explicit Cosine→no, explicit L2→yes on both sides). All convergence and std-ratio
cells pass.

**G5d verdict: PASS (28/28)**

---

## REFLECT

### DW end-to-end vs kernel-level

DW ratios in G5a are identical to S28-L2-EXPLICIT G3b (same numbers, bit-for-bit).
No expansion between kernel-level sprint gate and Python-path CI harness. The
structural proof is clean.

### LG finding

LG has a ~14% Cosine/L2 gap at N=1000, 0/5 seeds passing. The ratio range
[1.1403, 1.1498] is narrower and higher than the DW pre-S28 range [0.83–0.87]
— meaning MLX LG Cosine is consistently ~14% worse than CPU LG Cosine (not
random disagreement, one direction). This is the same class of finding DEC-032
documented for DW: MLX LG implements L2; CPU LG defaults to Cosine.

### Changes beyond `_cpu_fit_nonoblivious`

The structural change required three helper function edits to be internally
consistent per DEC-031 Rule 3:
- `_cpu_fit_nonoblivious`: DW branch → `score_function='Cosine'`; LG branch keeps `score_function='L2'`
- `_mlx_fit_nonoblivious`: policy-aware `score_function = "Cosine" if DW else "L2"`
- `test_nonoblivious_monotone_convergence`: same policy-aware dispatch; path-label comments updated

### DEC-032 closure recommendation

**CLOSED-FOR-DW**: The DW gap is structurally closed. MLX DW dispatches Cosine,
the CPU-side force-L2 is removed, and end-to-end parity is confirmed at 5/5 seeds.

The DEC-032 scope statement is "Covers `FindBestSplitPerPartition` (DW and LG
code paths)." The LG path remains open. A correct disposition for DECISIONS.md is:

> **Status**: PARTIALLY CLOSED (DW). LG open — see S29-FU-LG.

The S28-CLOSE task (#77) should update DEC-032 status accordingly, referencing
this report for evidence.

---

## Files Modified

- `tests/test_python_path_parity.py` — DW force-L2 removed; LG force-L2 retained
  with updated comment; `_mlx_fit_nonoblivious` made policy-aware; `test_nonoblivious_monotone_convergence` made policy-aware; path-label comments updated throughout

## Files Created

- `docs/sprint28/fu-fu3-revalidate/t5-gate-harness.py` (Cosine-both-sides evidence harness)
- `docs/sprint28/fu-fu3-revalidate/t5-gate-results.json` (raw per-seed data)
- `docs/sprint28/fu-fu3-revalidate/t5-gate-report.md` (this file)
