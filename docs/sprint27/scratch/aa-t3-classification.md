# S27-AA-T3 — Anchor Classification

**Produced:** 2026-04-22 (Sprint 27, Track B)
**Agent:** @qa-engineer
**Branch:** `mlx/sprint-27-correctness-closeout`
**Inputs:** T1 inventory (`d4e2d7cf88`), T2 re-run results (`800fdc8fce`)
**Purpose:** Classify all 18 anchors for T4 action. Pre-flight escalation check included.

---

## Pre-flight Escalation Check

**CLASS-B COUNT: 0**

No anchor exhibits a current-code regression where the anchor was correct and current code is wrong. All drifted anchors are explained by deliberate, committed code changes. Proceed to full classification.

---

## Classification Table

| ID | Path | Class | Drift (T2) | Reasoning | Sprint/DEC explaining drift | T4 action | Urgency |
|----|------|-------|-----------|-----------|----------------------------|-----------|---------|
| AN-001 | `python/tests/test_qa_round9_sprint4_partition_layout.py:91` | **a** | 3.3e-07 (rel) | Updated to DEC-028-aware value at `634a72134d` (S26-D0-9); sub-ULP gap is truncated string precision only | S26-D0-9, DEC-028 (`634a72134d`) | Done — no T4 action (already current) | — |
| AN-002 | `python/tests/test_qa_round9_sprint4_partition_layout.py:107` | **a** | 5.7e-07 (rel) | Same DEC-028 update batch as AN-001; residual is string truncation | S26-D0-9, DEC-028 (`634a72134d`) | Done — no T4 action | — |
| AN-003 | `python/tests/test_qa_round9_sprint4_partition_layout.py:108` | **a** | 8.8e-08 (rel) | Same DEC-028 update batch | S26-D0-9, DEC-028 (`634a72134d`) | Done — no T4 action | — |
| AN-004 | `python/tests/test_qa_round9_sprint4_partition_layout.py:109` | **a** | 1.4e-08 (rel) | Same DEC-028 update batch | S26-D0-9, DEC-028 (`634a72134d`) | Done — no T4 action | — |
| AN-005 | `python/tests/test_qa_round9_sprint4_partition_layout.py:155` | **a** | 0 (bit-exact) | Same DEC-028 update batch; multiclass probas bit-exact on current tip | S26-D0-9, DEC-028 (`634a72134d`) | Done — no T4 action | — |
| AN-006 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py:66` | **a** | 6.82e-02 (rel) | Captured 2026-04-11 post-BUG-002 (`f804c56742`); four subsequent kernel changes each shifted the value: BUG-S18-001 L1a fix (`19fa5ce6cc`, Apr 17), DEC-015 side-fix + DEC-016 T1 fuse-valid (`77db8b5631`, `92f3832169`, Apr 19), S22/S23 T2 promotion (`4d1eda1f4c`..`eaf05bc21d`), DEC-023 v5 all-feature T1-accum (`36e01438a1`, Apr 21) — all legitimate shipping changes | S18 BUG-S18-001 (`19fa5ce6cc`), S19 DEC-015/016 (`77db8b5631`, `92f3832169`), S22/S23 T2 promotion, S24 DEC-023 (`36e01438a1`) | Update to `0.11097676` | P0 (live test fails) |
| AN-007 | `python/tests/test_qa_round10_sprint5_bench_and_scan.py:67` | **a** | 2.71e-02 (rel) | Same capture date and same kernel-generation chain as AN-006; K=3 multiclass path traverses same histogram kernel | S18 BUG-S18-001 (`19fa5ce6cc`), S19 DEC-015/016, S22/S23 T2 promotion, S24 DEC-023 (`36e01438a1`) | Update to `0.61788315` | P0 (live test fails) |
| AN-008 | `CHANGELOG.md:41` | **a** | 4.03e-02 (rel) | Captured at `fd04d34684` (S18-10 docs, Apr 17); same S19/S22/S23/S24 kernel changes explain the 4% drift; K=10 multiclass exercises same histogram accumulation path. **Repeat-offender flag:** this anchor reset once before at Sprint 8 TODO-022 (`2.22267818` → `1.78561831`); T5/DEC-031 should add a standing clause requiring this to be a live-asserted test, not a CHANGELOG comment | S19 DEC-015/016, S22/S23 T2 promotion, S24 DEC-023 (`36e01438a1`) | Update to `1.85752499`; note in T5 that this anchor is a repeat-offender for DEC-031 standing clause | P2 (docs-only) |
| AN-009 | `docs/sprint19/results.md:41` | **a** | 0 (bit-exact) | Config #8 (10k/RMSE/128b), DEC-023 Value A; still exact on current tip — DEC-023 v5 fix stabilised this | S24 DEC-023 v5 (`36e01438a1`) fixed and stabilised | Done — no T4 action | — |
| AN-010 | `docs/sprint19/results.md:47` | **a** | 0 (bit-exact) | Primary 50k determinism gate; DEC-023 v5 + all post-S19 changes preserve this value exactly | S24 DEC-023 v5 (`36e01438a1`) | Done — no T4 action | — |
| AN-011 | `docs/sprint19/results.md:34–49` | **a** | 0 (all 18 bit-exact) | Full 18-config sweep; all values match S19 reference exactly on current tip; T2 re-ran all 18 | S24 DEC-023 v5 (`36e01438a1`) | Done — no T4 action | — |
| AN-012 | `docs/sprint18/parity_results.md:31–48` | **c** | ~2e-02 (max rel) | S18 parity sweep was captured at fixed kernel `19fa5ce6cc`; superseded by S19 kernel (DEC-016 T1 fuse-valid + EvalAtBoundary removal) which changed accumulation order and numeric topology; S19 values (AN-011) are the authoritative post-supersession reference | S18 kernel superseded by S19 DEC-016 (`92f3832169`) and S24 DEC-023 (`36e01438a1`) | Update doc to note superseded by S19 values (AN-011); add pointer to `b04b1efd2c` | P2 (docs-only) |
| AN-013 | `docs/sprint17/parity_results.md:22–39` | **d** | — (UNRUNNABLE) | `csv_train_sprint16`/`csv_train_sprint17` binaries no longer exist; path used CSV+subprocess pipeline predating bench_boosting; no live assertion anywhere references these values; superseded binary infrastructure is gone | — | Remove from doc or mark `(DEAD — see DEC-031)` per Ramos policy in T5 | P3 (dead) |
| AN-014 | `benchmarks/results/m3_max_128gb.md:13–30` | **d** | — (NOT-ASSERTED, budget) | catboost_mlx 0.3.0 (Sprint 14, `7b36f60a82`), 12+ sprints stale; predates DEC-028, DEC-029, DEC-023 v5; no test enforces these values; `benchmarks/bench_mlx_vs_cpu.py` exists but is unrunnable at any reasonable CI budget | — | Remove from doc or mark `(DEAD — see DEC-031)` per Ramos policy in T5 | P3 (dead) |
| AN-015 | `python/tests/test_qa_round12_sprint9.py:655,661` | **d** | — (UNRUNNABLE) | Fixture references `.github/workflows/mlx_test.yaml` (underscore); actual file is `mlx-test.yaml` (hyphen); all 6 `TestCIBenchWorkflow` tests skip permanently; the CI-embed intent never landed; values have had zero enforcement for 15+ sprints | — | Wire to real assertion or remove from test; policy decision in T5/DEC-031 | P3 (dead) |
| AN-016 | `docs/sprint26/d0/g1-g3-g4-report.md:101` | **a** | 5.1e-08 (rel) | S26 G5 determinism anchor (SymmetricTree, N=10k, seed=1337, rs=0); 1e-8 gap is single-run vs 100-run mean float32 noise, not code drift; no kernel changes since `cbbfc29257` affect this path | S26 G5 result (`cbbfc29257`), no intervening change | Update to `0.19457838` (single-run measured value) or note 1e-8 sub-ULP; P2 since docs-only | P2 (docs-only) |
| AN-017 | `benchmarks/sprint26/fu2/fu2-gate-report.md:101` | **deferred-a** | 5.8e-08 (rel, pre-FU-1) | FU-1-DEPENDENT: uses Depthwise grow policy at depth>=2/3 which exercises the buggy `ComputeLeafIndicesDepthwise` (BUG-A + BUG-B per `34f62b32c9`); current value `0.17222002` is pre-fix; post-FU-1 fix will change this value legitimately; the 1e-8 pre-fix gap is noise-only | S27 FU-1 in-flight (`34f62b32c9`, DEC-030) | DEFER — re-capture after FU-1-T3 merges to master | P2 (docs-only, pre-fix) |
| AN-018 | `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md:31` | **c** | 6.39e-03 (rel) | Captured at S19 intermediate tip `0f992cf863` under A1 BATCH_DOCS=64 variant; A1 produced +9.4% production regression and was explicitly dropped/reverted in the same sprint close (`0f992cf863`); current tip produces T1-only value `0.47740927` (=AN-010); the doc correctly records the A1 intermediate; its own text explains the drop | S19 A1 port dropped at `0f992cf863` (same SHA as capture — this is a scratch file that was committed at the moment of decision) | Update value to `0.47740927` (current T1-only / AN-010 canonical) or add note that post-revert the parity value is AN-010 | P2 (docs-only scratch) |

---

## Summary Counts

| Class | Count | IDs |
|-------|-------|-----|
| **a** (stale-capture, new value correct) | 12 | AN-001..011, AN-016 |
| **a — done, no T4 action** | 8 | AN-001..005, AN-009..011 |
| **a — T4 update required** | 4 | AN-006, AN-007, AN-008, AN-016 |
| **b** (regression — escalate) | **0** | — |
| **c** (documented-supersession) | 2 | AN-012, AN-018 |
| **d** (dead anchor) | 3 | AN-013, AN-014, AN-015 |
| **deferred-a** (FU-1-dependent) | 1 | AN-017 |
| **Total** | 18 | |

---

## Urgency Distribution

| Priority | Count | IDs |
|----------|-------|-----|
| P0 (live test fails, blocking CI) | 2 | AN-006, AN-007 |
| P1 (silent drift in asserted test) | 0 | — |
| P2 (docs-only drift) | 5 | AN-008, AN-012, AN-016, AN-017 (deferred), AN-018 |
| P3 (dead anchor, no enforcement) | 3 | AN-013, AN-014, AN-015 |
| Done (already current, no T4 action) | 8 | AN-001..005, AN-009..011 |

---

## P0 List (CI-blocking — T4 must land before any CI run)

1. **AN-006** — `python/tests/test_qa_round10_sprint5_bench_and_scan.py:66` — update `BINARY_ANCHOR_LOSS` from `0.11909308` to `0.11097676`
2. **AN-007** — `python/tests/test_qa_round10_sprint5_bench_and_scan.py:67` — update `MULTICLASS_ANCHOR_LOSS` from `0.63507235` to `0.61788315`

These two live tests fail on current tip when bench_boosting is built from source. T4 must land the P0 updates as the first commit.

---

## T2 Hypothesis Divergences

T2's classifications are confirmed for all anchors. No disagreements with T2 hypotheses, with two clarifications:

1. **AN-018**: T2 described this as "kernel-revert artifact." More precisely, this is class **c** (documented-supersession) rather than class **a** — the anchor file itself documents the A1 intermediate intentionally; the correct post-supersession value is AN-010 (`0.47740927`). The distinction matters for T4 action (update with pointer to the superseding sprint-close, not just overwrite).

2. **AN-008 repeat-offender flag**: T2 correctly called this class-a but did not flag the Sprint 8 TODO-022 precedent (prior reset `2.22267818` → `1.78561831`, now drifted again to `1.85752499`). T3 adds this to the T5/DEC-031 standing-order recommendation.

---

## T4 Ordered Work Queue

Execute in this order:

| Order | ID | Action | File |
|-------|----|--------|------|
| 1 (P0) | AN-006 | Update `BINARY_ANCHOR_LOSS = 0.11097676` | `python/tests/test_qa_round10_sprint5_bench_and_scan.py:63` |
| 2 (P0) | AN-007 | Update `MULTICLASS_ANCHOR_LOSS = 0.61788315` | `python/tests/test_qa_round10_sprint5_bench_and_scan.py:64` |
| 3 (P2) | AN-008 | Update CHANGELOG K=10 value to `1.85752499`; add repeat-offender note | `CHANGELOG.md:41` |
| 4 (P2) | AN-012 | Add superseded-by-S19 note pointing to AN-011 / `b04b1efd2c` | `docs/sprint18/parity_results.md` |
| 5 (P2) | AN-016 | Add note: 1e-8 sub-ULP from single-run proxy; canonical 100-run mean remains `0.19457837` | `docs/sprint26/d0/g1-g3-g4-report.md:101` |
| 6 (P2) | AN-018 | Add note: post-A1-drop parity value is AN-010 (`0.47740927`); this doc records the pre-drop intermediate intentionally | `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md:31` |
| 7 (P3) | AN-013 | Mark `(DEAD — see DEC-031)` or remove — Ramos picks policy in T5 | `docs/sprint17/parity_results.md` |
| 8 (P3) | AN-014 | Mark `(DEAD — see DEC-031)` or remove — Ramos picks policy in T5 | `benchmarks/results/m3_max_128gb.md` |
| 9 (P3) | AN-015 | Wire to real assertion or remove — Ramos picks policy in T5 | `python/tests/test_qa_round12_sprint9.py` |
| DEFER | AN-017 | Re-capture after FU-1-T3 merges to master | `benchmarks/sprint26/fu2/fu2-gate-report.md:101` |

---

## Notes for T5 / DEC-031

- **AN-008 repeat-offender**: This is the third numeric lifetime for this CHANGELOG comment (S8 TODO-022 → `1.78561831`, S18+ kernel changes → `1.85752499`). DEC-031 should mandate promoting this to a live-asserted test in `test_qa_round10_sprint5_bench_and_scan.py` (same file as AN-006/007), eliminating the CHANGELOG-comment-only enforcement pattern.
- **AN-015 fixture**: The `mlx_test.yaml` vs `mlx-test.yaml` naming error has silently disabled 6 CI workflow assertions for 15+ sprints. DEC-031 policy options: (a) fix the path and restore the CI-embedding intent, (b) port assertions to a standalone pytest instead of parsing a YAML, (c) remove entirely if the CI-embedding design is abandoned.
- **AN-013/014**: Both dead-path anchors. DEC-031 policy: mark as `(DEAD)` with pointer, or remove. These are docs-only and have no enforcement path.
