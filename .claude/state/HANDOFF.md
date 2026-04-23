# Handoff — CatBoost-MLX

> Last updated: 2026-04-23 (Sprint 28 ACTIVE — Score Function Fidelity; branch cut from 4b3711f82b)

## Current state

- **Active sprint**: Sprint 28 — Score function fidelity. Branch `mlx/sprint-28-score-function-fidelity` cut from master at `4b3711f82b` on 2026-04-23.
- **Campaign**: Post-Verstappen correctness. S27 closed zero-kernel-change. S28 implements Cosine score function (DEC-032). R8 stays at 1.01× through S28.
- **Production kernel**: v5 (`784f82a891`) — unchanged.
- **Open PRs**: S27 PR #25 MERGED (`4b3711f82b`). PR #24 (S26-FU-2) merged earlier.
- **Known bugs**:
    - BUG-T2-001 RESOLVED (`784f82a891`).
    - BUG-007 MITIGATED 2026-04-22 (`71aabaa842`) — two-layer defense (Python wrapper sorts; C++ throws on unsorted).
    - K=10 anchor mismatch RESOLVED Sprint 8 (TODO-022, `CHANGELOG.md:27`).
    - Sibling S-1 (`kHistOneByte` writeback race) latent; guarded by compile-time `static_assert` at `histogram.cpp:126`.
    - **S27-FU-1 RESOLVED** (`fb7eb59b5f`): `ComputeLeafIndicesDepthwise` encoding + split-lookup bugs fixed. DEC-030. DEC-029 Risks entry retired.
    - **S27-FU-3 SCOPED to S28** (DEC-032): DW N=1000 asymmetry is a score-function fidelity gap (MLX hardcodes L2; CPU defaults Cosine). Conditional gate G3-FU3 PASS 5/5 with `score_function='L2'` on CPU. Cosine port is S28 scope.

## Sprint 28 — Score Function Fidelity (ACTIVE)

Branch: mlx/sprint-28-score-function-fidelity (cut from 4b3711f82b, 2026-04-23)
Previous sprint: S27 closed 2026-04-23 via PR #25 (4b3711f82b)
Scope (small-sprint shape per @ml-product-owner 2026-04-23):
  1. S28-AUDIT — formality confirmation (grep verified zero score_function refs in catboost/mlx/)
  2. S28-COSINE — port Cosine gain from CPU reference into FindBestSplitPerPartition
  3. S28-L2-EXPLICIT — enum-dispatch {L2, Cosine}; remove hardcode
  4. S28-REBLESS — explicit score_function labels on all Python-path parity test cells
  5. S28-FU3-REVALIDATE — 5 DW N=1000 cells pass with Cosine both sides (force-L2 branch removed)
  6. CR + SA + CLOSE

Ride-alongs deferred (per Ramos decision 2026-04-23): AN-008 Rule-5 promotion, CR Nit 2, SA Note 2, AA Item H, NewtonL2/NewtonCosine variants.

Next agent: @ml-engineer (S28-AUDIT writeup, then S28-COSINE implementation with @research-scientist for Cosine formula derivation if needed).

## Sprint 27 — Correctness Closeout — CLOSED

**Branch**: `mlx/sprint-27-correctness-closeout` (branches off master after PR #24 S26-FU-2 merges)
**Scope**: Close three known correctness debts on the non-oblivious / anchor surface before any R8 perf work resumes. Zero kernel changes. Zero perf work. R8 stays at 1.01×.

### Track A — S27-FU-1: Depthwise validation-path index fix (sequential, blocks FU-3)

- **T1** @qa-engineer — Repro harness: instrument `ComputeLeafIndicesDepthwise` at `csv_train.cpp:1751`; capture returned indices vs expected bit-packed BFS on DW N=10k d=3.
- **T2** @ml-engineer — CPU-source audit: confirm CatBoost BFS encoding (`nodeIdx − numNodes` is wrong decode — should traverse from node 0); draft DEC-030.
- **T3** @ml-engineer — Implement fix: replace arithmetic decode with root-to-leaf traversal mirroring `ComputeLeafIndicesLossguide`.
- **T4** @qa-engineer — Gate G1-FU1: DW validation RMSE (`use_best_model=True`) matches within rs=0 tight band across 3 seeds × {N=10k, N=50k}.
- **T5** @technical-writer — DEC-030 authored; DEC-029 Risks entry retired.

### Track B — S27-AA: Anchor audit (parallel to Track A)

- **T1** @qa-engineer — Enumerate all numeric anchors in committed test/bench files. Produce inventory `{path, line, value, last-touched-sha, captured-context}`.
- **T2** @qa-engineer — Re-run each anchor's generating harness on current master. Diff ≥1e-2 flags drift.
- **T3** @qa-engineer — For each drifted anchor: classify (a) stale-capture / (b) real-regression / (c) documented-supersession.
- **T4** @ml-engineer — Landing commits — ONE commit per anchor update, message cites class. Class-(b) escalates to Ramos before landing.
- **T5** @technical-writer — DEC-031 authored: "Anchor hygiene protocol."

### Track C — S27-FU-3: DW N=1000 parity-asymmetry triage (blocks on FU-1 landing)

- **T1** @qa-engineer — Instrument `FindBestSplitPerPartition` on DW N=1000 depth-0: per-partition `(gain_MLX, gain_CPU, chosen_split, gradRms)` across 3 seeds.
- **T2** @qa-engineer — Control: run ST at N=1000 with matched config (DW-specific or shared small-N issue).
- **T3** @qa-engineer + @ml-product-owner — Verdict doc: (a) BUG — open fix; (b) NOISE — tighten gate scope to N≥10k; (c) ACCEPTED — widen pred_std_R band with rationale. Ramos decides (b)/(c).
- **T4** @ml-engineer — Gate adjustment (b/c) OR fix landing (a) — separate commit.
- **T5** @technical-writer — DEC-032 authored: verdict + rationale.

### Track D — Quality gates (end-of-sprint, sequential)

- **S27-CR** @code-reviewer — Full code review (all FU-1 + AA + FU-3 diffs).
- **S27-SA** @security-auditor — Security audit (low-risk — no kernel changes, validation-path data flow only).
- **S27-CLOSE** @technical-writer — Sprint close doc at `docs/sprint27/sprint-close.md` with gate summary, drift inventory, decision links.

### Ordering & dependencies

Track A ∥ Track B → Track C (blocks on A) → Track D (sequential).

### Exit gates

| Gate | Criterion | Path coverage |
|------|-----------|---------------|
| G0 | Root cause docs complete (DEC-030, DEC-031, DEC-032) | Docs |
| G1-FU1 | DW validation RMSE rs=0 ratio ∈ [0.98, 1.02], 3 seeds × {N=10k, 50k} | C++ validation path (`use_best_model=True` during DW training) |
| G2-AA | All committed anchors match harness within 1e-2 OR have class-(a/c) rationale commit. Zero class-(b) uninvestigated. | Anchor-generating harness per row |
| G3-FU3 | DW N=1000 ratios ∈ [0.98, 1.02] with `score_function='L2'` on both sides (CPU explicit, MLX hardcoded). Path: `FindBestSplitPerPartition` gain-scope equivalence **conditional on L2 matching**. Unconditional algorithm parity (Cosine port) is S28 scope. Evidence: `docs/sprint27/fu3/t4-gate-report.md` — 5/5 PASS, ratios [0.9956, 1.0011]. | DW FindBestSplitPerPartition at small N |
| G4 | bench_boosting v5 ULP=0 preserved (18-cell sweep, kernels unchanged) | Kernel-output only |
| G5 | `tests/test_python_path_parity.py` still 8/8 PASS | Python/nanobind orchestration |
| G6 | R8 bench_boosting e2e ratio drift ≤ 2% vs master baseline | End-to-end bench harness |
| G7 | Determinism: gate config 100 runs max−min ≤ 1e-6 | End-to-end training determinism |

### Kill-switches

1. FU-1 fix produces train-time parity drift (v5 ULP=0 breaks OR Python-path pytest breaks) → abort Track A, open triage, keep DEC-029 Risks entry.
2. AA discovers class-(b) regression → halt AA landing, escalate to Ramos.
3. FU-3 triage reveals DW N=1000 asymmetry caused by FU-1 → revert FU-1, re-plan.
4. Any commit slips DEC-012 atomicity → revert, resplit.
5. Wall-clock > 4 working sessions without Track A or B landing → escalate scope.

### Timeline

3–4 working sessions. If all tracks land clean: 3. If FU-3 is class-(a) real bug: 4.

## Sprint 26 FU-2 — RandomStrength in FindBestSplitPerPartition — CLOSED

**Branch**: `mlx/sprint-26-fu2-noise-dwlg` (stacked on `66a4b5e869`)
**Date closed**: 2026-04-22
**Verdict**: ALL GATE PASS. 0 kill-switches fired. APPROVE-WITH-NITS from @code-reviewer.

DEC-028's `gradRms`-based noise formula extended from `FindBestSplit` (SymmetricTree) to
`FindBestSplitPerPartition` (Depthwise and Lossguide). CPU source audit confirmed identical
global-scalar mechanism for all three grow policies. Implementation: 47 lines in `csv_train.cpp`.

**Gate summary**: G1-DW 12/12 PASS (N≥10k), G1-LG 18/18 PASS, G2 ST non-regression 18/18 PASS,
G5 determinism max−min 1.49e-08 (threshold 1e-6). Five DW N=1000 failures are pre-existing
(verified on pre-FU-2 binary) — tracked as S26-FU-3.

**Nit-1 fixed**: gate report now has path-coverage labels. Nits 2/3/4 recorded as tech-debt.
**DEC-028**: footnote added (no new DEC-030 — pure extension, no new design content).

**Files of record**: `docs/sprint26/fu2/sprint-close.md`, `benchmarks/sprint26/fu2/fu2-gate-report.md`.

---

## Sprint 26 — Python-Path Parity — D0 CLOSED

### Verdict: D0 PASS on all exit gates. DEC-028 + DEC-029 RESOLVED the Python-path leaf-magnitude collapse. R8 unchanged at 1.01× (S26 is correctness-first). v5 production kernel untouched.

**Branch tip at D0 close**: pre-state `2680252573`; state close commit adds after this write.
**Date closed**: 2026-04-22

### Problem

Python subprocess path (`csv_train`) showed systematic leaf-magnitude shrinkage (pred_std_R ≈ 0.69×) vs CPU CatBoost. Depthwise/Lossguide showed catastrophic collapse (~560%/598% RMSE delta). v5's ULP=0 record is kernel-output only and did NOT cover `FindBestSplit`, basePred, quantization borders, or nanobind orchestration. Surfaced as a gap in parity-gate coverage.

### Root causes (two, landed under DEC-012)

| # | DEC | Path | Summary |
|---|-----|------|---------|
| 1 | DEC-028 | SymmetricTree noise | `FindBestSplit` scaled RandomStrength noise by `totalWeight / numPartitions` (dimensionally wrong — scales with dataset size). Replaced with CPU's `sqrt(sum(g²)/N)` gradient-RMS formula. |
| 2 | DEC-029 | Depthwise/Lossguide model JSON | `TTreeRecord.SplitProps` never populated in non-oblivious paths → `WriteModelJSON` emitted `"splits": []` → Python predict sent every doc to leaf 0. Added `SplitBfsNodeIds`, populated `SplitProps` in both paths, emitted `grow_policy` + `bfs_node_index` per split (+ `leaf_bfs_ids` for Lossguide), dispatched Python predict on `grow_policy` with bit-packed BFS traversal. |

### Exit gate results (all PASS)

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| G0 | Root cause(s) in DECISIONS | DEC-028 + DEC-029 entries complete | PASS |
| G1 | SymmetricTree 18-cell parity, segmented | rs=0 9/9 within ±2% (max 0.43%); rs=1 9/9 MLX ≤ CPU, pred_std_R ∈ [0.9996, 1.087], Pearson > 0.99 | PASS |
| G2 | Depthwise + Lossguide rs=0 parity | DW −0.64%, LG −1.01% vs CPU (pre-fix 561%/598%) | PASS |
| G3 | Python-path regression test live | 8/8 pytest PASS in 6.32s (`tests/test_python_path_parity.py`) | PASS |
| G4 | bench_boosting ULP=0 preserved | Kernel sources untouched; v5 record unchanged | PASS |
| G5 | Determinism | 100 runs @ N=10k/seed=1337/rs=0, max−min = 1.49e-08, std 6.17e-09 | DETERMINISTIC |

### Segmented-gate rationale (methodology note)

Strict symmetric `ratio ∈ [0.98, 1.02]` false-fails cells where MLX is *better* than CPU. CPU and MLX use independent RNGs; at same seed they draw different noise realizations. Segmenting the gate:
- **rs=0**: tight `ratio ∈ [0.98, 1.02]` (no PRNG divergence to explain away).
- **rs=1**: one-sided `MLX_RMSE ≤ CPU_RMSE × 1.02` **AND** `pred_std_R ∈ [0.90, 1.10]`.

`pred_std_R` catches leaf-magnitude shrinkage directly — DEC-028's signature was 0.69×. Segmentation retained transparently alongside the strict-symmetric result (12/18 under strict).

### Follow-ups (opened, not blocking S26 D0 close)

1. **ComputeLeafIndicesDepthwise validation path**: C++ returns `nodeIdx − numNodes` instead of bit-packed partition order. Affects validation RMSE tracking during Depthwise training only; does not affect training correctness or Python predictions. Listed in DEC-029 Risks.
2. **MLX Depthwise/Lossguide RandomStrength noise path**: `FindBestSplitPerPartition` has no noise injection. At rs=1 these policies under-fit CPU by ~10–12% at N=10k. Pre-existing — not a S26 regression. Needs a separate parameter-threading pass.

### Files of record

- `docs/decisions.md` DEC-028, DEC-029
- `docs/sprint26/d0/g1-g3-g4-report.md` — gate report
- `docs/sprint26/d0/d0-8-verification.md` — rs=0/rs=1 controlled table
- `docs/sprint26/d0/depthwise-lossguide-root-cause.md`, `leaf-magnitude-code-diff.md` — diagnostics
- `benchmarks/sprint26/d0/g1_sweep.py` + `g1-results.md` — 18-cell sweep
- `benchmarks/sprint26/d0/g4_determinism.py` + `g4-determinism.md` — 100-run determinism
- `benchmarks/sprint26/d0/one_tree_depthwise.py` + `one-tree-depthwise-instrumentation.txt` — DEC-029 evidence
- `tests/test_python_path_parity.py` — CI regression harness
- Cross-project: `../LESSONS-LEARNED.md` — 24 principle-first lessons captured during S26

## Sprint 24 — CLOSED

### Verdict: D0 PASS on parity (DEC-023 RESOLVED); FAIL on R8 preservation (Verstappen retroactive retreat). R8: 1.01× post-fix.

**Branch tip at close**: `784f82a891`
**Date closed**: 2026-04-21

| Track | Verdict | Key finding |
|-------|---------|-------------|
| D0 — DEC-023 v5 fix | PASS — all 4 gates | ULP=0, 18/18 parity, 100/100 gate determinism |
| R8 preservation | FAIL — retroactive | 1.90× was predicated on non-deterministic T2; v5 collapses to 1.01× |
| S24-BENCH-G1 — championship suite | NOT RUN | Campaign retreated before suite started |

### D0 detail

**Problem**: DEC-023 — features 1-3 `atomic_fetch_add` on float in T2-accum; bimodal output
at config #8 (~50/50 between 0.48231599 / 0.48231912, 105 ULP gap).

**Path 5 falsified**: All T2-sort + int-fixed-point variants retaining feature-0's bin-range scan
over `sortedDocs` pinned to Value B (105 ULP off T1's Value A). Root cause: reduction topology
difference between sort-based scan and T1's SIMD fold. Integer accumulation made features 1-3
deterministic but did not change feature-0's incompatible topology.

**CPU anchor (Path X)**: CPU CatBoost at config #8 = 0.068 (~24M ULP from both A and B).
Inconclusive — bench_boosting is a GPU-kernel-speed harness, not a CatBoost conformance test.
T1 Value A (0.48231599) remains the declared parity anchor by construction.

**Off-by-one retest (false positive)**: Proposed off-by-one between scoring kernel ("bin ≥ b
right") and apply path ("bin > b right") was a coordinate-system labeling artifact. Both paths
encode `raw_bin > splitIdx`, consistent with CatBoost's `IsTrueHistogram`. No bug present.
Diagnostic at `docs/sprint24/d0_offby1_cascade_retest.md`.

**v5 fix**: All four features (0-3) in T2-accum rewritten to T1-style SIMD-shuffle accumulation
reading from `docIndices`. T2-sort kernel removed from dispatch. ULP=0 is structural — v5
executes the identical FP computation as T1.

**Acceptance criteria results** (all 4 gates PASS):

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| S24-D0-G1 | Config #8: 10/10 deterministic | 10/10 at 0.48231599, ULP=0 | PASS |
| S24-D0-G2 | 18/18 ULP=0, ≥5 runs | 18/18 ULP=0, all 5/5 | PASS |
| S24-D0-G3 | Gate config: 100/100 deterministic | 100/100 at 0.47740927 | PASS |
| S24-D0-G4 | hist_ms ratio ≥ 0.45× | 0.959× | PASS |

### R8 collapse

| Metric | Pre-fix (T2 v4, S22 D4) | Post-fix (T2 v5, S24 D0) |
|--------|:-----------------------:|:------------------------:|
| hist_ms (gate config) | ~6.85 ms (0.317× T1) | ~20.75 ms (0.959× T1) |
| e2e speedup vs S16 baseline | **1.90×** | **~1.01×** |
| Verstappen ≥1.5× | cleared by 40 pp | **FAILED retroactively** |

T2's 0.317× hist_ms ratio derived from the sort-based feature-0 bin-range scan — the same
mechanism that produced a different reduction topology from T1and caused DEC-023. These are not
separable: the speed came from the topological difference; fixing the topology eliminates the
speed.

### DEC-023 resolved

Commit `784f82a891`. v5 is the shipped kernel. The 1.90× record is superseded by 1.01×.

### DEC-026 opened

Research track for S25: cascade-robust GAIN comparison. Hypothesis: a lexicographic tiebreak at
near-tie GAIN comparisons (when `|GAIN_A - GAIN_B| < ε`) could block the cascade amplification
that makes config #8's 1-2 ULP/bin topology difference grow to 105 ULP at iters=50. If the
tiebreak succeeds, T2 Path 5 (sort + int-fixed-point) becomes shippable at R8 ≈ 1.85–1.90×.
This is research, not engineering. Falsification checkpoints at each gate. See
`DECISIONS.md DEC-026`.

## Sprint 25 — CLOSED (FALSIFIED)

### Verdict: FALSIFIED at G1 on day 1. R8 stays at 1.01×. v5 is final production kernel.

**Branch tip at close**: (S25 FALSIFIED closeout commit)
**Date closed**: 2026-04-21

### G1 empirical result

| Quantity | Value | Source |
|---|---|---|
| Sweep runs | 180 (18 configs × 5 runs × 2 kernels) | 5 min 4 s wall |
| Determinism | 5/5 per (config, kernel) | all 180 bit-identical |
| T1 vs DEC-008 reference | 18/18 exact | T1 reproduces every reference loss |
| T1 vs Path 5 agreement | 17/18 bit-exact | only config #8 diverges (T1=A, Path5=B) |
| Flip events (earliest-per-iter) | 35 total, 7 unique × 5 runs | all at config #8 |
| ε_min (required to gate flips) | 2.200e-03 | config #8 iter 45 depth 0 |
| ε_max (incl. zero-gain ties) | 0.0 | configs 1/2/8/14 pure nodes |
| ε_max⁺ (positive floor) | 1.043e-07 | config #1 iter 40 depth 3 |
| Safety ratio (positive) | 4.74e-05 (target ≥ 2.0) | **21,091× below threshold** |

**Structural cause**: Path 5's flip gaps span 5.96e-08 to 2.2e-03 — the full range of legitimate
top-2 separations at non-#8 configs. No ε discriminates "ambiguous split" from "clear split"
when both share the same gain separation. Kill-switch fired cleanly.

**Implication**: R8 stays at 1.01× (post-S24 honest position, unchanged). Verstappen ≥1.5×
gate remains retroactively failed from S24 D0. v5 (`784f82a891`) is the final production
kernel. DEC-027 (alternative accumulation paths such as XGBoost-style per-feature
deterministic radix-sum) is acknowledged for future research but **not opened** as part of
S25 closure — Ramos dedicates dedicated time for it later.

See `docs/sprint25/g1_epsilon_calibration.md` for the full verdict doc including §9 forward
paths, and `benchmarks/sprint25/g1/results/` for raw artifacts.

## Next actions

1. **S27 — ACTIVE.** Merge PR #24 (S26-FU-2) first, then cut `mlx/sprint-27-correctness-closeout` from master. Start Track A (FU-1 repro harness) and Track B (anchor inventory) in parallel.
2. **S26 D0 — CLOSED.** PR pending Ramos open if not yet merged. CI sanity: pytest 8/8 on `tests/test_python_path_parity.py`; bench_boosting kernel sources untouched.
3. **DEC-027 — deferred (unchanged)**. Not opened. Reserved for a dedicated future research sprint.
4. **Standing orders** (unchanged): DEC-012 one-change-per-commit; no Co-Authored-By; RR-AMATOK only; parity sweep protocol ≥5 runs per non-gate + 100 runs at gate unconditionally.
5. **Standing order (S26 addition, carried forward)**: parity gates that are kernel-ULP only MUST be explicitly labeled as kernel-output-only in their gate spec. Python-path / nanobind / `FindBestSplit` / leaf-estimation parity requires its own harness (see `tests/test_python_path_parity.py`).

## Standing orders (carried forward)

- **No `Co-Authored-By: Claude` trailer** in any commit message — global policy.
- **RR-AMATOK fork only** — do not push or PR to `catboost/catboost` upstream.
- **DEC-012 one-structural-change-per-commit** — still active.
- **Honest R8** — 1.01× is the new position. Do not round, inflate, or annotate "but we had X
  at some point". The 1.90× figure is documented as superseded in DECISIONS.md and CHANGELOG.
- **Parity sweep protocol**: ≥5 runs per non-gate config; gate config unconditionally 100 runs.
  Standing order from S23 D0, unchanged.

## Prior sprints — status

- **Sprint 26 D0** — CLOSED 2026-04-22 on `mlx/sprint-26-python-parity`. DEC-028 (RandomStrength noise formula) + DEC-029 (non-oblivious tree SplitProps + BFS index) landed under DEC-012. All 5 exit gates PASS + determinism confirmed. R8 unchanged at 1.01×. PR pending open.
- **Latent-bugs cleanup (PR #20)** — merged 2026-04-22 as `71aabaa842`. Three commits under DEC-012: ledger hygiene (close K=10 + BUG-007, reframe S-1), `BuildDatasetFromArrays` groupIds sortedness CB_ENSURE, and `histogram.cpp` S-1 `static_assert`. No production behavior change.
- **State refresh (PR #19)** — merged 2026-04-22 as `1afd0a35b2`. Docs-only alignment of `HANDOFF.md` / `TODOS.md` / `CHANGELOG-DEV.md` with post-stack-merge reality.
- **CI fix (PR #18)** — merged 2026-04-22 as `9b0c03fec2`. Three commits unblocking the stack: MLX 0.31+ CLI breakage in `mlx-build.yaml`, stale `0.3.0`/`minor==3` version pins in `test_qa_round13_sprint10.py`, and overly-broad BUG-001 MAE sentinel in `test_qa_round8_sprint3_losses.py` (narrowed to SIGABRT-only). No production code changes.
- **Sprint 25** — CLOSED, FALSIFIED. Merged 2026-04-22 as `5caa6e64cf` (PR #17). DEC-026 FALSIFIED at G1: ε-threading impossible (safety ratio 4.74e-05 vs 2.0 target). R8 stays at 1.01×. DEC-027 deferred. No production code changes; shipped as empirical falsification evidence.
- **Sprint 24** — CLOSED. Merged 2026-04-22 as `1385e056ca` (PR #16). DEC-023 RESOLVED via v5 (T1 accumulation topology). R8 1.90× → 1.01× retroactive. Verstappen ≥1.5× gate failed. DEC-026 cascade-robust GAIN research opened S25.
- **Sprints 0–23** — merged to master.
