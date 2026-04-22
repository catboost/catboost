# Handoff — CatBoost-MLX

> Last updated: 2026-04-22 (Sprint 26 D0 CLOSED — Python-path parity correctness gap resolved; DEC-028 + DEC-029 shipped; R8 unchanged at 1.01×; no open PRs)

## Current state

- **Branch**: `mlx/sprint-26-python-parity` at tip `2680252573` (state close commit added after).
- **Campaign**: Post-Verstappen correctness sprint. S26 D0 Python-path parity CLOSED — all exit gates PASS. R8 unchanged at 1.01× (S26 is correctness-first, not perf).
- **Production kernel**: v5 (`784f82a891`) — unchanged. S26 did not touch `catboost/mlx/kernels/kernel_sources.h`; bench_boosting ULP=0 record preserved (G4).
- **Open PRs**: none. Most recent merges to master: #16 (`1385e056ca`) → #17 (`5caa6e64cf`) → #18 (`9b0c03fec2`) → #19 (`1afd0a35b2`) → #20 (`71aabaa842`). S26 D0 PR pending Ramos open.
- **Known bugs**:
    - BUG-T2-001 RESOLVED (`784f82a891`).
    - BUG-007 MITIGATED 2026-04-22 (`71aabaa842`) — two-layer defense (Python wrapper sorts; C++ throws on unsorted).
    - K=10 anchor mismatch RESOLVED Sprint 8 (TODO-022, `CHANGELOG.md:27`).
    - Sibling S-1 (`kHistOneByte` writeback race) latent; guarded by compile-time `static_assert` at `histogram.cpp:126`.
    - **S26-new follow-up**: `ComputeLeafIndicesDepthwise` (C++ validation path) still returns `nodeIdx − numNodes` — affects validation RMSE tracking only, not training correctness or Python predictions. Tracked in DEC-029 Risks section.
    - **Pre-existing follow-up (not a S26 regression)**: MLX Depthwise/Lossguide have no RandomStrength noise path — at `rs=1` these policies under-fit CPU by ~10–12% at N=10k. `FindBestSplitPerPartition` is where noise threading would need to be added. Scope: separate future sprint.

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

1. **S26 D0 — CLOSED.** 8 commits on `mlx/sprint-26-python-parity` + state close commit. PR pending Ramos open. CI sanity: pytest 8/8 on `tests/test_python_path_parity.py`; bench_boosting kernel sources untouched.
2. **S26 D1 (optional, not scheduled)** — Two candidate follow-ups are listed under *Sprint 26 → Follow-ups*: (a) ComputeLeafIndicesDepthwise validation-path fix, (b) MLX Depthwise/Lossguide RandomStrength noise path. Both are carry-forward, not blocking merge. Open only when Ramos sets scope.
3. **DEC-027 — deferred (unchanged)**. Not opened. Reserved for a dedicated future research sprint.
4. **Standing orders** (unchanged): DEC-012 one-change-per-commit; no Co-Authored-By; RR-AMATOK only; parity sweep protocol ≥5 runs per non-gate + 100 runs at gate unconditionally.
5. **New standing order (S26 addition)**: parity gates that are kernel-ULP only MUST be explicitly labeled as kernel-output-only in their gate spec. Python-path / nanobind / `FindBestSplit` / leaf-estimation parity requires its own harness (see `tests/test_python_path_parity.py`). Reason: v5 ULP=0 record coexisted with a 0.69× Python-path leaf-magnitude collapse for multiple sprints because the kernel gate was silently misread as "full parity".

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
