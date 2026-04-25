# Handoff — CatBoost-MLX

> Last updated: 2026-04-25 (S34-PROBE-F-LITE T0 closed on math verdict — branch `mlx/sprint-34-probe-f-lite`. Per-side mask is WRONG for one-hot Cosine; leave `csv_train.cpp:1698` joint-skip. CR-S33-S1 / SA-N1 resolved as no-fix-needed. T1 empirical sweep not run; math-first sufficient. New ticket #128 S35-Q4-L2-PARENT-TERM opened from math-derivation §8.2 Q4. S33 PR #29 open.)

## Current state

- **Active sprint**: **S34-PROBE-F-LITE** on branch `mlx/sprint-34-probe-f-lite`. T0a (math) + T0b (code) converged: per-side mask is WRONG for one-hot Cosine; recommended fix shape is "leave joint-skip as-is". CR-S33-S1 / SA-N1 resolved as no-fix-needed. T1 (empirical CPU-reference sweep) NOT run — math-first sufficient. Verdict: `docs/sprint34/probe-f-lite/verdict.md`. **S33 ARCHIVED** on branch `mlx/sprint-33-iter2-scaffold` tip `addd09ae0e`, PR #29 open. All S33-L4-FIX commits landed (1, 1.5, 2, 3a, 3b); docs-only close-out commits S33-S2/S33-S3/S33-S1-RESCOPE also landed.
- **S33 status**: FULLY CLOSED. L0→L3 chain closed. PROBE-E confirmed partition-state mechanism. S33-L4-FIX landed: per-side mask (Commits 1+1.5), four-gate validation (Commit 2), ST guard removal (Commit 3a), LG guard removal (Commit 3b). DEC-036 CLOSED. DEC-040 CLOSED. DEC-042 FULLY CLOSED (ordinal-branch scope). DEC-032 FULLY CLOSED. #93 + #94 COMPLETED.
- **#123 S33-L4-FIX**: COMPLETED 2026-04-25. Gate report: `docs/sprint33/commit2-gates/REPORT.md`. Commit 3 closures: `e1d72d64e8` (ST), `d599e5b033` (LG).
- **Surviving narrowing (PROBE-C corrected, 2026-04-24)**: at iter=2, depth=0 AND depth=1 of tree[1] AGREE between CPU and MLX (depth-0 ULP-identical at border 0.014169; depth-1 borders differ by 4.6e-6 due to MLX's 127-vs-128 grid offset, but both pick feat=1 at the equivalent logical position). **Divergence kicks in at depth=2**: CPU picks feat=0 again (border=−0.947), MLX picks feat=10 (border=+0.306). y = 0.5·X[0] + 0.3·X[1] + 0.1·noise → feat=10 is pure noise; CPU correctly stays on signal, MLX prefers a noise feature. Mechanism is in the Cosine gain argmax at iter=2 depth=2, with depths 0–1 already shared. The L3 "depth-0 divergent" verdict is fully retracted: it conflated CPU CBM stored-coords (split_index=3) with MLX upfront-grid bin=64 — both physically 0.014169.
- **PROBE-C side findings** (`docs/sprint33/probe-c-borders/data/`): MLX's 127-border grid is a strict subset of CPU's 128 (ULP=1 tolerance); each feature is missing exactly one border at CPU index 6 (e.g. feat=0 missing −1.65587). This 1-border deficit is real but does not directly explain the depth=2 divergence — the depth=2 split is on feat=0 vs feat=10, not on the missing border. The deficit may compound across depths/iters but is not the primary lever.
- **PROBE-D result (2026-04-24)** (`docs/sprint33/probe-d/FINDING.md`, `data/cos_accum_seed42_depth{0..5}.csv`): fp32-vs-fp64 gain shadow at iter=2 d=0..5, 2540 (feature, bin) cells per depth. Max abs(g_f32−g_f64) = 3.89e-5; argmax(g_f32)==argmax(g_f64) at every depth. **Smoking gun**: at d=2, 18 noise features cluster at gain ~101.95, signal feat=0 evaluates to 81.89, feat=1 to 77.77 — signal/noise gain INVERSION. CPU's d=2 pick (feat=0, bin=21, border=−0.946874) is bin-mapped ULP-identically and scores 81.887 in MLX; MLX's winner (feat=10, bin=79) scores 101.954. Gap = **20.07 gain units** — three orders of magnitude bigger than any plausible fp32 noise. **Precision-class hypothesis closed.**
- **PROBE-E result (2026-04-24)** (`docs/sprint33/probe-e/FINDING.md`, `data/cos_leaf_seed42_depth{0..5}.csv`): per-(feat, bin, partition) capture at iter=2 d=2 of MLX's actual contribution and CPU's counterfactual. **Mechanism FULLY SPECIFIED**: at `csv_train.cpp:1980` the `if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;` line skips the entire partition's contribution to both `cosNum` and `cosDen`. CPU's reference (`short_vector_ops.h:155+` SSE2 `UpdateScoreBinKernelPlain`) instead masks the empty-side average to 0 but **adds the non-empty side**. Per-partition smoking gun for CPU's d=2 pick (feat=0, bin=21): 4 partitions, 2 are degenerate (wL=0); MLX adds {5718, 0, 974, 0} = cosN 6692 → gain 81.83; CPU adds {5718, 275, 974, 4761} = cosN 11728 → gain 108.32. Δ=+26.49 gain units, exactly enough to flip the d=2 pick from feat=10 (noise, 101.79) to feat=0 (signal). Top-5 by CPU gain is feat=0 bins 104–108 (signal), all skips=2/4. Skip rate grows monotonically with depth: 0% / 2.5% / 5.0% / 7.6% / 10.6% / 14.6% at d=0..5. **All 127 non-trivial bins on feat=0 at d=2 have skips=2** (d=0 split on feat=0 fixed every doc's relationship to feat=0 range). DEC-036 root-caused.
- **Next step** (S33-L4-FIX #123 REOPENED with mechanism known): replace the `continue` at `csv_train.cpp:1980` with a per-side mask in the per-partition update — when `weightLeft < 1e-15` contribute zero from the left side but still add the right side's `sumR²/(wR+λ)` (and den equivalent), and vice versa. Validate on the same 50k anchor: predicted MLX d=2 pick flips to feat=0 bin ≈ 102–108 matching CPU; iter=2 tree[1] d=2 split matches CPU; DEC-036 drift collapses below 1.07× R8 threshold. If the partition-loop fix needs reducer-level coordination (e.g. for empty-leaf grad/hess mass), splits cleanly into a separate task.
- **Preserved diagnostic artifacts** (`docs/sprint33/l4-fix/data/`): `mlx_grad_iter2.bin`, `mlx_hess_iter2.bin` (bit-identical to CPU at iter=2 start; rules out S1 GRADIENT class); `mlx_hist_d0_iter2.bin` (correct; per-feature histogram total ≈ −739 is right; the L3 verdict's expected formula `20×sum_g=0.228` was wrong); `mlx_partstats_d0_iter2.bin` (partition-stats, may be useful for next phase).
- **S32 status**: CLOSED. Branch `mlx/sprint-32-cosine-gain-term-audit`, tip `3e472ac49f`. DEC-038 (allVals fix) + DEC-039 (fold_count cap 127) shipped. G3a PASS, G3b FAIL (52.6%), G3c PASS, G3d PASS.
- **S31 status**: CLOSED jointly with S32. All S31 quality gates subsumed into S32 close.
- **Campaign**: Post-Verstappen correctness. DEC-036 CLOSED (2026-04-25, ordinal-branch). DEC-032 FULLY CLOSED (guards removed via S33-L4-FIX Commits 3a/3b). DEC-042 FULLY CLOSED (ordinal scope; one-hot branch under investigation in S34-PROBE-F-LITE).
- **Production kernel**: v5 (`784f82a891`) — unchanged. Kernel sources md5 `9edaef45b99b9db3e2717da93800e76f` byte-identical across all S31+S32+S33 commits.
- **Open PRs**: none.
- **Known bugs**:
    - BUG-T2-001 RESOLVED (`784f82a891`).
    - BUG-007 MITIGATED 2026-04-22 (`71aabaa842`).
    - K=10 anchor mismatch RESOLVED Sprint 8 (TODO-022, `CHANGELOG.md:27`).
    - Sibling S-1 (`kHistOneByte` writeback race) latent; guarded by compile-time `static_assert` at `histogram.cpp:126`.
    - **S27-FU-1 RESOLVED** (`fb7eb59b5f`): `ComputeLeafIndicesDepthwise` encoding + split-lookup bugs fixed. DEC-030. DEC-029 Risks entry retired.
    - **S28-DEC-032 FULLY CLOSED** (2026-04-25): `EScoreFunction` enum dispatched. DW+Cosine ships in-envelope. LG+Cosine and ST+Cosine guards removed (S33-L4-FIX Commits 3a/3b). All three grow policies now support Cosine. See DEC-042 for the closing mechanism.
    - **S29-DEC-034 RESOLVED** (outcome A, `64a8d9076b`): LG+Cosine shares float32 joint-denominator compounding with ST+Cosine. Single Kahan fix addresses both.
    - **DEC-036 CLOSED 2026-04-25** (PROBE-E + S33-L4-FIX): mechanism was degenerate-child skip at `csv_train.cpp:1980`; per-side mask fix collapsed drift from 52.6% to 0.027%. See DEC-042. Superseded by DEC-042 (ordinal-branch closure).
    - **DEC-041 INVALIDATED**: premise (static vs dynamic quantization) falsified by probes A/B. Number is dead; do not reuse. Successor will be DEC-042 or later.
    - **Latent P11**: hessian-vs-sampleWeight semantics swap at `csv_train.cpp:3780, 3967`. Fires under Logloss/Poisson/Tweedie/Multiclass. Not blocking RMSE path.
    - **Meta-pattern (S30→S33)**: agents shipping plausible-but-wrong verdicts in cold subagent contexts under shipping pressure is now a recognized failure mode. Successor probe runs in active main context with L0-L3 evidence held live, not in fresh subagent contexts.

## Sprint 29 — DEC-032 Closeout + LG Mechanism Spike (CLOSED 2026-04-23)

**Branch**: `mlx/sprint-29-dec032-closeout`
**Cut from**: master `987da0e7d5`
**Tip**: `fa7f9b55fc` (7 commits)
**Authoritative record**: `docs/sprint29/sprint-close.md`

### Summary

CLI-GUARD ported to C++ nanobind entry and CLI entry — SA-H1 closed. LG mechanism spike ran
in one session; outcome A verdict (`64a8d9076b`). DEC-034 RESOLVED. DEC-032 still PARTIALLY
CLOSED pending S30-COSINE-KAHAN. CR PASS-WITH-NITS (0 must-fix). SA PASS (0 findings).

### S29 commits (oldest → newest)

| SHA | Tag | Description |
|-----|-----|-------------|
| `33ce5f1d66` | S29-00 | Kickoff; state files |
| `73e9460a31` | S29-CLI-GUARD-T1 | Guards ported to C++ nanobind + CLI |
| `c73f5073af` | S29-CLI-GUARD-T2 | pytest coverage (4 cases) |
| `503ebacdb2` | S29-LG-SPIKE-T1 | LG+Cosine iter-1 drift measurement |
| `64a8d9076b` | S29-LG-SPIKE-T2 | DEC-034 verdict — outcome A |
| `3f87b85e38` | S29-CR + S29-SA | Gate reports — CR PASS-WITH-NITS, SA PASS |
| `fa7f9b55fc` | S29-CR SF-1 | Verdict wording tightened |

## Sprint 28 — Score Function Fidelity (CLOSED)

**Branch**: mlx/sprint-19-hist-writeback (working branch carrying S28 commits)
**Tip**: e0b0b1b527
**Previous sprint**: S27 closed 2026-04-23 via PR #25 (4b3711f82b)
**Authoritative record**: `docs/sprint28/sprint-close.md`

### Summary

MLX now dispatches on `EScoreFunction` enum across all three grow policies (DepthWise,
Lossguide, SymmetricTree), matching CPU CatBoost's `{L2, Cosine}` behavior with explicit
rejection of `{NewtonL2, NewtonCosine}`. DW+Cosine ships in-envelope at 1.6% drift.
LG+Cosine and ST+Cosine are guarded at the Python API pending S29 root-cause (RCA) and
Kahan summation work. All 28 parity cells re-blessed with explicit `score_function` labels.

### S28 commits (oldest → newest)

| SHA | Tag | Description |
|-----|-----|-------------|
| `0409e632fa` | S28-00 | Branch kickoff, state files updated |
| `da02da0259` | S28-AUDIT | Formal confirmation: zero score_function plumbing in catboost/mlx/ |
| `83f30c3677` | S28-COSINE | `ComputeCosineGainKDim` helper from CPU `TCosineScoreCalcer` |
| `0ea86bde21` | S28-L2-EXPLICIT | `EScoreFunction` enum + `ParseScoreFunction` + DW/LG dispatch + nanobind binding + Python `_validate_params` rejecting `NewtonL2`/`NewtonCosine` |
| `4083add248` | S28-OBLIV-DISPATCH | Dispatch mirrored into `FindBestSplit` (SymmetricTree) |
| `c07e895f7c` | S28-REBLESS | 8 parity cells relabeled with explicit `score_function` |
| `dca62f0d72` | S28-FU3-REVALIDATE | DW force-L2 lifted; LG retains force-L2 pending S29 RCA |
| `b9577067ef` | S28-{LG,ST}-GUARD | Two Python `ValueError` guards for Cosine+Lossguide and Cosine+SymmetricTree |
| `e0b0b1b527` | S28-CR-S1 | Dead `ComputeCosineGain` scalar helper removed (code-review cleanup) |

### Parity suite

28/28 PASS at `b9577067ef`. Unchanged at `e0b0b1b527`.

## Sprint 30 — S30-COSINE-KAHAN — CLOSED 2026-04-24 (PR #28, `17451f4780`)

**Branch**: `mlx/sprint-30-cosine-kahan` merged → master `17451f4780`; branch deleted.
**Basis**: DEC-034 outcome A (moderate confidence); DEC-035 executed in full (phased plan + full verification battery).
**Outcome**: Precision fix class exhausted. Two proper fixes shipped (K4 + Fix 2); ST/LG guards remain in place. DEC-036 opens structural divergence investigation for S31.
**Authoritative record**: `docs/sprint30/sprint-close.md`.

### Executed phases

| Phase | Task | Gate | Result |
|-------|------|------|--------|
| T1 | #90 INSTRUMENT | G1 mechanism fingered | PASS (cosDen, residual 4.067e-3) |
| T2 | #91 KAHAN | G2 ≥10× residual reduction | PASS (12.5×); K4 fired → fp64 widening |
| T3 | #92 MEASURE | G3a/G3b/G3c 2-tier envelope | **FAIL** (53.30% ST; K2 fired) |
| D1 | #100 CPU AUDIT | CPU precision baseline | CPU is fp64 throughout (static_assert `__m128d`) |
| D2 | #101 FULL-STACK | Locate binding layer | Initially ruled out L3/L4; V2 later invalidated the methodology |
| D2-redux | #106 FIX METHODOLOGY | Honest L3/L4 measurement | L3/L4 RULED OUT (5.03e-5 residual, 0/18 flips) |
| D3 | #102 LG OUTCOME A/B | Discriminate LG path | Outcome B confirmed for LG (priority-queue divergence) |
| D4 | #107 JOINT-DENOM 64× | V5 amplification hypothesis | FALSIFIED (measured 2.42×, not 64×) |
| V1 | #103 N-SCALING | L0 precision-class predictor | FLAT — exponent b = 0.0017 |
| V5 | #105 DW @ 50k | Isolate ST-specific mechanism | MIXED — L0 real but 8.4× DW/ST gap unexplained |
| V6 | #109 N=500 CONFIRMER | Cheap L1 falsification | **L1 FALSIFIED** — drift 50.72% at N=500 vs 53.30% at N=50k (b ≈ 0 across 100× N range) |
| Fix 2 | #108 FP64 GAIN | L3/L4 widening | SHIPPED; ST drift 53.30% → 53.30% (prediction failed cleanly) |

### Ships from S30

- **K4** — fp64 cosNum/cosDen accumulator widening (commit-family around `108c7a59d2`)
- **Fix 2** — fp64 `totalGain`, `bestGain`, `TBestSplitProperties::Gain`, `perturbedGain`, `TLeafCandidate::Gain` (`90a0cb4475`, `364d4ee962`)
- **13 verdict docs** under `docs/sprint30/` — full chain of evidence for precision-class exhaustion
- **Instrumentation behind `COSINE_RESIDUAL_INSTRUMENT`** in `catboost/mlx/tests/csv_train.cpp` — retained for S31 audit reuse
- **Guards unchanged** — Python + C++ nanobind + CLI all still reject `{ST,LG}+Cosine`

### Does NOT ship

- T4a (#93 ST-REMOVE) — deferred; mechanism not fixed
- T4b (#94 LG-REMOVE) — deferred; additional outcome-B mechanism confirmed by D3
- T5 (#95 CLI exit wrap) — carried to S30 close or S31 T-cleanup
- T6 (#96 S29 CR nits) — carried to S30 close or S31 T-cleanup

### Close-out tasks — DONE

- #97 S30-CR — APPROVE (0 must-fix, 5 nits). `docs/sprint30/sprint-close/cr-report.md`.
- #98 S30-SA — PASS-WITH-FINDINGS (0 CRITICAL/HIGH/MEDIUM, 3 LOW, 3 INFO). `docs/sprint30/sprint-close/sa-report.md`.
- #99 S30-CLOSE — close doc, DEC-035 closure addendum, DEC-036 OPEN, PR #28 merged `17451f4780`.

---

## Sprint 32 — COSINE-GAIN-TERM-AUDIT — CLOSED 2026-04-24

**Branch**: `mlx/sprint-32-cosine-gain-term-audit` (tip `3e472ac49f`, 7 commits on S31 tip `9b3a5238a7`).
**Authoritative record**: `docs/sprint32/sprint-close.md`.

### What shipped

- **DEC-037** (S31 co-fix, `746d5090b5`): border count off-by-one + DP algorithm restoration. CLOSED S31.
- **DEC-038** (`901bc760ac`): `GreedyLogSumBestSplit` allVals fix. CLOSED.
- **DEC-039** (`901bc760ac`): `fold_count` cap 127 (T2_BIN_CAP contract). CLOSED.

### Gate summary

| Gate | Result |
|------|--------|
| G3a — depth=0 gain ratio (3 seeds) | PASS: 1.000000 ± 5e-7 |
| G3b — iter=50 ST+Cosine drift | FAIL: 52.6% (DEC-036 open) |
| G3c — bench_boosting v5 ULP=0 | PASS: 0.48231599 = AN-009 |
| G3d — 18-config L2 non-regression | PASS: 18/18 |

### S31 — ITER1-AUDIT — CLOSED jointly with S32

**Branch tip**: `9b3a5238a7` (8 commits; merged into S32 branch history).

All S31 deliverables subsumed. T3b G1 PASS (GAIN-FORMULA localized). T3-MEASURE, T4a/b deferred to S33.

---

## S34 status — T0 closed on math verdict

S34-PROBE-F-LITE T0a (mathematician) + T0b (ml-engineer) converged independently: **per-side mask is WRONG for one-hot Cosine; leave `csv_train.cpp:1698` as joint-skip**. Cosine has no parent-term subtraction anywhere → no cancellation when wR=0 → mirroring L1980's fix would inject `totalSum²/(totalWeight+λ)` into cosNum_d for every degenerate (p,k) → bin-dependent argmax bias toward rare-category bins. Predicts the 3% in-session regression directionally. T1 empirical sweep NOT run — math-first closed the question.

**Resolves**: CR-S33-S1, SA-N1-S33 (both as no-fix-needed). DEC-042 ordinal-scope footnote stands; one-hot branch confirmed correct as-is.

**Verdict doc**: `docs/sprint34/probe-f-lite/verdict.md`.

**New ticket from S34 side finding**: #128 S35-Q4-L2-PARENT-TERM — MLX's L2 path subtracts `totalSum²/(totalWeight+λ)` per-(p,k) at `csv_train.cpp:1704, 1973`; CPU's L2 path subtracts no parent term. Argmax-invariant at depth=0 numPartitions=1; variable elsewhere. Currently papered over by G4d.

## Entry point for next session

**Carry-forwards from S31/S32/S33** (still open):
- S31-T-LATENT-P11: low priority; hessian-vs-sampleWeight semantics swap at `csv_train.cpp:3780, 3967`.
- S31-T-CLEANUP: SA-I2 try/catch + S29 CR nits.
- #113 S31-T3-MEASURE: re-run S30 T3 gate matrix post-fix.
- **#128 S35-Q4-L2-PARENT-TERM**: investigate L2 parent-term divergence (new from S34 math derivation §8.2 Q4).

## PR state

S33 PR #29 OPEN: https://github.com/RR-AMATOK/catboost-mlx/pull/29 (DEC-036 closed for ordinal branch). S34 close-out commit pending. All prior PRs (S28, S29, S30) merged.

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
