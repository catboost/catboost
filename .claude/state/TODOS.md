# Active Tasks — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git/agent-memory on 2026-04-15. Sprint 16+ is source of truth.
> Last header refresh: 2026-04-25 (S33 + S34 + S35 ALL MERGED — PRs #29 #31. Master tip `a10ebd63e1`. DEC-036 CLOSED, DEC-042 FULLY CLOSED for both ordinal and one-hot branches. Local merged branches cleaned up.)

## Current state (2026-04-25)

- **Active branch**: `master` at `a10ebd63e1` (PR #31 merge). No active sprint branch.
- **Production kernel**: v5 (`784f82a891`), shipped S24 D0. ULP=0 structural parity across DEC-008 envelope via `bench_boosting`. Kernel sources md5 `9edaef45b99b9db3e2717da93800e76f` byte-identical S30 → S35.
- **R8 (honest)**: 1.01× e2e vs S16 baseline. Unchanged.
- **Open PRs**: none. PRs #29 (S33), #31 (S34/S35) merged 2026-04-25.
- **Active DEC**: **DEC-036 CLOSED** (2026-04-25, ordinal); **DEC-040 CLOSED** (investigation concluded); **DEC-041 INVALIDATED** (dead number); **DEC-042 FULLY CLOSED** (both ordinal + one-hot branches resolved); DEC-037/038/039 CLOSED. #93/#94/#123/#127/#128/#129 COMPLETED.
- **Open backlog**: #113 S31-T3-MEASURE re-run, #114 S31-T-CLEANUP, S31-T-LATENT-P11 (Logloss/Poisson/Tweedie), SA carry-forwards (SA-L1-S33 hard-coded paths, SA-L3-S30/SA-N2-S33 instrumentation env-var hardening, SA-I2-S29 #95 CLI exit wrap).

## Sprint 33 — Iter≥2 Runaway Divergence SCAFFOLD — OPEN 2026-04-24

**Branch**: `mlx/sprint-33-iter2-scaffold` (cut from S32 tip `9fcc9827d9`).
**Driver**: DEC-040. Closes DEC-036.
**Strategy**: L0-L4 cost-ordered falsification of three frames (A trajectory lock-in, B per-iter mechanism, C config/RNG).
**Hard rule**: any second structural change discovered while fixing the first → STOP, commit first atomically, then continue (DEC-012 enforcement after S31/S32 violations).

### Tasks

- [x] **#119 S33-L0-CONFIG** — Dump CPU vs MLX effective config field-by-field. Falsifies Frame C-config (~45 min). Verdict: `docs/sprint33/l0-config/verdict.md`. Owner: @ml-engineer. **COMPLETED 2026-04-24. Overall class: NO-DIFF. Frame C-config FALSIFIED. L0-PASS. Proceed to L1.**
- [x] **#120 S33-L1-DETERMINISM** — Shift to deterministic config; remeasure drift. Falsifies Frame C-RNG (~2h). Blocked by #119. **COMPLETED 2026-04-24. Class: FALSIFIED. Median drift 52.643% (seeds 42/43/44) — statistically identical to S32 baseline 52.6%. Frame C-RNG FALSIFIED. Frame C (Config/RNG) fully closed. Proceed to #121 L2-GRAFT. Verdict: `docs/sprint33/l1-determinism/verdict.md`.**
- [x] **#121 S33-L2-GRAFT** — Inject CPU iter=1 tree into MLX; measure post-graft drift. Discriminates Frame A vs B. **COMPLETED 2026-04-24. Class: FRAME-B. Median grafted drift 51.291% vs ungrafted 52.643% (ratio 0.974). Graft had zero effect — per-iter persistent bug confirmed. Frame A falsified. Verdict: `docs/sprint33/l2-graft/verdict.md`. Proceed to #122 L3-ITER2.**
- [x] **#122 S33-L3-ITER2** — Per-leaf, per-doc iter=2 instrumentation (conditional on Frame B). **COMPLETED 2026-04-24. Class: SPLIT. S1-GRADIENT bit-identical; S2-SPLIT divergent (CPU feat=0,bin=3 vs MLX feat=0,bin=64). S3/S4 cascade. Histogram anomaly: hist grad sum=-739 vs expected +0.23 (iter=2 grads correct; histogram kernel receives stale data). Suspected root: csv_train.cpp:4534 statsK construction. Verdict: `docs/sprint33/l3-iter2/verdict.md`. Data: `docs/sprint33/l3-iter2/data/`. Proceed to #123 L4-FIX.**
- [x] **#123 S33-L4-FIX** — COMPLETED 2026-04-25. Commits 1 (`10c72b4e96`) + 1.5 (`e98c6725cd`) shipped per-side mask for Cosine and L2 paths. Commit 2 gate validation PASS (all 5 gates): G4a 0.0001%, G4b 0.027% (was 52.6%), G4c ULP=0, G4d 18/18 [0.9991,1.0008], G4e 5/5 DW+Cosine intact. DEC-036 RESOLVED, DEC-042 RESOLVED. Guard removal (Commit 3, #93/#94) is now unblocked. See `docs/sprint33/commit2-gates/REPORT.md`.
- [x] **#124 S33-PROBE-C** — COMPLETED 2026-04-24. iter=2 divergence at d=2 (not d=0); CPU feat=0 vs MLX feat=10. L3 retracted. Tip `044ec9a5a8`.
- [x] **#125 S33-PROBE-D** — COMPLETED 2026-04-24. fp32/fp64 gain shadow at d=0..5; max residual 3.89e-5 vs 20-unit gap. **Precision class CLOSED.** Tip `d246e00fae`.
- [x] **#126 S33-PROBE-E** — COMPLETED 2026-04-25. Per-(feat, bin, partition) capture at iter=2 d=2 confirms partition-state class. Mechanism: `csv_train.cpp:1980` `continue` skips entire partition; CPU's mask formula adds non-empty side. Per-partition smoking gun: feat=0 bin=21 (CPU's pick) MLX gain 81.83 vs CPU gain 108.32 (Δ=+26.49); feat=10 bin=79 (MLX's pick) identical 101.79 both. Top-5 by CPU gain at d=2 is 5/5 feat=0 (signal). Skip rate 0%/2.5%/5%/7.6%/10.6%/14.6% at d=0..5. **DEC-036 root-caused; DEC-042 opened.** FINDING: `docs/sprint33/probe-e/FINDING.md`.

### Carry-forwards (still pending)

- **#114 S31-T-CLEANUP** — S30 carry-forwards (CLI exit wrap + S29 CR residuals).
- **#113 S31-T3-MEASURE** — re-run T3 gate matrix post-fix (will run after L4).
- [x] **#93 S30-T4a-ST-REMOVE** — COMPLETED 2026-04-25. S28-ST-GUARD removed (Commit 3a `e1d72d64e8`). G4a 0.0001%, G4b 0.027% (pre-fix 52.6%). Python path sanity ratio 1.000271. 4/4 tests PASS.
- [x] **#94 S30-T4b-LG-REMOVE** — COMPLETED 2026-04-25. S28-LG-GUARD removed (Commit 3b `d599e5b033`). LG+Cosine drift measured post-fix: iter=1 0.0000%, iter=50 0.382% (<=2% threshold). 4/4 tests PASS.
- **S31-T-LATENT-P11** — DEFERRED 2026-04-25 (S36 PROBE-F1 commit `dc0bad3916`). Math-first verdict: variable-swap fix, ~150 lines + kernel md5 change. T1 empirical probe REFUTED predicted drift magnitude by ~3 orders: iter=50 Logloss 0.108% max (Adult, real-world) vs predicted 10-30%; aggressive Poisson stress 0.252% vs predicted 20-50%. Math direction confirmed (drift monotone-increasing with iter, iter=1 zero) but implicit-bias caveat is empirically active. Fix recipe documented at `docs/sprint36/p11/math-derivation.md` §8.1 + `code-reading.md`. Reopen if real-world workload surfaces drift > 1%.
- [x] **#127 S34-PROBE-F-LITE** — COMPLETED 2026-04-25. Math-first verdict (T0a + T0b converged): per-side mask is WRONG for one-hot Cosine; leave `csv_train.cpp:1698` as joint-skip. Cosine has no parent-term subtraction anywhere → no cancellation when wR=0 → mirroring L1980's fix would inject `totalSum²/(totalWeight+λ)` into cosNum_d for every degenerate (p,k) → bin-dependent argmax bias toward rare-category bins (matches the empirical 3% regression direction). T1 empirical sweep not run; math-first closed the question. CR-S33-S1 and SA-N1 resolved as no-fix-needed. Verdict: `docs/sprint34/probe-f-lite/verdict.md`.
- [x] **#128 S35-Q4-L2-PARENT-TERM** — COMPLETED 2026-04-25. Math-first verdict: **argmax-invariant for the ordinal L2 path. High confidence.** MLX's per-(p,k) parent term is a constant offset across all (feat, bin) candidates (totalSum, totalWeight depend on (p,k) only — never on (f,b)); CPU's L2 has no parent term but argmax operates on absolute scores so additive constant is invisible. G4d's pass is mathematically expected, not luck. Active set A(f,b) = {(p,k) : W_{p,k} > 2·1e-15} = A_step ∀(f,b) by pigeonhole on wL+wR=W_{p,k}. Verdict: `docs/sprint35/q4-l2-parent-term/math-derivation.md`. New finding spawned #129.
- [x] **#129 S35-1H-L2-PER-SIDE** — COMPLETED 2026-04-25. Per-side mask applied to one-hot L2 at `csv_train.cpp:1704+`; one-hot Cosine joint-skip preserved per S34 verdict. Smoke (8k synthetic 1-cat anchor, depth=6, bins=32, lr=0.05, l2=3, seed=42): L2 loss curve byte-identical pre/post (math no-op-but-correct confirmed); Cosine loss curve byte-identical pre-S33-S1/post-#129 (joint-skip semantics preserved via `if (!wL_pos || !wR_pos) break;`). Kernel md5 unchanged.

---

## Sprint 27 — Correctness Closeout

**Branch**: `mlx/sprint-27-correctness-closeout`
**Scope**: Close S27-FU-1 (DW validation-path index), S27-AA (anchor audit), and S27-FU-3 (DW N=1000 parity triage). Zero kernel changes. Zero perf work.

### Track A — S27-FU-1: Depthwise validation-path index fix

- [x] **S27-FU-1-T1**: Repro harness: instrument `ComputeLeafIndicesDepthwise` at `csv_train.cpp:1751`; capture returned indices vs expected bit-packed BFS on DW N=10k d=3 — @qa-engineer
- [x] **S27-FU-1-T2**: CPU-source audit: confirm CatBoost BFS encoding (`nodeIdx − numNodes` wrong decode); draft DEC-030 — @ml-engineer
- [x] **S27-FU-1-T3**: Implement fix: replace arithmetic decode with root-to-leaf traversal mirroring `ComputeLeafIndicesLossguide` — @ml-engineer
- [x] **S27-FU-1-T4**: Gate G1-FU1: DW validation RMSE (`use_best_model=True`) within rs=0 tight band, 3 seeds × {N=10k, N=50k} — @qa-engineer
- [x] **S27-FU-1-T5**: DEC-030 authored; DEC-029 Risks entry retired — @technical-writer

### Track B — S27-AA: Anchor audit (parallel to Track A)

- [x] **S27-AA-T1**: Enumerate all numeric anchors in committed test/bench files; produce inventory `{path, line, value, last-touched-sha, captured-context}` — @qa-engineer
- [x] **S27-AA-T2**: Re-run each anchor's generating harness on current master; diff ≥1e-2 flags drift — @qa-engineer
- [x] **S27-AA-T3**: For each drifted anchor: classify (a) stale-capture / (b) real-regression / (c) documented-supersession — @qa-engineer
- [x] **S27-AA-T4**: Landing commits — ONE commit per anchor update, message cites class. Class-(b) escalates to Ramos before landing — @ml-engineer
- [x] **S27-AA-T5**: DEC-031 authored: "Anchor hygiene protocol" — @technical-writer

### Track C — S27-FU-3: DW N=1000 parity-asymmetry triage (blocks on FU-1 landing)

- [x] **S27-FU-3-T1**: Instrument `FindBestSplitPerPartition` on DW N=1000 depth-0: per-partition `(gain_MLX, gain_CPU, chosen_split, gradRms)` across 3 seeds — @qa-engineer — DONE (`0931ad6e9c`; triage doc `docs/sprint27/scratch/fu3-t1-triage.md`)
- [x] **S27-FU-3-T2**: Control: run ST at N=1000 with matched config — @qa-engineer — DONE (T1 absorbed: ST at N=1000 confirmed non-divergent; per-partition gain diff is DW-specific at small N. Result in `fu3-t1-triage.md §ST control`)
- [x] **S27-FU-3-T3**: Verdict doc — @qa-engineer + @ml-product-owner — DONE (verdict: 4th class, fidelity gap / configuration divergence. `score_function='L2'` on CPU reproduces MLX to ±0.11%. Not a/b/c. DEC-032 captures framing. This commit.)
- [x] **S27-FU-3-T4**: Gate adjustment — `score_function='L2'` explicit on DW parity harness CPU side. Do NOT widen N scope — @ml-engineer — DONE (`fc44bfc936`; gate G3-FU3 5/5 PASS at `591f4ce3e6`)
- [x] **S27-FU-3-T5**: DEC-032 authored: verdict + rationale — @technical-writer — DONE (this commit)

### Track D — Quality gates (sequential, end-of-sprint)

- [x] **S27-CR**: Full code review (all FU-1 + AA + FU-3 diffs) — @code-reviewer — APPROVE (`44bb9ee74b`)
- [x] **S27-SA**: Security audit (low-risk — no kernel changes, validation-path data flow only) — @security-auditor — PASS-WITH-NOTES (`24e80dde45`)
- [x] **S27-CLOSE**: Sprint close doc at `docs/sprint27/sprint-close.md` with gate summary, drift inventory, decision links — @technical-writer — DONE (this commit)

---

## Sprint 30 — S30-COSINE-KAHAN — CLOSED 2026-04-24

**Branch**: `mlx/sprint-30-cosine-kahan` (merged via PR #28 → master `17451f4780`; branch deleted).
**Basis**: DEC-034 outcome A (moderate confidence); DEC-035 executed in full.
**Outcome**: Precision fix class exhausted across 13 verdict docs. K4 + Fix 2 shipped as proper (logically correct) fixes that remove precision floors; both ST and LG guards remain in place. DEC-036 opens structural divergence investigation for S31.
**Authoritative record**: `docs/sprint30/sprint-close.md`.

### Primary track — executed

- [x] **#90 S30-T1-INSTRUMENT** — cosDen fingered; residual 4.067e-3 at iter-1 on ST anchor. Verdict: `docs/sprint30/t1-instrument/verdict.md`. — @ml-engineer
- [x] **#91 S30-T2-KAHAN** — Kahan/Neumaier applied; K4 fired → fp64 widening of cosNum/cosDen. 12.5× measurement-layer reduction. Verdict: `docs/sprint30/t2-kahan/verdict.md`. — @ml-engineer
- [x] **#92 S30-T3-MEASURE** — ALL PRIMARY GATES FAIL: G3a 53.3%, G3b 1.27–1.31, G3c 1.44–1.45, G4 PASS, G5 FAIL. Verdict: `docs/sprint30/t3-measure/verdict.md`. — @qa-engineer
- [>] **#93 S30-T4a-ST-REMOVE** — **DEFERRED post-S30**. Mechanism not fixed; guard stays. — @ml-engineer
- [>] **#94 S30-T4b-LG-REMOVE** — **DEFERRED post-S30**. D3 confirmed outcome B mechanism for LG in addition to ST's mechanism. — @ml-engineer

### Diagnostic battery (D1–D4) — executed after T3 failure

- [x] **#100 S30-D1-CPU-COSINE-AUDIT** — CPU is fp64 throughout; `__m128d` hardware commit with `static_assert`. Verdict: `docs/sprint30/d1-cpu-audit/verdict.md`. — @research-scientist
- [x] **#101 S30-D2-FULL-STACK-INSTRUMENT** — Stack instrumentation initially ruled out L3/L4 (0/18 flips). Later invalidated by V2 audit. Verdict: `docs/sprint30/d2-stack-instrument/verdict.md`. — @ml-engineer
- [x] **#102 S30-D3-LG-OUTCOME-AB** — Outcome B confirmed for LG (priority-queue divergence). Flip rate 0.11 → 0.81 across `max_leaves ∈ {8, 16, 31, 64}`, 0/12 trees bit-identical. Verdict: `docs/sprint30/d3-lg-outcome-ab/verdict.md`. — @research-scientist

### Verification battery (V1, V2, V5, V6, D4, D2-redux, Fix 2) — executed after D1–D3

- [x] **#103 S30-V1-DRIFT-VS-N** — L0-scaling FALSIFIED (b = 0.0017 across N ∈ {1k..50k}). Verdict: `docs/sprint30/v1-drift-vs-n/verdict.md`. — @ml-engineer
- [x] **#104 S30-V2-D2-METHOD-AUDIT** — D2 L3/L4 methodology BIASED (gain_f32 and gain_f64 both derived from same fp64). Verdict: `docs/sprint30/v2-d2-audit/verdict.md`. — @research-scientist
- [x] **#105 S30-V5-DW-AT-SCALE** — DW@50k 6.33%, ST@50k 53.30%; 8.4× gap unexplained by L0 alone. Verdict: `docs/sprint30/v5-dw-at-scale/verdict.md`. — @ml-engineer
- [x] **#106 S30-D2-REDUX** — Honest fp32 shadow at `csv_train.cpp:1523-1548`. L3/L4 RULED OUT (5.03e-5 residual, 0/18 flips). Verdict: `docs/sprint30/d2-redux/verdict.md`. — @ml-engineer
- [x] **#107 S30-D4-JOINT-DENOM** — V5 64× amplification hypothesis FALSIFIED (measured 2.42×). Verdict: `docs/sprint30/d4-joint-denom/verdict.md`. Commit `7245099659`. — @ml-engineer
- [x] **#108 S30-FIX2-FP64-GAIN** — Widened `totalGain`, `bestGain`, `TBestSplitProperties::Gain`, `perturbedGain`, `TLeafCandidate::Gain` to `double`. ST drift 53.30% → 53.30% (prediction failed). Verdict: `docs/sprint30/fix2-fp64-gain/verdict.md`. Commits `90a0cb4475` + `364d4ee962`. — @ml-engineer
- [x] **#109 S30-V6-N500-L1-CONFIRMER** — L1 hypothesis FALSIFIED. Drift flat 50.72% @ N=500 → 53.30% @ N=50k (b ≈ 0 across 100× N range). Verdict: `docs/sprint30/v6-n500-confirmer/verdict.md`. Commit `187a5e661f`. — @ml-engineer

### Secondary track — carried forward to S31-T-CLEANUP

- [>] **#95 S30-T5-CLI-EXIT-WRAP** — carried to S31-T-CLEANUP. Owner: @ml-engineer
- [>] **#96 S30-T6-CLEANUP** — carried to S31-T-CLEANUP. Owner: @ml-engineer

### Quality gates (end-of-sprint) — closed

- [x] **#97 S30-CR** — APPROVE (0 must-fix, 5 nits). `docs/sprint30/sprint-close/cr-report.md`. — @code-reviewer
- [x] **#98 S30-SA** — PASS-WITH-FINDINGS (0 CRITICAL/HIGH/MEDIUM, 3 LOW, 3 INFO). `docs/sprint30/sprint-close/sa-report.md`. — @security-auditor
- [x] **#99 S30-CLOSE** — Close doc shipped (`docs/sprint30/sprint-close.md`); DEC-035 PARTIALLY CLOSED; DEC-036 OPEN; DEC-034 partially falsified for ST. PR #28 merged `17451f4780`. — @technical-writer

### Kill-switches (triggered)

- **K1 (T1 mechanism miss)**: not triggered — T1 fingered cosDen correctly.
- **K2 (LG-Stress fail)**: **FIRED** at T3 G3c (1.44–1.45 vs [0.98, 1.02]). T4b deferred to post-S31.
- **K3 (perf regression)**: not measured past T3 failure.
- **K4 (Metal auto-reassociation)**: **FIRED** at T2. Fp64 denominator fallback applied (see DEC-035 closure addendum).

---

## Sprint 32 — COSINE-GAIN-TERM-AUDIT — CLOSED 2026-04-24

**Branch**: `mlx/sprint-32-cosine-gain-term-audit` (tip `3e472ac49f`).
**Outcome**: DEC-038 + DEC-039 shipped. G3a PASS, G3b FAIL (52.6%), G3c PASS, G3d PASS.
**Authoritative record**: `docs/sprint32/sprint-close.md`.

### Primary track — DONE

- [x] **#115 S32-T1-CODEPATH** — SAME-PATH; H1 eliminated. `docs/sprint32/t1-codepath/verdict.md`. Commit `0e24e7f8b7`.
- [x] **#116 S32-T2-INSTRUMENT** — FORMULA CORRECT; root cause = border grid divergence (not formula error). `docs/sprint32/t2-terms/verdict.md`. Commits `5d3899090c`, `1762e8d49c`.
- [x] **#117 S32-T3-FIX** — DEC-038 (allVals) + DEC-039 (fold_count cap 127) shipped. `docs/sprint32/t3-fix/verdict.md`. Commit `901bc760ac`. DEC-012 violation noted (two fixes in one commit).
- [x] **#118 S32-T4-VALIDATE** — Gates run. G3a PASS / G3b FAIL / G3c PASS / G3d PASS. DEC-038/039 formalized. Sprint-close doc produced. State files updated.

### Carry-forward to S33

- [ ] **S31-T4a-ST-REMOVE** — Blocked by G3b FAIL. Deferred to S33 (gated on drift ≤ 2%).
- [ ] **S31-T4b-LG-REMOVE** — Blocked by G3b FAIL. Deferred to S33.
- [ ] **S31-T-LATENT-P11** — hessian-vs-sampleWeight semantics; not blocking. Carry to S33.
- [ ] **S31-T-CLEANUP** — SA-I2 try/catch + S29 CR nits. Carry to S33.
- [ ] **DEC-036 S33 investigation** — L0-L4 scaffold (config audit → determinism → graft → iter=2 instrument → fix). New DEC-040 to be authored at S33 kickoff by orchestrator.

---

## Sprint 31 — ITER1-AUDIT — CLOSING 2026-04-24

**Branch**: `mlx/sprint-31-iter1-audit` (cut from master `17451f4780`, S30 PR #28 merge).
**Basis**: DEC-036 — structural divergence investigation; precision class exhausted in S30.
**Rationale**: V6 ruled out all precision-class fixes via flat N-scaling (b ≈ 0 across 100× N range). ST+Cosine 53% aggregate drift must come from **algorithmic divergence** between CPU and MLX, not fp32 accumulation. T1 (preflight-first) localizes the mechanism: cheap source-diff runs before any instrumentation work.

### Primary track — T1-PRE → T2 port → T3 measure

- [x] **S31-T1-PRE-SOURCEDIFF** — Verdict **(ii) PRE-SPLIT DIVERGENCE — K4 fires**. Formula algebraically equivalent across 11 audited rows (no divergence on numerator / denominator / L2 regularization / parent subtraction / K-dim sum order / sign convention). Pre-split divergence at **P6: feature quantization borders**. MLX's `QuantizeFeatures` (`csv_train.cpp:816–889`) uses a custom percentile-midpoint equal-frequency algorithm; CPU CatBoost default is `GreedyLogSum` (`binarization_options.h:16`). Different candidate-split populations are being argmax'd at every layer. Secondary latent bugs (not firing at S28 anchor): P5 `scaledL2Regularizer` missing, P11 hessian-vs-sampleWeight semantics swap under non-trivial hess losses. Verdict: `docs/sprint31/t1-pre/verdict.md`. Commit `aed81c63d7`. — @research-scientist.

- [x] **S31-T2-PORT-GREEDYLOGSUM** — Port CatBoost `GreedyLogSum` border-selection algorithm into MLX, replacing the custom percentile-midpoint code at `csv_train.cpp:816–889`. Also fixed latent P5 (`ScaleL2Reg`). **Gates: G2a PASS (qualified; 84/100 exact, 16 equal-score tie-breaks); G2b FAIL (53.03% drift vs 53.30% baseline; border divergence NOT the mechanism); G2c PASS (bench_boosting AN-009 preserved); G2d PASS (18/18 L2 parity)**. T1-PRE qualifier fires. Commits `768ee50abd`–`dada4f7b26`. Verdict: `docs/sprint31/t2-port-greedylogsum/verdict.md`. — @ml-engineer.

- [ ] **S31-T3-MEASURE** — Re-run S30 T3 gate matrix (G3a ST, G3b LG-mid, G3c LG-stress) post-T2. **Blocked by T3b (G2b failed; T3b must close ST+Cosine first).** Owner: @qa-engineer.

- [x] **S31-T3b-T1-AUDIT** — G1 PASS. First diverging layer: depth=0. Mechanism: **GAIN-FORMULA** — Cosine score ~5.4% lower in MLX than CPU (ratio 0.946), shifting argmax to wrong bin. Co-fix DEC-037 shipped (border count maxBins, not maxBins-1; greedy unweighted GreedyLogSumBestSplit restored). Build script + run harness + comparison driver + audit JSON + verdict at `docs/sprint31/t3b-audit/`. Commit `746d5090b5`. Owner: @ml-engineer. **Next: S32 cosNum/cosDen per-partition instrumentation at depth=0.**

- [ ] **S31-T4a-ST-REMOVE** — Reopens S30 #93 if S31-T3 G3a passes. Atomic removal across Python + C++ nanobind + CLI + tests. **Blocked by S31-T3 G3a.** Owner: @ml-engineer.

- [ ] **S31-T4b-LG-REMOVE** — Reopens S30 #94 if S31-T3 G3b AND G3c pass. **Blocked by S31-T3 G3b + G3c.** Owner: @ml-engineer.

- [ ] **S31-T-LATENT-P11** — Follow-up (not blocking close): document the hessian-vs-sampleWeight semantics swap at MLX `csv_train.cpp:3780, 3967` as a correctness bug that fires under any loss with non-trivial hessian (Logloss, Poisson, Tweedie, Multiclass). File as a tracked risk under DEC-036 umbrella or an ad-hoc DEC-037 depending on S31 close scope. Owner: @technical-writer.

### Secondary track — carry-forward from S30

- [ ] **S31-T-CLEANUP** — S30 #95 `csv_train:main()` try/catch wrap (SA-I2) + S30 #96 S29 CR residuals (N-1, N-2, N-3, SF-3). Atomic per-commit per DEC-012. Can run in parallel with primary track. Owner: @ml-engineer.

### Quality gates (end-of-sprint)

- [ ] **S31-CR** — Code review of T1-AUDIT instrumentation + T2 fix + T-CLEANUP diffs. Owner: @code-reviewer.
- [ ] **S31-SA** — Security audit; confirm guards remain intact until T4a/T4b land. Owner: @security-auditor.
- [ ] **S31-CLOSE** — Sprint close + DEC-036 status transition + DEC-032 status transition (conditional on guard removal). Owner: @technical-writer.

### Kill-switches (pre-authorized per DEC-036)

- **K1 (no iter=1 divergence)**: CPU and MLX produce bit-identical iter=1 split sequences → expand audit to iter=2 leaf-value + approx-update comparison. **Pre-authorized.**
- **K2 (feature-port gap)**: divergence is a missing MLX feature (e.g., different Cosine variant) → S31 re-plans as port work, not a precision/structural sprint. Escalate to Ramos before re-plan.
- **K3 (seed-independent false positive)**: 0 / 3 seeds diverge at iter=1 → mechanism is not deterministic-structural; revisit precision hypothesis with new evidence (e.g., RandomStrength paths). Escalate to Ramos.
- **K4 (pre-split divergence)**: T1-PRE finds basePred / quantization borders / initial gradients differ between CPU and MLX → S31 re-scopes to a pre-split fix track; T1-AUDIT deferred. **Pre-authorized** (trivial-class fix expected).
- **K5 (cross-cutting fix)**: mechanism located but fix requires changes across histogram kernels + score calcer + node aggregation → budget warning; escalate to Ramos before committing to implementation scope.

---

## Sprint 29 — DEC-032 Closeout + LG Mechanism Spike (CLOSED 2026-04-23)

**Branch**: `mlx/sprint-29-dec032-closeout` (cut from master `987da0e7d5`, S28 merge)
**Tip**: `fa7f9b55fc` (7 commits)
**Authoritative record**: `docs/sprint29/sprint-close.md`

### CLI-GUARD track

- [x] **#82 S29-CLI-GUARD-T1** — Guards ported to `train_api.cpp:TrainConfigToInternal` + `csv_train.cpp:ParseArgs`. — `73e9460a31` — @ml-engineer

- [x] **#83 S29-CLI-GUARD-T2** — 4 pytest cases covering nanobind + CLI guard paths. — `c73f5073af` — @ml-engineer

### LG mechanism spike

- [x] **#84 S29-LG-SPIKE-T1** — Iter-1 drift measured: mean 0.0024% (3 seeds), peak iter-50 0.197%. — `503ebacdb2` — @research-scientist

- [x] **#85 S29-LG-SPIKE-T2** — DEC-034 verdict: outcome A (shared compounding), moderate confidence. — `64a8d9076b` — @research-scientist

### Human checkpoint

- [x] **#86 S29-BRANCH-DECISION** — Ramos decision: close S29, carry Kahan to S30 as S30-COSINE-KAHAN. Outcome A confirmed. — Ramos

### Quality gates

- [x] **#87 S29-CR** — CR PASS-WITH-NITS (0 must-fix, 3 SF, 3 nits). — `3f87b85e38` — @code-reviewer

- [x] **#88 S29-SA** — SA PASS (0 findings, 2 info); SA-H1 CLOSED. — `3f87b85e38` — @security-auditor

- [x] **#89 S29-CLOSE** — Sprint close doc + DEC-032/034/035 updates + state files. — (this commit) — @technical-writer

---

## Sprint 28 — Score Function Fidelity (CLOSED 2026-04-23)

**Branch**: `mlx/sprint-19-hist-writeback` (tip `e0b0b1b527`)
**Rationale**: DEC-032. MLX hardcodes L2 Newton gain; CPU default is Cosine. Fidelity gap discovered S27-FU-3-T1. S28 closes it properly.
**Scope**: Small-sprint, stream A only. Ride-alongs deferred: AN-008 Rule-5 promotion, CR Nit 2, SA Note 2, AA Item H, NewtonL2/NewtonCosine variants.
**Authoritative record**: `docs/sprint28/sprint-close.md`

- [x] **S28-AUDIT**: Grep audit confirming `score_function` is not plumbed anywhere in `catboost/mlx/`. Zero hits confirmed. Hardcoded L2 call site at `csv_train.cpp:~L1281`. — `da02da0259` — @ml-engineer

- [x] **S28-COSINE**: `ComputeCosineGainKDim` helper ported from CPU `TCosineScoreCalcer`. Cosine gain path wired into `FindBestSplitPerPartition`. Gate G2a/G2b PASS: DW N=1000 rs=0 5-seed ratios ∈ [0.98, 1.02]. — `83f30c3677` — @ml-engineer

- [x] **S28-L2-EXPLICIT**: `EScoreFunction` enum + `ParseScoreFunction` added. Dispatch wired into DW/LG paths in `FindBestSplitPerPartition`. Nanobind binding. `_validate_params` rejects `NewtonL2`/`NewtonCosine`. Gate G3a/G3b/G3c PASS. — `0ea86bde21` — @ml-engineer

- [x] **S28-OBLIV-DISPATCH**: Dispatch mirrored into `FindBestSplit` (SymmetricTree). All three grow policies now dispatch on `EScoreFunction`. Gate G7 PASS. — `4083add248` — @ml-engineer

- [x] **S28-REBLESS**: 8 parity cells in `tests/test_python_path_parity.py` relabeled with explicit `score_function`. AN-017 re-captured. Gate G5a–G5d PASS. — `c07e895f7c` — @qa-engineer + @technical-writer

- [x] **S28-FU3-REVALIDATE**: DW force-L2 lifted; DW 5/5 PASS under `score_function='Cosine'` both sides. LG retains force-L2 pending S29-LG-COSINE-RCA. Gate G6a–G6d PASS. — `dca62f0d72` — @qa-engineer

- [x] **S28-{LG,ST}-GUARD**: Two Python `ValueError` guards added for `Cosine+Lossguide` and `Cosine+SymmetricTree`. — `b9577067ef` — @ml-engineer

- [x] **S28-CR**: Code review. PASS-WITH-NITS. CR-S1 resolved; CR-N1/N2/N3 remain nits. — `docs/sprint28/fu-cr/t6-cr-report.md` — @code-reviewer

- [x] **S28-SA**: Security audit. PASS-WITH-FINDINGS. SA-H1 deferred to S29-CLI-GUARD (non-blocking). — `docs/sprint28/fu-sa/t6-sa-report.md` — @security-auditor

- [x] **S28-CR-S1**: Dead `ComputeCosineGain` scalar helper removed. — `e0b0b1b527` — @ml-engineer

- [x] **S28-CLOSE**: Sprint close doc + state files finalized. Parity suite 28/28. — (this commit) — @technical-writer

---

## Sprint 26 — Python-Path Parity — D0 CLOSED

**Branch**: `mlx/sprint-26-python-parity`
**Framing**: correctness-first (option α per Ramos 2026-04-22).
**Pre-sprint triage**: `docs/sprint26/d0/pre-sprint-triage.md`, raw evidence in
`benchmarks/sprint26/d0/RESULTS.md`.
**Verdict**: D0 PASS on all exit gates. DEC-028 + DEC-029 shipped under DEC-012. R8 unchanged at 1.01×. v5 production kernel untouched.

### D0 — Triage + fix + gates (DONE)

- [x] S26-D0-1 — MLX vs CPU leaf-estimation algebra diff with file:line pointers — @ml-engineer **DONE**
- [x] S26-D0-2 — MLX vs CPU RMSE target grad/hessian diff — @ml-engineer **DONE**
- [x] S26-D0-3 — Instrumentation plan (Σgrad, Σhess, l2, leaf@root@depth-0 logging) — @ml-engineer **DONE**
- [x] S26-D0-4 — Root cause identified; DEC-028 opened; fix landed — @ml-engineer **DONE** (DEC-028 RandomStrength noise formula; commit `24162e1006`)
- [x] S26-D0-5 — Python-path 18-config parity sweep; segmented exit gate — @qa-engineer **DONE**
- [x] S26-D0-6 — DEC-028 RandomStrength fix shipped — @ml-engineer **DONE** (`24162e1006`)
- [x] S26-D0-7 — 18-cell G1 sweep + G3 Python-path regression harness + G4 determinism — @qa-engineer **DONE**
- [x] S26-D0-8 — DEC-029 Depthwise/Lossguide SplitProps + BFS index fix (Track A C++ + Track B Python) — @backend-developer **DONE**

### Exit gate verdicts

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| G0 | Root cause(s) in DECISIONS | DEC-028 + DEC-029 complete | PASS |
| G1 | SymmetricTree 18-cell segmented parity | 18/18 (rs=0 max 0.43%; rs=1 MLX ≤ CPU, pred_std_R ∈ [0.9996, 1.087]) | PASS |
| G2 | Depthwise + Lossguide rs=0 parity | DW −0.64%, LG −1.01% (pre-fix 561%/598%) | PASS |
| G3 | Python-path CI regression harness | `tests/test_python_path_parity.py` — 8/8 PASS in 6.32s | PASS |
| G4 | bench_boosting ULP=0 preserved | Kernel sources untouched | PASS |
| G5 | Determinism | 100 runs max−min = 1.49e-08 | DETERMINISTIC |

### Follow-ups

- [>] S26-FU-1 — **Promoted to S27-FU-1-T1..T5** (Track A, Sprint 27 Correctness Closeout).
- [x] S26-FU-2 — MLX Depthwise/Lossguide RandomStrength noise path (`FindBestSplitPerPartition`). **CLOSED 2026-04-22** — DEC-028 formula extended; all gates PASS; branch `mlx/sprint-26-fu2-noise-dwlg`. See `docs/sprint26/fu2/sprint-close.md`.
- [>] S26-FU-3 — **Promoted to S27-FU-3-T1..T5** (Track C, Sprint 27 Correctness Closeout, blocks on FU-1).

### S26-FU-2 task history (DONE 2026-04-22)

- [x] T1 — D0 triage: CPU uses global scalar gradRms for all grow policies — `docs/sprint26/fu2/d0-triage.md`
- [x] T2 — Thread `gradRms` into `FindBestSplitPerPartition` (C++ impl) — `478e8d5c9d`
- [x] T3 — Manual smoke test DW + LG with rs=1 — bundled `478e8d5c9d`
- [x] T4 — Extend `test_python_path_parity.py` to DW/LG — `715b15b613`
- [x] T5 — G1 54-cell sweep (DW + LG + ST) — `ee5a90707b`
- [x] T6 — G5 Depthwise determinism 100-run — `ee5a90707b`
- [x] T7 — Code review (@code-reviewer): APPROVE-WITH-NITS, 0 blockers
- [x] T8 — Sprint close: gate report Nit-1 fix, DEC-028 footnote, state files, close report

---

---

## Sprint 21 — A1 Measurement Sprint (CLOSED — 6/6 exit gates PASS, 0× perf shipped)

**Branch**: `mlx/sprint-21-hist-tg-reduction`
**Campaign**: Operation Verstappen — battle 6 of 9 — A1 measurement sprint (pivot from TG-count reduction after D0 kill-switch fired)
**Verdict**: All 6 A1 exit gates PASS. Zero production source modified (A1-G6 discipline). Two levers retired, one promoted to viable-set.

### Closed

- [x] S21-D0 — D0 kill-switch: fixed per-TG overhead = 2.5% ± 1.3% at depth 6 (R²=0.9989). Kill-switch threshold 10% NOT met → FIRED. Variant A never activated. DEC-018 RETIRED. — @performance-engineer **DONE** (`a0c473e3b7`)
- [x] S21-D1-R3 — Host-side `eval()` sync instrumentation in `bench_boosting` (`--per-kernel-profile`). Per-dispatch stdev < 5% of mean at gate config. — @ml-engineer **DONE** (`ac378d8de6`)
- [x] S21-D1-R1 — L2 stats pre-permute direct mechanism test. Zero-gather upper bound (`stat = 1.0f`) at 1664-TG production shape: +2.61% slower (12.6 pp below 10% gate). **FALSIFIED.** DEC-019. — @ml-engineer + @performance-engineer **DONE** (`fedf9d5348`)
- [x] S21-D1-R2 — T2 sort-by-bin production-shape micro-bench. Sort+accum at 1664-TG shape: −64.8% (band 63.6–66.7%), clearing 50% gate by 28–34 pp. Parity gate B: max ULP 64, mass conservation 0 ULP. **VIABLE.** DEC-020. — @ml-engineer + @performance-engineer **DONE** (`13322feaca`)
- [x] S21-D1-R4 — Sprint 22 kickoff synthesis. Lever ranking with mechanism-direct gates; D0 kill-switch spec at ratio > 0.60; R8 honest ledger. — @technical-writer **DONE** (`a7a206b90d`)

---

## Sprint 22 — T2 Sort-by-Bin Integration (CLOSED — 4/4 gates PASS, R8 1.90×)

**Branch**: `mlx/sprint-22-t2-integration`
**Campaign**: Operation Verstappen — battle 7 of 9 — single-lever integration sprint
**Gate config**: 50k/RMSE/d6/128b
**Lever**: T2 sort-by-bin (DEC-020)
**Verdict**: **CLOSED. Verstappen ≥1.5× gate CLEARED. Cumulative R8 = 1.90×. PR #14 pending (Ramos opens).**

### Closed

- [x] S22-D0 — In-situ T2 integration probe. `hist_ms(T2) / hist_ms(T1)` = 0.328× cross-session at gate config (inside optimistic band). Kill-switch (>0.60) NOT triggered. — @ml-engineer **DONE** (`4333c82a7e`)
- [x] S22-D1 — 18-config DEC-008 parity sweep. **FAILED 18/18** at D0 T2 kernel (ULP 1,327–2,583,206). Triggered four-phase diagnostic arc (D1a→D1b→D1c). Root cause: `maxPartDocs = ceil(N/K)` uniform-partition assumption overflows under real argsort-permuted skewed splits. — @qa-engineer / @ml-engineer **DONE** (`73baadf445` contains full arc)
- [x] S22-D1a — Blit-ordering hypothesis (fill_gpu pool reuse). **REFUTED**: fill_gpu is compute, not blit; eval barriers did not fix parity. — @ml-engineer **DONE** (bundled in `73baadf445`)
- [x] S22-D1b — Depth-parity indexing hypothesis. **REFUTED**: even-depth failure pattern explained by split-distribution artifact, not indexing. — @ml-engineer **DONE** (bundled in `73baadf445`)
- [x] S22-D1c — Root-cause: `bench_boosting.cpp:526` `maxPartDocs = ceil(numDocs / numActiveParts)` overflow. Depth-1 on 50k: sizes [442, 49558] vs maxPartDocs=25000; 24558-doc overflow into neighbor slot. — @ml-engineer **DONE** (`docs/sprint22/d1c_t2_troubleshoot.md`, bundled in `73baadf445`)
- [x] S22-D2 — Option III slab-by-partOffsets fix. `sortedDocs` reorganized to per-(groupIdx, statIdx) slabs of size `numDocs` indexed by `partOffsets[partIdx]`. Buffer 5.2 MB at gate config (vs 333 MB worst-case for Option I). Overflow structurally impossible. — @ml-engineer **DONE** (`73baadf445`)
- [x] S22-D3 — Parity exit gate (independent QA). 18/18 ULP=0 bit-exact; 100/100 determinism; edge cases EC-1–EC-5 all ULP=0. Bug β (atomic-scatter float drift) does not exist — retires S21 D1-R4 Kahan concern (DEC-022). — @qa-engineer **GATE PASS** (`docs/sprint22/d3_parity_gate.md`, bundled in `73baadf445`)
- [x] S22-D4 — Perf exit gate. Ratio 0.317× cross-session (band 0.315–0.319×); S22 multiplier 1.778×; cumulative R8 = **1.07 × 1.778 = 1.90×**; Verstappen gate cleared by 40 pp. N=1000 ratio exceedances (0.651–0.694×) are structural amortization artifact, non-blocking (gate anchored at 50k). — @performance-engineer **GATE PASS** (`docs/sprint22/d4_perf_gate.md`, bundled in `73baadf445`)
- [x] S22-D5 — Code review exit gate. 0 blockers. 7 nits; NIT-6 (stray blank line) removed by security auditor; 6 nits deferred to Sprint 23. — @code-reviewer **GATE PASS** (`docs/sprint22/d5_code_review.md`, bundled in `73baadf445`)
- [x] S22-D6 — Security audit exit gate. 0 CRITICAL, 0 HIGH; bounds-proof confirms D1c overflow class structurally eliminated; max-safe-N 14.3M docs (286× headroom over 50k); zero secrets/PII. 2 MEDIUM advisory (DoS/robustness, bench harness scope). — @security-auditor **GATE PASS** (`docs/sprint22/d6_security_audit.md`, bundled in `73baadf445`)

---

## Sprint 23 — T2 Scratch→Production Promotion + NIT Catalog + Tree-Search Research (CLOSED)

**Branch**: `mlx/sprint-23-t2-promotion`
**Campaign**: Operation Verstappen — battle 8 of 9 — **CLOSED 2026-04-21**
**Gate config**: 50k/RMSE/d6/128b
**Verdict**: PASS (pre-existing-bug footnote). D0 4/4 gates (G1 with errata). R1 DEFERRED. R2 FALSIFIED. R8 1.90× unchanged. PR #15 pending.

### D0 — T2 scratch→production promotion (blocking)

- [x] S23-D0 — Promote `kernel_sources_t2_scratch.h` contents into `catboost/mlx/kernels/kernel_sources.h` under `kT2SortSource` and `kT2AccumSource` sections. Move `DispatchHistogramT2` from `bench_boosting.cpp` into `catboost/mlx/methods/histogram.cpp` with production-quality API (CB_ENSURE error handling, factored kernel registration, clean public interface). Remove `CATBOOST_MLX_HISTOGRAM_T2` compile-time flag and make T2 the default dispatch path. Per DEC-012: 4 atomic commits landed (tip `84529b47ed`). — @ml-engineer **DONE — KILL-SWITCH TRIPPED**

  **Verdict**: PASS (kill-switch tripped on pre-existing bug; records corrected; proceed to R1/R2). Parity sweep: 17/18 ULP=0; config #8 (N=10000/RMSE/128b) BIMODAL ~50/50 at (0.48231599, 0.48231912) — 105 ULP gap. Pre-existing in S22 D2/D3 tip `73baadf445`; not introduced by promotion. Gate config #14 (N=50000/RMSE/128b) 100/100 deterministic at 0.47740927. R8 1.90× record unaffected. See `docs/sprint23/d0_bimodality_verification.md` and DEC-023.

### NIT cleanup batch (from D5 code review, deferred to S23)

- [ ] S23-NIT1 — Replace inline literals (`256u`, `128u`, `0x7Fu`, `4u`) in T2 kernels with named constants from `kHistHeader` (`BLOCK_SIZE`, `BINS_PER_BYTE`, `FEATURES_PER_PACK`). File: `kernel_sources_t2_scratch.h` (or promoted `kernel_sources.h`). — @ml-engineer
- [ ] S23-NIT2 — Pull duplicate `offBase` arithmetic + `129u` magic number into a named constant (`BIN_OFFSETS_STRIDE = 129u`) in `kHistHeader` or a new `kT2Header`; add clarifying comment (`128 bins + 1 total`). — @ml-engineer
- [ ] S23-NIT3 — T2-accum empty-partition hardening: add explicit `if (partSize == 0) return;` at start of T2-accum mirroring T2-sort, eliminating reliance on float-to-uint zero-bit-aliasing from `init_value=0.0f`. — @ml-engineer
- [ ] S23-NIT4 — Add host-side CB_ENSURE that `maxBlocksPerPart == 1` when T2 is active; document the 1-block-per-partition constraint in the T2-sort kernel header comment. — @ml-engineer
- [ ] S23-NIT5 — Remove unused `numTGs` uniform from both T2-sort and T2-accum `input_names` and from host-side `numTGsArr`; remove mention from kernel header comments. — @ml-engineer
- [ ] S23-NIT7 — Harmonize feature 1-3 bin mask: change T2-accum feat 1-3 `& 0xFFu` to `& 0x7Fu` (matching T2-sort feat-0 and DEC-016 envelope), or use `& 0xFFu` for all features matching T1. Bundle with NIT1 in one pass. — @ml-engineer

### Tree-search research track (rank #2 post-T2, from d1r4_synthesis.md §3)

- [x] S23-R1 — EvalAtBoundary readback elimination (S19-11 carry-forward). **VERDICT: NOT VIABLE — architectural mismatch.** 3 live sites in `structure_searcher.cpp` (lines 290, 609, 705) are on Depthwise/Lossguide paths only. gate config (oblivious tree) never enters these paths; bench_boosting has its own inline loop that bypasses structure_searcher.cpp entirely. 0/3 sites replaced. Gate perf delta: 0 ms (sites unreachable from gate path). Parity: unchanged (17/18 ULP=0, config #8 bimodal unchanged). R8: 1.90× unchanged. Per-site: A=SKIP (depthwise restructure, no gate perf), B=SKIP (lossguide, parity harness gap), C=SKIP (scope exceeds budget, lossguide restructure). Forward: Depthwise/Lossguide benchmarks required before these sites can be targeted. See `docs/sprint23/r1_evalatboundary.md`. — @ml-engineer **DONE (no-op — kill-switch fired)**
- [x] S23-R2 — Dispatch inversion research spike. **FALSIFIED — structural algebraic blocker.** `H[f][b] = Σ_p h_p[f][b]` is not invertible; all five mask mechanisms (A–E) are algebraically or empirically rejected. Atomic contention 64× worse than DEC-023 trigger. Mechanism E is DEC-017 T3b with a different label (same +42.3% regression predicted). Day-1 kill-switch invoked; Day 2 not exercised. Do not re-enter without new mask-mechanism evidence. DEC-025 FALSIFIED. See `docs/sprint23/r2_dispatch_inversion_spike.md`. — @research-scientist **DONE (no-op — kill-switch fired)**

### Carry-forward

- [ ] S19-11 — See S23-R1 above (same task, merged).

---

---

## Sprint 24 — DEC-023 Atomic-Float Race Fix (CLOSED)

**Branch**: `mlx/sprint-24-dec023-fix`
**Campaign**: Operation Verstappen — battle 9 of 9 — **CLOSED 2026-04-21**
**Verdict**: D0 PASS on parity (DEC-023 RESOLVED). FAIL on R8 preservation (retroactive retreat 1.90× → 1.01×). Verstappen ≥1.5× gate failed. PR #16 pending.

### D0 — DEC-023 fix (DONE)

- [x] S24-D0 — v5: all-feature T1-style SIMD-shuffle accumulation. All four features (0-3) in T2-accum read from `docIndices` via T1's SIMD-shuffle + linear fold. T2-sort removed from dispatch. ULP=0 structural. All 4 acceptance criteria PASS. Commit `784f82a891`. R8 consequence: 1.90× → 1.01×. — @ml-engineer **DONE**

- [x] S24-SO-1 — Parity sweep protocol documented in `docs/sprint24/README.md §5`. Standing order effective S23 D0 forward: ≥5 runs per non-gate; 100 runs at gate. — @technical-writer **DONE**

---

## Sprint 25 — DEC-026 Cascade-Robust GAIN Research (CLOSED — FALSIFIED)

**Branch**: `mlx/sprint-25-dec026-cascade` (stacked on S24)
**Campaign**: Post-Verstappen research — R8 recovery investigation
**R8 entry position**: 1.01× (honest post-S24)
**R8 exit position**: 1.01× (unchanged — research FALSIFIED at G1)
**Verdict**: FALSIFIED at G1. Kill-switch fired cleanly on day 1. G2–G5 not attempted.
**Authority**: `DECISIONS.md DEC-026` (FALSIFIED 2026-04-21); `docs/sprint25/g1_epsilon_calibration.md` (verdict doc)

### Research track — DEC-026 (owner: @research-scientist)

- [x] S25-G1 — Epsilon calibration study. **FALSIFIED** 2026-04-21. 180-run empirical sweep (18 configs × 5 runs × 2 kernels) established ε_min = 2.200e-03 vs ε_max⁺ = 1.043e-07 → safety ratio 4.74e-05 (target ≥ 2.0). Path 5 flip gaps span the full range of legitimate top-2 separations; no ε threads the needle. Kill-switch fired. — @research-scientist **DONE (FALSIFIED)**

- [x] S25-G2 — Tiebreak implementation in scoring kernel. **CANCELLED** — blocked on G1, never attempted. — @ml-engineer **DONE (CANCELLED)**

- [x] S25-G3 — T2 Path 5 rebuild. **CANCELLED** — blocked on G2. — @ml-engineer **DONE (CANCELLED)**

- [x] S25-G4 — 18-config parity sweep + determinism. **CANCELLED** — blocked on G3. — @qa-engineer **DONE (CANCELLED)**

- [x] S25-G5 — Model-quality validation. **CANCELLED** — blocked on G4. — @qa-engineer **DONE (CANCELLED)**

### Sprint 25 merge gate

FALSIFIED. R8 stays at 1.01×. Verstappen ≥1.5× gate remains retroactively failed from S24 D0.
v5 (`catboost/mlx/kernels/kernel_sources.h` at `784f82a891`) is the final production kernel.
S25 closeout ships the G1 empirical artifacts + verdict doc as falsification evidence; no
production code changes.

**DEC-027 (deferred)**: alternative accumulation paths (e.g., XGBoost-style per-feature
deterministic radix-sum) are acknowledged as a possible future research direction but are
NOT opened at S25 close. Ramos to dedicate time later.

---

## Sprint 20–25 Backlog (one-line each, expanded per sprint)

- Sprint 20: T3b atomic-CAS — CLOSED, FALSIFIED (DEC-017 RETIRED). PR #12 merged.
- Sprint 21: A1 measurement — CLOSED, 0× perf, T2 promoted to viable-set. PR #13 merged.
- Sprint 22: T2 sort-by-bin integration — CLOSED, R8 1.90× (superseded by S24). Verstappen gate cleared at S22. PR #14 merged. Note: S22 D3 parity corrected to 17/18 (DEC-020 footnote + DEC-023).
- Sprint 23: T2 scratch→production promotion — CLOSED. 8 commits. D0 PASS (pre-existing bug). R1 DEFERRED. R2 FALSIFIED. DEC-023/024/025. PR #15 merged.
- Sprint 24: DEC-023 v5 fix — CLOSED. DEC-023 RESOLVED. R8 1.90× → 1.01× retroactive. Verstappen ≥1.5× failed. DEC-026 opened. PR #16 **merged 2026-04-22** (`1385e056ca`).
- Sprint 25: DEC-026 cascade-robust GAIN research — CLOSED, FALSIFIED at G1 (safety ratio 4.74e-05 vs 2.0 target). R8 stays at 1.01×. DEC-027 deferred for future dedicated research. PR #17 **merged 2026-04-22** (`5caa6e64cf`).
- CI unblock (PR #18): MLX 0.31+ CLI fix + stale version-pin cleanup + BUG-001 MAE sentinel narrowing. **Merged 2026-04-22** (`9b0c03fec2`). No production code changes.

---

## Carry-forward from pre-Sprint 16 — ALL CLEARED 2026-04-22 (PR #20, `71aabaa842`)

- **BUG-007** (MITIGATED 2026-04-22): Python wrapper pre-sorts group_ids (`core.py:1131-1137`); C++ `BuildDatasetFromArrays` now CB_ENSUREs sortedness. See `KNOWN_BUGS.md`.
- **bench_boosting K=10 anchor mismatch** (RESOLVED Sprint 8, TODO-022): `1.78561831` is canonical; prior `2.22267818` was a stale mismatched-params capture. See `CHANGELOG.md:27`.
- **Sibling S-1** (latent, guard hardened): `kHistOneByte` writeback race now guarded by `constexpr` + `static_assert(maxBlocksPerPart == 1, ...)` at `histogram.cpp:126`. Fix only if multi-block dispatch is ever needed.

**DEC-027 (deferred, unchanged)**: not on this backlog. Reserved for a dedicated future research sprint per S25 closeout.
