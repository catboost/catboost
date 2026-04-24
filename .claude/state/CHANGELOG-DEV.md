# Developer Changelog — CatBoost-MLX

> Coverage: Sprints 0–15 reconstructed from git log on 2026-04-15. Sprint 16+ is source of truth.

## 2026-04-24 — Sprint 32 CLOSED (DEC-038/DEC-039 shipped; DEC-036 reframed for S33)

Branch: `mlx/sprint-32-cosine-gain-term-audit`, base S31 tip `9b3a5238a7`, tip `3e472ac49f`.
7 commits on branch. No kernel sources touched (v5 `784f82a891` unchanged).

### What shipped

Three correctness fixes in `catboost/mlx/tests/csv_train.cpp`:

**DEC-038** (`901bc760ac`): `GreedyLogSumBestSplit` was receiving deduplicated values instead
of all-docs array. CPU's `TFeatureBin` uses `features.Values` (N docs with duplicates);
`BinEnd - BinStart = document count`. Using unique values changed the score landscape,
causing ~2-index border grid offset and the 5.4% Cosine gain deficit. Fix: pass `allVals`.

**DEC-039** (`901bc760ac`): Histogram kernel T2_BIN_CAP contract violated at `fold_count=128`.
`bin_value=128` at `posInWord=0` features sets bit 31 (= VALID_BIT), which is stripped by
`p_clean = p_s & 0x7FFFFFFF`, aliasing 391 docs to bin_value=0 and dropping them from the
histogram. Fix: `maxBordersCount = std::min(maxBins, 127u)`. Contract was already documented
in `kernel_sources.h:38` ("Safe ONLY when every feature's fold count <= 127").

**DEC-037** (S31 co-fix, `746d5090b5`): border count off-by-one + DP algorithm wrong; closed S31.

### Gate results

| Gate | Result |
|------|--------|
| G3a — depth=0 gain ratio (3 seeds) | PASS: ratios 1.000000/1.000000/1.000000 (delta < 5e-7) |
| G3b — iter=50 ST+Cosine drift | FAIL: 52.6% (DEC-036 reframed; target was ≤2%) |
| G3c — bench_boosting v5 ULP=0 | PASS: 0.48231599 = AN-009 anchor; kernel md5 byte-identical |
| G3d — 18-config L2 non-regression | PASS: 18/18 cells in [0.98, 1.02] envelope |

### DEC-036 reframe

The "GAIN-FORMULA" framing (T3b ratio 0.946) was a surface symptom of the border bugs.
With borders fixed, depth=0 gain ratio is 1.000000. But iter=50 drift is 52.6% (unchanged
from 53.30% pre-fix). The 0.75% iter=1 residual compounds to 52.6% at iter=50 at ~9%/iter
— ~70x amplification (geometric 1.0075^50 = 1.45; observed 1.526). Runaway divergence.
S33 will audit this with L0-L4 scaffold: config → determinism → graft experiment → iter=2
instrumentation → fix. DEC-040 opens at S33 kickoff.

### DEC-012 atomicity violations (2 this sprint)

1. DEC-037 bundled with T3b verdict doc in `746d5090b5` (S31).
2. DEC-038 + DEC-039 bundled in `901bc760ac` (S32 T3).

S33 hard rule: "if you find a second structural issue while fixing the first, STOP and
commit the first atomically before continuing the investigation."

### Key non-obvious finding

The 52.6% drift at iter=50 from a 0.75% iter=1 residual implies 70x amplification — not
geometric compounding. This is trajectory lock-in, not a precision floor. The DW/ST gap
(6.33% vs 52.6%) remains unexplained and may be diagnostic.

---

## 2026-04-24 — Sprint 30 CLOSING (precision fix class exhausted; DEC-036 opens structural investigation)

Branch: `mlx/sprint-30-cosine-kahan`, base `4d855d47db`, tip `187a5e661f`. S30 executed full
phased T1→T4 plan per DEC-035 plus an extensive verification battery (D1/D2/D2-redux/D3/D4/
V1/V2/V5/V6/Fix 2) after T3 failed all primary envelope gates. No kernel sources touched
(v5 `784f82a891` unchanged). Two scalar-type widenings shipped outside the kernel. Both ST
and LG guards remain in place.

### Executed phases and verdicts

| Phase | Task | Gate | Result |
|-------|------|------|--------|
| T1 | #90 INSTRUMENT | G1 mechanism fingered | PASS (cosDen, residual 4.067e-3) |
| T2 | #91 KAHAN | G2 ≥10× residual reduction | PASS (12.5×); K4 fired → fp64 widening |
| T3 | #92 MEASURE | G3a/G3b/G3c | **FAIL** (53.30% ST; K2 fired at G3c 1.44–1.45) |
| D1 | #100 CPU AUDIT | CPU precision baseline | CPU is fp64 throughout (`__m128d`, static_assert) |
| D2 | #101 FULL-STACK | Locate binding layer | Initially ruled out L3/L4; V2 later invalidated the methodology |
| D2-redux | #106 METHOD FIX | Honest fp32 shadow | L3/L4 RULED OUT (5.03e-5 residual, 0/18 flips) |
| D3 | #102 LG OUTCOME A/B | Discriminate LG path | Outcome B confirmed for LG (priority-queue divergence) |
| D4 | #107 JOINT-DENOM 64× | V5 amplification hypothesis | FALSIFIED (measured 2.42×, not 64×) |
| V1 | #103 N-SCALING | L0 precision-class predictor | FLAT — exponent b = 0.0017 |
| V5 | #105 DW @ 50k | Isolate ST-specific mechanism | MIXED — L0 real but 8.4× DW/ST gap unexplained |
| V6 | #109 N=500 CONFIRMER | Cheap L1 falsification | **L1 FALSIFIED** — drift 50.72% @ N=500 vs 53.30% @ N=50k (b ≈ 0) |
| Fix 2 | #108 FP64 GAIN | L3/L4 widening | SHIPPED; ST drift 53.30% → 53.30% (prediction failed cleanly) |

### Commits (oldest → newest, S30 branch tip `187a5e661f`)

| SHA | Tag | Purpose |
|-----|-----|---------|
| (S30-00 kickoff, state files) | S30-00 | Branch kickoff; DEC-035 ultrathink elaboration |
| `108c7a59d2`-family | S30-T1/T2/K4 | cosNum/cosDen accumulator instrumented + fp64 widened |
| (T3 verdict) | S30-T3 | 18-config measurement: all primary envelope gates fail |
| (D1 verdict) | S30-D1 | CPU precision audit — CPU is fp64 throughout |
| (D2 verdict + D3 verdict) | S30-D2/D3 | Stack instrumentation; LG outcome B confirmed |
| `2d03cf8702` | S30-D2-REDUX | Corrected fp32 shadow methodology at `csv_train.cpp:1523-1548`; L3/L4 RULED OUT honestly |
| `7245099659` | S30-D4 | V5 64× amplification hypothesis FALSIFIED (measured 2.42×) |
| (V1 + V5 verdicts) | S30-V1/V5 | N-scaling + DW-at-scale falsification |
| (V2 verdict) | S30-V2 | D2 methodology audit — L3/L4 residual was cast ULP only |
| `90a0cb4475` | S30-FIX2 | Fp64 widening of totalGain/bestGain/TBestSplitProperties::Gain/perturbedGain/TLeafCandidate::Gain |
| `364d4ee962` | S30-FIX2-VERDICT | Fix 2 verdict: 53.30% → 53.30% (prediction failed cleanly) |
| `187a5e661f` | S30-V6 | N=500 L1 confirmer — hypothesis FALSIFIED; b ≈ 0 across 100× N range |

### What ships from S30

- **K4 (fp64 cosNum/cosDen)** — logically correct precision fix; removes a floor that would otherwise re-surface after the structural mechanism is resolved.
- **Fix 2 (fp64 gain/argmax)** — logically correct precision fix; same rationale as K4.
- **13 verdict docs** under `docs/sprint30/` — full chain of evidence for precision-class exhaustion.
- **Instrumentation** behind `COSINE_RESIDUAL_INSTRUMENT` in `catboost/mlx/tests/csv_train.cpp` — retained for S31 audit reuse.

### What does NOT ship

- T4a/T4b guard removal (#93/#94) — deferred; mechanism not fixed.
- K4 and Fix 2 are correct but **insufficient** alone. Guards remain at Python `_validate_params`, `train_api.cpp:TrainConfigToInternal`, and `csv_train.cpp:ParseArgs`.

### DEC transitions

- **DEC-035**: ACTIVE → **PARTIALLY CLOSED**. Precision fix class exhausted; atomicity clause and phased-plan rationale preserved for audit trail.
- **DEC-034**: RESOLVED (outcome A) → **PARTIALLY FALSIFIED for ST** (V6 N-scaling rules out pure precision mechanism); **LG outcome B confirmed dominant** for LG (D3 verdict).
- **DEC-032**: PARTIALLY CLOSED — unchanged. Both `{ST,LG}+Cosine` guards still in place.
- **DEC-036**: OPEN — structural divergence investigation; S31 T1 is the iter=1 algorithmic audit. See DECISIONS.md.

### S31 kickoff

Spawn @ml-engineer (or @research-scientist) on **S31-T1-ITER1-AUDIT**. Build iter=1 split-selection comparison harness dumping `(feature_idx, bin_idx, gain)` per layer from CPU CatBoost and MLX. First diverging layer names the mechanism class. Deliverable: `docs/sprint31/t1-audit/verdict.md`.

### Lessons captured

1. **Precision-layer hypothesis pattern**: we hypothesized four different precision mechanisms in sequence (cosDen, L3/L4 gain cast, joint-denominator 64× amplification, L0 histogram N-scaling). All four were measurably correct at the measurement layer but failed to move the trajectory layer. Flat N-scaling (b ≈ 0) is the cheap oracle that falsifies the whole class at once.
2. **Measurement-layer gates can mask trajectory-layer failures**: G2 passed at 12.5× residual reduction but G3a did not move. Gate specs must test the lever's actual mechanism against the target outcome, not a measurement-layer proxy — this is the DEC-028 "kernel-ULP=0 ≠ full-path parity" trap in a different costume.
3. **Verification audits must audit their own methodology**: V2 discovered D2's L3/L4 rulings were biased (`gain_f32` and `gain_f64` were both derived from the same `double`). Always verify that the measurement actually measures the claimed quantity.
4. **Two falsified predictions in a row (D4 L3/L4, V6 L1) means stop guessing precision and measure structure directly** — the S31 iter=1 audit.

## 2026-04-24 — Sprint 31 T2 COMPLETE (G2b FAIL — border divergence ruled out; T3b T1-AUDIT active)

Branch: `mlx/sprint-31-iter1-audit` (continuing). Tip `dada4f7b26`. 5 commits this session
(`768ee50abd..dada4f7b26`). T2-PORT-GREEDYLOGSUM fully executed and closed. T3b T1-AUDIT
fallback activated.

### Gate results

| Gate | Criterion | Result |
|------|-----------|--------|
| G2a | Borders byte-match CPU CatBoost GreedyLogSum | PASS (qualified) — 84/100 exact; 16 equal-score tie-breaks |
| G2b | ST+Cosine aggregate drift ≤ 2% at S28 anchor | **FAIL** — 53.03% (baseline 53.30%; delta 0.27pp = noise) |
| G2c | bench_boosting v5 ULP=0 preserved | PASS — AN-009 anchor `0.48231599`; kernel sources diff = 0 |
| G2d | 18-config L2 parity non-regression | PASS — 18/18 cells in acceptance envelope |

### Commits (this session)

| SHA | Description |
|-----|-------------|
| `768ee50abd` | T2 port GreedyLogSum into QuantizeFeatures |
| `627b968983` | T2 G2a probe — borders byte-match infrastructure |
| `bfb20d3241` | T2 P5 fix — ScaleL2Reg at all three split/leaf sites |
| `661ef0bc2c` | T2 gate probes — G2b ST+Cosine drift + G2d L2 parity |
| `dada4f7b26` | T2 verdict — G2b FAIL; T3b T1-AUDIT fallback triggered |

### What ships

- **GreedyLogSum border-selection port** — `csv_train.cpp:816–889` now uses CPU-equivalent
  algorithm. G2a qualified-pass: algorithm is correct; 16/100 tie-breaks differ from pip
  catboost v1.2.10 but both are valid GreedyLogSum outputs.
- **P5 ScaleL2Reg fix** — `scaledL2 = L2RegLambda * sumAllWeights / docCount` wired at
  all three call sites (Lossguide, Depthwise, SymmetricTree FindBest* + l2Arr for leaf Newton).
  No-op at S28 anchor (uniform weights); load-bearing for non-uniform sample weights.
- **G2a, G2b, G2d probe harnesses** — `docs/sprint31/t2-port-greedylogsum/`.
- **Verdict doc** — `docs/sprint31/t2-port-greedylogsum/verdict.md`.

### What does NOT close

- **ST+Cosine 53% drift** — unchanged. Border divergence ruled out as root cause. The only
  remaining diagnostic path is S31-T3b: instrumented iter=1 side-by-side dump.
- **SA-H1 Cosine guards** — remain active at all three layers.
- **T3, T4a, T4b** — all still blocked on T3b.

### Lessons captured

1. **The T1-PRE qualifier pattern works**: pre-announcing that a fix may not close the gap
   prevents scope creep when the gate fails. The S26-D0 P10 probe (0.06% from borders at
   L2+RS=0+N=10k) was the correct low-confidence signal that borders were unlikely to be
   the mechanism at Cosine+ST.

2. **G2b FAIL interpretation**: with 84/100 border byte-matches and G2b only improving
   0.27pp, the 53% drift does not originate from quantization. Consistent with V6's flat
   N-scaling (b ≈ 0) — a structural algorithmic difference, not a data-dependent precision floor.

### Next step

**S31-T3b-T1-AUDIT** is now ACTIVE. Owner: @ml-engineer. Instrumented iter=1 dump using
`COSINE_RESIDUAL_INSTRUMENT` infrastructure already in `csv_train.cpp`. Compare CPU vs MLX
on parent stats, top-K=5 split candidates, and winning split tuple at every depth level
of iteration 1. First diverging layer names the mechanism class.

---

## 2026-04-24 — Sprint 31 T3b-T1-AUDIT COMPLETE + DEC-037 border fix

Branch: `mlx/sprint-31-iter1-audit`, base `17451f4780`, commit `746d5090b5`.

### What happened

S31-T3b-T1-AUDIT built the full instrumented iter=1 comparison pipeline:

1. **`build_mlx_audit.sh`** — compiles `csv_train.cpp` with `-DITER1_AUDIT -DCOSINE_T3_MEASURE`
2. **`run_mlx_audit.py`** — runs MLX binary on canonical S26 data (N=50k, seeds 42/43/44),
   writes per-layer JSON: parent stats + top-K=5 + winner
3. **`compare_splits.py`** — diffs CPU (pip catboost) vs MLX JSONs, classifies first divergence
   per DEC-036 mechanism table

### DEC-037 co-fix (shipped)

Root cause: `QuantizeFeatures` was calling `GreedyLogSumBestSplit` with
`maxBordersCount = maxBins - 1 = 127`, while CPU CatBoost uses `border_count = 128`.

Investigation path:
1. Initial fix attempt: changed `maxBins - 1` → `maxBins` but also rewrote to a DP
   (document-count weighted), which was incorrect. CatBoost's `TGreedyBinarizer` uses the
   **unweighted** `TFeatureBin` path (count of unique values, not documents).
2. Reverted to the original greedy priority-queue approach (correct), with only the
   border count changed to `maxBins`.

**Files changed**: `catboost/mlx/tests/csv_train.cpp` — `GreedyLogSumBestSplit` restored,
`maxBordersCount = maxBins`.

### G1 PASS verdict

| Criterion | Result |
|-----------|--------|
| First diverging layer | depth=0 (seeds 42, 44); depth=2 (seed 43) |
| Mechanism class | **GAIN-FORMULA** |
| Gain ratio (MLX/CPU) | ~0.946 (5.4% deficit, consistent all seeds/depths) |
| DEC-037 border fix | Shipped — 128 borders now match CPU |

The Cosine gain formula in `FindBestSplit` produces values ~5.4% lower than CPU's
`CosineScoreCalcer`. This shifts the argmax to a different bin. The partition stats
(sumH) match at depth=0 confirming the histogram inputs are identical — only the
score computation is wrong.

Verdict doc: `docs/sprint31/t3b-audit/verdict.md`.

### What ships

- **DEC-037 border count fix** — `maxBordersCount = maxBins` (was `maxBins - 1`)
- **T3b-T1-AUDIT pipeline** — build script, run harness, compare driver, audit data
- **G1 PASS** — GAIN-FORMULA mechanism class named at depth=0

### What does NOT close

- **ST+Cosine 53% drift** — unchanged. GAIN-FORMULA identified but not yet fixed.
- **SA-H1 Cosine guards** — remain active.
- **T3, T4a, T4b** — still blocked.

### Next step (S32)

Instrument `FindBestSplit` to dump `cosNum`, `cosDen`, `wL`, `wR`, `gL`, `gR`
per partition per bin at depth=0 for seed=42. Compare term-by-term against CPU's
`CosineScoreCalcer`. Identify the exact expression causing the 5.4% deficit.

---

## 2026-04-23 — Sprint 29 CLOSED (DEC-032 closeout partial advance; DEC-034 resolved)

Branch: `mlx/sprint-29-dec032-closeout`, base `987da0e7d5`, tip `fa7f9b55fc`. 7 commits
(`33ce5f1d66..fa7f9b55fc`). All gate criteria met. Parity suite 28/28 at all commits.
No kernel sources touched.

### Commits

| Commit | Tag | Purpose |
|--------|-----|---------|
| `33ce5f1d66` | S29-00 | Branch kickoff; state files updated; scope (E) locked |
| `73e9460a31` | S29-CLI-GUARD-T1 | Port `Cosine+{LG,ST}` rejection to `train_api.cpp:TrainConfigToInternal` + `csv_train.cpp:ParseArgs` (55 LoC, 2 C++ files) |
| `c73f5073af` | S29-CLI-GUARD-T2 | pytest coverage: 4 cases covering nanobind + CLI guard paths |
| `503ebacdb2` | S29-LG-SPIKE-T1 | LG+Cosine iter-1 drift measurement; harness + data artifacts (docs/ only) |
| `64a8d9076b` | S29-LG-SPIKE-T2 | DEC-034 verdict — outcome A (shared mechanism); moderate confidence |
| `3f87b85e38` | S29-CR + S29-SA | CR PASS-WITH-NITS (0 must-fix); SA PASS (0 findings); SA-H1 CLOSED |
| `fa7f9b55fc` | S29-CR SF-1 | Verdict wording tightened ("reaches 0.197%..." vs "<0.2%") |

### CLI-GUARD ports

`train_api.cpp:25-51` and `csv_train.cpp:241-267` now mirror the Python `_validate_params`
guards byte-for-byte (error text, TODO markers, exception types). Defense-in-depth: Python
is first line, C++ is second. `grep -rn 'TODO-S29-LG-COSINE-RCA'` returns exactly four sites
(Python + C++ nanobind + C++ CLI + test) — single-point-of-removal for S30.

### pytest coverage

`tests/test_cli_guards.py` — 4 tests, 2 paths:
- nanobind `_core.train()`: `pytest.raises(ValueError)` + TODO-marker substring assert
- `csv_train` CLI subprocess: `returncode != 0` + stderr TODO-marker assert

Forward-compatible assertion (`returncode != 0`, not `== 1`) survives planned S30-CLI-EXIT-WRAP.

### LG spike measurement

Cell: N=1000, depth=3, max_leaves=8, bins=128, lr=0.03, seeds={0,1,2}.
iter-1 mean drift: 0.0024% (per-seed: 0.0046 / 0.0015 / 0.0010).
50-iter peak: 0.197% (seed=1).
iter=1 BFS split sequences bit-identical CPU vs MLX (seed=0).

Cell-mismatch note: t5-gate-report's 14% LG ratio was pre-S28 algorithmic divergence (MLX L2
vs CPU Cosine, closed by `0ea86bde21`). The 0.0024% figure is the first honest post-S28
measurement.

### Gate results

| Gate | Report | Verdict |
|------|--------|---------|
| T1-CR — Code review | `docs/sprint29/fu-cr/t1-cr-report.md` | PASS-WITH-NITS (0 must-fix, 3 SF, 3 nits) |
| T1-SA — Security audit | `docs/sprint29/fu-sa/t1-sa-report.md` | PASS (0 findings, 2 info) |
| SA-H1 closure | `docs/sprint29/fu-sa/t1-sa-report.md` | CLOSED |
| Parity suite | (prior evidence; 28/28 at all S29 commits) | PASS |

### Decision updates

- **DEC-034**: PENDING-SPIKE → RESOLVED (outcome A). Moderate confidence.
- **DEC-032**: still PARTIALLY CLOSED. SA-H1 closed; guards remain until S30-COSINE-KAHAN.
- **DEC-035**: NEW — S30-COSINE-KAHAN planned (Kahan fix for shared joint-Cosine denominator).

### S30 carry items

- **S30-COSINE-KAHAN** (primary): apply Kahan/Neumaier to `ComputeCosineGainKDim` shared
  joint-denominator; gate both ST+Cosine and LG+Cosine; remove all guards atomically.
- **S30-CLI-EXIT-WRAP** (secondary, SA-I2-S29): add try/catch in `csv_train.cpp:main()` for
  graceful `exit(1)` instead of SIGABRT(134).
- **S31-LG-DEEP-RESIDUAL** (conditional): open only if post-Kahan drift persists on deep LG
  cells (depth>3, max_leaves>8).

---

## 2026-04-23 — Sprint 29 OPENED (DEC-032 Closeout + LG Mechanism Spike)

Branch `mlx/sprint-29-dec032-closeout` cut from master `987da0e7d5` (S28 merge commit). Scope (E)
per Ramos ultrathink triage. 8 tasks (#82–#89) created by orchestrator. Kickoff commit S29-00
lands state files only — no production code changes.

### Tasks opened

| ID | Tag | Purpose |
|----|-----|---------|
| #82 | S29-CLI-GUARD-T1 | Port Cosine+{LG,ST} guards to `train_api.cpp` + `csv_train.cpp` |
| #83 | S29-CLI-GUARD-T2 | Unit + CLI tests for C++ guards (blocked #82) |
| #84 | S29-LG-SPIKE-T1 | Instrument LG+Cosine iter-1 drift (parallel, 1-session cap) |
| #85 | S29-LG-SPIKE-T2 | Verdict doc: outcome A/B/C (blocked #84) |
| #86 | S29-BRANCH-DECISION | Human checkpoint: Ramos decides stretch vs close (blocked #85) |
| #87 | S29-CR | Code review (blocked #82, #86) |
| #88 | S29-SA | Security audit / SA-H1 closure (blocked #82, #86) |
| #89 | S29-CLOSE | Sprint close + DEC-032 fully CLOSED (blocked #87, #88) |

### Scope refinements

- Spike capped at 1 session; LG/ST Kahan carries to S30 on outcomes B/C.
- Iter-1 discriminator: LG ≈1% → outcome A (Kahan viable); LG ≥5% → outcome B (algorithmic); ambiguous → outcome C.
- T5 (#86) is a human-only decision; no auto-advance.

### New decisions

- DEC-034 (PENDING-SPIKE): LG-Cosine mechanism resolution. Resolves at #86 post-verdict.
- DEC-032: annotation updated — S29-CLI-GUARD (#82/#83) are the closing work items; #89 promotes to fully CLOSED.

---

## 2026-04-23 — Sprint 28 CLOSED (Score Function Fidelity, DEC-032 partially)

Branch: `mlx/sprint-19-hist-writeback`, tip `e0b0b1b527`. 9 commits. All exit gates PASS or
PASS-WITH-NITS/FINDINGS (non-blocking). Parity suite 28/28 at tip. PR ready — human-triggered.

### Commits

| Commit | Tag | Purpose |
|--------|-----|---------|
| `0409e632fa` | S28-00 | Branch kickoff; state files updated with acceptance criteria |
| `da02da0259` | S28-AUDIT | Formal grep audit: zero `score_function` refs in `catboost/mlx/` pre-S28; L2 call site confirmed at `csv_train.cpp:~L1281` |
| `83f30c3677` | S28-COSINE | `ComputeCosineGainKDim` helper ported from CPU `TCosineScoreCalcer` |
| `0ea86bde21` | S28-L2-EXPLICIT | `EScoreFunction` enum + `ParseScoreFunction`; dispatch in `FindBestSplitPerPartition` (DW/LG); nanobind binding; Python `_validate_params` rejecting `NewtonL2`/`NewtonCosine` |
| `4083add248` | S28-OBLIV-DISPATCH | Dispatch mirrored into `FindBestSplit` (SymmetricTree) |
| `c07e895f7c` | S28-REBLESS | 8 parity cells labeled with explicit `score_function`; AN-017 re-captured |
| `dca62f0d72` | S28-FU3-REVALIDATE | DW force-L2 lifted (passes Cosine both sides); LG retains force-L2 pending S29-LG-COSINE-RCA |
| `b9577067ef` | S28-{LG,ST}-GUARD | `ValueError` guards for `Cosine+Lossguide` and `Cosine+SymmetricTree` |
| `e0b0b1b527` | S28-CR-S1 | Dead `ComputeCosineGain` scalar helper removed (code-review CR-S1) |

### Gate reports

| Report | Path | Verdict |
|--------|------|---------|
| G2a/G2b — Cosine gate | `docs/sprint28/fu-cosine/t2-gate-report.md` | PASS |
| G3a/G3b/G3c — L2-explicit gate | `docs/sprint28/fu-l2-explicit/t3-gate-report.md` | PASS |
| G5a–G5d — Rebless gate | `docs/sprint28/fu-rebless/t4-rebless-report.md` | PASS |
| G6a–G6d — FU3-Revalidate gate | `docs/sprint28/fu-fu3-revalidate/t5-gate-report.md` | PASS |
| G7 — Obliv-dispatch gate | `docs/sprint28/fu-obliv-dispatch/t7-gate-report.md` | PASS |
| T6-CR — Code review | `docs/sprint28/fu-cr/t6-cr-report.md` | PASS-WITH-NITS |
| T6-SA — Security audit | `docs/sprint28/fu-sa/t6-sa-report.md` | PASS-WITH-FINDINGS |

### Key numbers

- DW+Cosine drift: 1.6% at N=1000/50k/50-iter — ships in-envelope.
- LG+Cosine: ~unacceptable drift — guarded at Python API; S29-LG-COSINE-RCA.
- ST+Cosine: ~0.77% @ 1 iter → ~47% @ 50 iter (float32 joint-denominator compounding) — guarded; S29-ST-COSINE-KAHAN.

### S29 carry items opened

- S29-CLI-GUARD (SA-H1): C++ / CLI bypass guards for forbidden combos.
- S29-LG-COSINE-RCA: Root-cause LG+Cosine unacceptable drift.
- S29-ST-COSINE-KAHAN: Kahan/Neumaier port for ST+Cosine denominator.

---

## 2026-04-23 — Sprint 28 KICKOFF (Score Function Fidelity)

Branch `mlx/sprint-28-score-function-fidelity` cut from master at `4b3711f82b` (S27 PR #25 merged). Small-sprint shape per Ramos 2026-04-23: stream A only, 8 tasks (S28-AUDIT through S28-CLOSE). Ride-alongs deferred: AN-008 Rule-5 promotion, CR Nit 2, SA Note 2, AA Item H, NewtonL2/NewtonCosine variants. State files updated with fleshed-out acceptance criteria per Ultrathink Task Planning standing order. Next agent: @ml-engineer picks up S28-AUDIT + S28-COSINE.

---

## 2026-04-22 — Sprint 27 CLOSED

26 commits (+ this close), 3 tracks closed cleanly. FU-1: DW leaf-index fix (DEC-030) — G1-FU1 6/6 PASS at `88cbe6d067`. AA: anchor audit + DEC-031 hygiene protocol — 0 class-b regressions, 4 class-a updates + 2 live-enforced anchors. FU-3: fidelity gap identified, scoped honestly to S28 via DEC-032 — G3-FU3 5/5 PASS at `591f4ce3e6` (conditional on CPU `score_function='L2'`). CR APPROVE `44bb9ee74b`, SA PASS-WITH-NOTES `24e80dde45`. PR pending Ramos. Sprint-close doc at `docs/sprint27/sprint-close.md`. Next: S28 score-function fidelity.

---

## 2026-04-22 — S27 Track C (FU-3) closed with scope-split to S28

FU-3 T1 identified the DW N=1000 asymmetry as a **fidelity gap**: MLX hardcodes L2 Newton gain; CPU CatBoost defaults to Cosine (`0931ad6e9c`). Not a parity-gate edge case. DEC-032 captures the honest framing: different algorithms, not parity-equivalent. Gate updated to require `score_function='L2'` on CPU side (NOT widening N scope — would be DEC-031 Rule-3 violation). S28 "Score function fidelity" opened as follow-up sprint to do the real port (audit plumbing → implement Cosine → re-bless aggregate parity claims → optional Newton variants).

---

## 2026-04-22 — S27 Tracks A + B closed (FU-1 + AA)

**Track A (FU-1 — Depthwise leaf-index fix)**: Two bugs in `ComputeLeafIndicesDepthwise` (encoding + split-lookup, 51.5% mismatch at depth=3). Fix per DEC-030: BFS-keyed split map + bit-packed partition accumulation. Gate G1-FU1 PASS 6/6 cells, ratios 0.9988–1.0027. Validation-only scope (call-site triage `eca086e4dd`). Commits: T1 `34f62b32c9`, audit `eca086e4dd`, DEC-030 `c7c09451e2`, fix `fb7eb59b5f`, gate `88cbe6d067`.

**Track B (AA — Anchor audit)**: 18 anchors inventoried, 0 class-b regressions, 4 class-a updates + 2 class-c + 3 class-d handled across 9 atomic commits. 2 anchors now live-enforced (AN-006, AN-007). DEC-031 codifies 5-rule anchor hygiene protocol. AN-008 flagged for Rule-5 promotion on next update (3rd lifetime).

**Remaining**: Track C (FU-3 DW N=1000 asymmetry triage) running in parallel; Track D (code review + security audit + sprint close) after FU-3 verdict.

---

## Sprint 27 — Track B anchor audit closed; DEC-031 adopted (2026-04-22)

**Branch**: `mlx/sprint-27-correctness-closeout`
**Task**: S27-AA-T5 (final Track B deliverable)

### What shipped

- **DEC-031 "Anchor hygiene protocol"** added to `.claude/state/DECISIONS.md`. Codifies the five standing rules derived from the S27-AA-T1–T4 audit of 18 committed numeric anchors:
  1. No new docs-only canonical values — every anchor must have a live pytest assertion.
  2. Anchor-change-on-path-change — any commit touching histogram/kernel/accumulation/leaf/gain must update or audit affected anchors atomically.
  3. Sprint-close drift check — re-run affected anchors at every sprint close as part of QA.
  4. Dead anchors removed or wired — class-d anchors resolve within the sprint they are found; never leave unreachable "canonical" values in docs.
  5. Repeat-offender promotion clause — AN-008 (3 lifetimes) must be promoted to a live test on its next value update.
- **MEMORY.md §Anchor hygiene** section added: 5-rule summary for future-agent consumption. AN-008 entry updated to reflect its third lifetime (`1.85752499`) and pending Rule 5 promotion obligation.

### Audit summary (T1–T4 recap for context)

| Class | Count | IDs |
|-------|-------|-----|
| a — stale-capture (T4 updated) | 4 | AN-006, AN-007, AN-008, AN-016 |
| a — already current (no T4 action) | 8 | AN-001–005, AN-009–011 |
| b — regression (escalate) | 0 | — |
| c — documented-supersession | 2 | AN-012, AN-018 |
| d — dead anchor | 3 | AN-013, AN-014, AN-015 |
| deferred-a (FU-1-dependent) | 1 | AN-017 |

T4 commit range: `adce339b56` (AN-006 P0) through `62f17df7a9` (AN-013/014 DEAD markers).

### Carry-forwards

- AN-017 re-capture deferred until FU-1-T3 merges to master (DW leaf-index fix, DEC-030).
- AN-008 live-test promotion (DEC-031 Rule 5) deferred to next kernel-touching sprint.
- CI lint for docs-only numeric values flagged as follow-up (out of S27 scope).

---

## Sprint 26 FU-2 closed — DEC-028 extended to FindBestSplitPerPartition (2026-04-22, CLOSED)

**Branch**: `mlx/sprint-26-fu2-noise-dwlg` (stacked on S26 D0 `66a4b5e869`)
**Sprint verdict**: ALL GATE PASS. 0 kill-switches fired. APPROVE-WITH-NITS from @code-reviewer (Nit-1 fixed at close; Nits 2/3/4 recorded as tech-debt).

### Root cause and fix

S26 D0 fixed RandomStrength noise in `FindBestSplit` (SymmetricTree). T1 source audit confirmed
CPU uses one global scalar (`CalcDerivativesStDevFromZeroPlainBoosting` → `scoreStDev`) computed
once per tree for all three grow policies. FU-2 extended `gradRms` threading into
`FindBestSplitPerPartition` (Depthwise and Lossguide paths). 47 lines changed in `csv_train.cpp`.
No kernel sources, leaf estimation, or SymmetricTree path modified.

**DEC decision**: Footnote added to DEC-028 in `docs/decisions.md` and `.claude/state/DECISIONS.md`.
No new DEC-030 opened — pure mirror of DEC-028's formula in the non-oblivious path; no new design
content.

### Commits landed

| Commit | Task | Description |
|--------|------|-------------|
| `7abd7b3bcf` | T1 | D0 triage doc — CPU global scalar gradRms confirmed |
| `478e8d5c9d` | T2+T3 | Thread gradRms into FindBestSplitPerPartition + smoke test |
| `715b15b613` | T4 | Extend test_python_path_parity.py to DW/LG |
| `ee5a90707b` | T5+T6 | G1 54-cell sweep + G5 Depthwise determinism artifacts |
| *(T8 commit SHA — set at commit time)* | T8 | Sprint close: Nit-1 fix, DEC-028 footnote, state files |

### Gate results

- **G1-DW** (segmented, N≥10k): 12/12 PASS. N=1000 failures (5 cells) are pre-existing — verified identical on pre-FU-2 binary.
- **G1-LG** (all cells): 18/18 PASS.
- **G2** (ST non-regression): 18/18 PASS — DEC-028 D0 fix intact.
- **G5** (DW determinism 100 runs): max−min 1.49e-08 (threshold 1e-6) — PASS.
- **KS-2/KS-3/KS-4/KS-5**: all CLEAR.

### What did NOT change

`catboost/mlx/kernels/kernel_sources.h` — untouched. `catboost/mlx/methods/histogram.cpp` — untouched. `catboost/mlx/methods/leaves/` — untouched. `catboost/mlx/gpu_data/` — untouched. v5 ULP=0 bench_boosting record preserved. bench_boosting binary does not exercise `FindBestSplitPerPartition`.

### Carry-forwards

- **S26-FU-1** (open): `ComputeLeafIndicesDepthwise` C++ validation path returns wrong index type.
- **S26-FU-3** (new): Depthwise N=1000 parity asymmetry — pre-existing, 5 failing cells (MLX better). Triage: per-partition gain comparison at depth-0.

---

## S26-D0-9 — Sprint 4 anchor update post-DEC-028 (2026-04-22)

**Branch**: `mlx/sprint-26-python-parity` (follow-up commit on PR #23)
**Trigger**: CI on PR #23 failed on `test_rmse_final_loss_matches_sprint4_anchor` (got 0.306348, expected ~0.432032).
**Attribution**: DEC-028 alone. `random_strength` ablation shows smooth monotone RMSE scaling at RS=0/1/2; pre-fix anchor lies off the curve. DEC-029 not exercised (tests use default `grow_policy="SymmetricTree"`).
**Stability**: determinism ~6e-9 under 1e-3 tolerance; seed=0 at 0.306348 is central in the seed sweep [0.304, 0.309].
**Scope**: 5 numeric constants in `python/tests/test_qa_round9_sprint4_partition_layout.py` across 3 tests (RMSE anchor, specific-predictions anchor, multiclass proba anchor).
**Precedent**: same pattern as TODO-022 Sprint 8 bench_boosting K=10 anchor update (`2.22267818 → 1.78561831`).
**Record**: `docs/sprint26/d0/d0-9-anchor-update.md`

## Sprint 26 D0 closed — Python-path parity; DEC-028 + DEC-029 shipped (2026-04-22, CLOSED)

**Branch**: `mlx/sprint-26-python-parity` (cut from `6c3953f239`)
**Framing**: correctness-first sprint. v5 kernel untouched (G4 preserved). R8 stays at 1.01×.
**Sprint verdict**: all 6 exit gates PASS (G0/G1/G2/G3/G4/G5 determinism).

### Commits landed (DEC-012 one-structural-change-per-commit)

| Commit | Role |
|--------|------|
| `24162e1006` | D0-6: DEC-028 RandomStrength noise formula — replace `totalWeight / numPartitions` with `sqrt(sum(g²)/N)` gradient-RMS |
| `0a2216138f` | D0: `.gitignore` match `catboost_info/` at any depth |
| `867784825e` | D0-7: G1 18-cell parity sweep + G4 100-run determinism artifacts |
| `20079cc4a3` | D0-7: G3 Python-path regression harness (`tests/test_python_path_parity.py`) |
| `cbbfc29257` | D0-7: G1/G3/G4 gate report |
| `9bd980a37f` | D0-8a: DEC-029 C++ — Depthwise/Lossguide SplitProps + `SplitBfsNodeIds` + `WriteModelJSON` `grow_policy` + `bfs_node_index` |
| `06fa2a58ee` | D0-8b: DEC-029 Python — `_predict_utils.py` dispatch on `grow_policy` + `_bfs_traverse_bitpacked` |
| `adb9d32835` | D0-8: DEC-029 decision entry + diagnostic artifacts |
| `2680252573` | D0-8: post-fix verification artifact (rs=0 algorithmic parity, rs=1 noise-path context) |

### Exit gate results

- **G0**: DEC-028 + DEC-029 entries complete in `docs/decisions.md`.
- **G1** (SymmetricTree 18-cell segmented): 18/18 PASS. rs=0 max |delta| = 0.43%, max |ratio−1| = 0.0043. rs=1 MLX_RMSE ≤ CPU_RMSE in every cell; pred_std_R ∈ [0.9996, 1.087]; Pearson > 0.99. Strict-symmetric would have been 12/18 (6 failures are MLX *better* than CPU at small N under rs=1 — unavoidable independent-RNG realization divergence).
- **G2** (Depthwise + Lossguide rs=0): DW −0.64%, LG −1.01% vs CPU. Pre-fix were +561% and +598%.
- **G3**: `tests/test_python_path_parity.py` — 8 parametrized tests — 8/8 PASS in 6.32s. Three orthogonal checks (RMSE ratio ±5%, pred_std_ratio ±10%, monotone-convergence ≤5% non-monotone).
- **G4**: `catboost/mlx/kernels/kernel_sources.h` untouched; v5 ULP=0 record intact.
- **G5** (determinism): 100 runs @ N=10k/seed=1337/rs=0, max−min = 1.49e-08 (std 6.17e-09). DEC-028 fix introduces no new non-determinism.

### Root causes

- **DEC-028**: `FindBestSplit` computed `noiseScale = randomStrength × totalWeight / (numPartitions × K)`; `totalWeight = N` for RMSE. At N=10k, noiseScale = 10,000 against a true root-split gain of ~1,602 → SNR 0.16 → noise dominates split selection → leaf magnitudes shrink. Fix: replace with CPU's `sqrt(sum(g²)/N)` formula. `gradRms` threaded from `RunTraining` into `FindBestSplit`.
- **DEC-029**: `TTreeRecord.SplitProps` was populated only in the SymmetricTree `else` branch. Depthwise/Lossguide `if` branches pushed `cursor` updates but not split descriptors → `WriteModelJSON` emitted `"splits": []` → `compute_leaf_indices` iterated an empty splits list → every doc assigned to leaf 0 → constant predictions at `leaf_values[0]`. Fix: populate `SplitProps` + new `SplitBfsNodeIds` in both non-oblivious paths, emit `grow_policy` and `bfs_node_index` per split (plus `leaf_bfs_ids` inverse map for Lossguide), dispatch Python predict on `grow_policy` with bit-packed BFS traversal that mirrors the C++ partition update.

### Methodology contributions (also captured in `../LESSONS-LEARNED.md`)

- **Segmented parity gate**: split symmetric `ratio ∈ [0.98, 1.02]` into (a) rs=0 tight (algorithmic parity) and (b) rs=1 one-sided + pred_std dual-check (preserves DEC-028-class regression catching without false-failing MLX-better cells).
- **`pred_std_R` as primary leaf-magnitude signal**: RMSE can be dominated by irreducible noise at small N; prediction std ratio catches leaf-magnitude shrinkage directly. DEC-028's signature was `pred_std_R ≈ 0.69`.
- **Parity-gate coverage label**: v5's "18/18 ULP=0" applied to kernel output only, not the `FindBestSplit` / nanobind / Python predict path. New standing order: gates must explicitly label their path coverage.

### Follow-ups

- **S26-FU-1** — `ComputeLeafIndicesDepthwise` validation path still returns `nodeIdx − numNodes` instead of bit-packed partition order. Affects validation RMSE tracking only. Listed in DEC-029 Risks.
- **S26-FU-2** — MLX Depthwise/Lossguide have no RandomStrength noise path. At rs=1, these policies under-fit CPU by ~10–12% at N=10k. Pre-existing — not a S26 regression. Scope: separate parameter-threading sprint.

### State updates

- `.claude/state/HANDOFF.md` — S26 D0 closed section added; current state + next actions rewritten; new S26 standing order captured (gate-coverage labeling).
- `.claude/state/TODOS.md` — S26 D0 items checked; follow-ups S26-FU-1 / S26-FU-2 opened.
- `.claude/state/DECISIONS.md` — DEC-028 + DEC-029 mirrored from `docs/decisions.md`.
- `.claude/state/MEMORY.md` — segmented-gate methodology, pred_std_R signal, Python-path coverage gap captured as cross-sprint lessons.
- Cross-project: `../LESSONS-LEARNED.md` (Frameworks-level) — 24 principle-first lessons including the S26 methodology contributions.

## Latent-bugs cleanup (2026-04-22, PR #20)

Triage + close-out of the three items carried forward since Sprint 12 / Sprint 23:

| Commit | Role |
|--------|------|
| `668e33ca4d` | state: close K=10 anchor + BUG-007; reframe S-1 as compile-time structural guard |
| `a9b2a1b757` | train_api: `BuildDatasetFromArrays` throws on unsorted `groupIds` (BUG-007 defense-in-depth) |
| `50efeb2ade` | histogram: `maxBlocksPerPart` promoted to `constexpr` + `static_assert` (Sibling S-1) |
| `71aabaa842` | Merge PR #20 to master |

**Scoping surprise**: all three items turned out to be doc-drift or latent-no-repro, not the engineering bugs the ledger implied. K=10 anchor was already fixed in Sprint 8 (TODO-022); BUG-007 was already handled at the Python layer (`core.py:1131-1137`); Sibling S-1's "NIT-4 CB_ENSURE" was in practice a hardcoded `const ui32 = 1` with no runtime guard. One ~80-line commit stack aligned the ledger with reality, added a C++ contract CB_ENSURE as defense-in-depth, and promoted the S-1 literal to a compile-time `static_assert` that fails loudly if anyone raises it.

No production behavior change. CI green (4/4 on PR #20).

## CI unblock + stack merge (2026-04-22)

PRs #16 (Sprint 24) and #17 (Sprint 25) had been sitting unmerged because their CI was red on two pre-existing breakages inherited from master: `mlx-build.yaml` calling removed `python -m mlx --includes/--libs` flags (dropped in MLX 0.31), and two stale tests (`test_version_is_0_3_0` hard-pinned to an old version, `test_mae_uppercase_fails_cleanly__bug001` asserting an overly-broad regression sentinel that fired when the BUG-001 crash was silently fixed). All three surfaced only after earlier PRs merged.

Fix landed in PR #18 (three atomic commits under DEC-012) and unblocked the stack:

| Commit | Role |
|--------|------|
| `c28cacabfe` | ci: resolve MLX headers via `python -m mlx --cmake-dir` walk-up (durable across flat + `mlx-metal` split layouts) |
| `a542856ace` | tests: replace hard-pinned `0.3.0` equality with `importlib.metadata` self-consistency + drop `minor == 3` pin |
| `b1aad56ec1` | tests: narrow BUG-001 MAE sentinel to SIGABRT-only (accept both clean-error and clean-accept outcomes) |
| `9b0c03fec2` | Merge PR #18 to master |
| `1385e056ca` | Merge PR #16 to master (Sprint 24, rebased onto #18 tip) |
| `5caa6e64cf` | Merge PR #17 to master (Sprint 25, rebased onto #16 tip) |

Stack is now clear. No production code changes in any of the three merges — S24 shipped v5 and S25 shipped falsification evidence, both already reflected in earlier changelog entries. PR #17 briefly closed when its base branch was auto-deleted post-#16-merge; restored via a temporary base-branch push and a base-retarget to master.

## Sprint 25 closed — DEC-026 FALSIFIED at G1; R8 unchanged at 1.01× (2026-04-21, CLOSED)

**Branch**: `mlx/sprint-25-dec026-cascade` (cut from Sprint 24 branch tip `3f4fff8a2d`, stacked on `mlx/sprint-24-dec023-fix`)
**Campaign**: Post-Verstappen research — R8 recovery investigation
**Sprint verdict**: FALSIFIED at G1 on day 1. ε-threading impossible by 21,091× under optimistic positive-gap reading. G2–G5 not attempted. No production code changes. PR #17 pending.

### Commits landed

| Commit | Role | Verdict |
|--------|------|---------|
| `59cbf1bb5c` | S25 kickoff — branch cut + scaffold corrections | — |
| (this commit) | S25 closeout — G1 empirical sweep scaffold + 180-run results + analyzer + verdict doc + state closeout | FALSIFIED |

### G1 empirical sweep (falsification evidence)

- **Scaffold** (`benchmarks/sprint25/g1/`): Path 5 reconstruction (T2-sort serial scatter + int-atomic fixed-point SCALE=2³⁰ for feats 1-3; feat-0 bin-range scan over sortedDocs), Option A dump kernel (`kScoreSplitsDumpSource` emits top-5 + rank=255 sentinel per NodePlaceholders eval), bench_boosting fork `g1_gain_dump.cpp` with `--kernel`, `--emit-gain-trace`, `--gain-topk` flags. `catboost/mlx/kernels/kernel_sources.h` UNTOUCHED throughout.
- **Sweep** (180 runs, 5 min 4 s wall): 18 DEC-008 configs × 5 runs × 2 kernels (T1 + Path 5). All 180 runs deterministic 5/5. T1 reproduces all 18 DEC-008 reference losses. 17/18 configs: T1 ≡ Path 5 bit-exact. Config #8 only: T1 = 0.48231599 (Value A), Path 5 = 0.48231912 (Value B) — 105 ULP.
- **Flip analysis**: 7 unique (iter, depth_level) flip events × 5 runs = 35 total, all at config #8. Bit-identical across runs. 6/7 at depth 0 (root split flip at iters 44-49); 1/7 at depth 1 (iter 43 near-tie, 5.96e-08 gap). Iter 45 depth 0 sets ε_min = 2.200e-03.
- **ε threading**: ε_min = 2.200e-03 vs ε_max⁺ = 1.043e-07 (configs 1/2/8/14 have zero-gain ties pinning strict ε_max = 0). Safety ratio 4.74e-05 vs required 2.0 — 21,091× below threshold.
- **Verdict**: Path 5's flip gaps span 5.96e-08 to 2.2e-03 — the full range of legitimate top-2 separations at non-#8 configs. No ε can simultaneously gate the 2.2e-03 flip at config #8 iter 45 and leave the 1.04e-07 legitimate separation at config #1 iter 40 depth 3 untouched. Cascade-robust GAIN approach is structurally infeasible under DEC-008 discipline.

### R8 and forward paths

- **R8**: 1.01× unchanged (post-S24 honest position). DEC-026 cannot recover pre-S24 1.90× under ULP=0 parity. Verstappen ≥1.5× gate remains retroactively failed from S24 D0.
- **DEC-027 deferred**: alternative accumulation paths (XGBoost-style per-feature deterministic radix-sum) acknowledged in verdict doc §9 option 4 but not opened as part of S25 closure. Ramos to revisit later in a dedicated research sprint.

### State updates

- `DECISIONS.md` DEC-026 → FALSIFIED; falsification result section added with full ε-threading table
- `HANDOFF.md` current state + prior sprints updated; S25 closed section added
- `TODOS.md` S25-G1 FALSIFIED; G2–G5 CANCELLED
- `KNOWN_BUGS.md` — no changes

## Sprint 24 closed — DEC-023 resolved via v5; R8 retroactive retreat 1.90× → 1.01× (2026-04-21, CLOSED)

**Branch**: `mlx/sprint-24-dec023-fix` (cut from Sprint 23 tip `5b9827ad93` after S17–S23 PR chain merge)
**Campaign**: Operation Verstappen — battle 9 of 9
**Sprint verdict**: D0 PASS (DEC-023 RESOLVED, all 4 acceptance criteria pass). FAIL on R8
preservation (Verstappen ≥1.5× gate failed retroactively at 1.01× post-fix). PR #16 pending.

### Commits landed

| Commit | Role | Verdict |
|--------|------|---------|
| (prior S24 work) | Path 5 diagnostic attempts (T2-sort prefix-sum + int-fixed-point) | FALSIFIED — all pin Value B |
| (prior S24 work) | Path X CPU anchor measurement | INCONCLUSIVE — bench_boosting not a conformance harness |
| (prior S24 work) | Off-by-one cascade retest | FALSE POSITIVE — both paths encode raw_bin > splitIdx |
| `784f82a891` | v5 cherry-picked — T2-accum all-feature T1-style accumulation; T2-sort removed | SHIPPED |

### D0 — DEC-023 fix (RESOLVED)

**Bug**: Features 1-3 `atomic_fetch_add_explicit(memory_order_relaxed)` on float in T2-accum
produced bimodal output at config #8 (105 ULP gap, ~50/50 between 0.48231599 and 0.48231912).
Carried from Sprint 23 D0 as DEC-023 OPEN.

**Diagnostic arc**:

*Path 5 (falsified)*: T2-sort deterministic prefix-sum scatter + int64 fixed-point for features 1-3.
All variants retaining feature-0's bin-range scan over `sortedDocs` pinned to Value B (105 ULP
off T1). Root cause: reduction topology difference between sort-based scan and T1's SIMD fold.
Integer accumulation eliminated S-5 non-associativity but did not change the topology. A
deterministic result at the wrong value is not a fix.

*Path X CPU anchor (inconclusive)*: CPU CatBoost at config #8 = 0.068, ~24M ULP from both A and
B. bench_boosting is not a CatBoost conformance harness (no boost_from_average, simplified split
loop). T1 Value A (0.48231599) remains the declared parity anchor by construction, not because
it matches CPU CatBoost.

*Off-by-one retest (false positive)*: Proposed mismatch between scoring kernel ("bin ≥ b") and
apply path ("bin > b") was a coordinate-system labeling artifact. Code audit confirmed both
paths encode `raw_bin > splitIdx` consistently with CatBoost's `IsTrueHistogram`. No bug.
Diagnostic preserved at `docs/sprint24/d0_offby1_cascade_retest.md`.

*v5 (correct fix)*: All four features (0-3) in T2-accum rewritten to T1-style SIMD-shuffle
accumulation reading from `docIndices`. T2-sort removed from dispatch. ULP=0 is structural —
v5 executes the identical FP computation as T1. Commit `784f82a891`.

**Acceptance-criteria results**:

| Gate | Criterion | Measured | Verdict |
|------|-----------|----------|---------|
| S24-D0-G1 | Config #8: 10/10 deterministic | 10/10 at 0.48231599, ULP=0 | PASS |
| S24-D0-G2 | 18/18 ULP=0, ≥5 runs per config | 18/18 ULP=0, all 5/5 det. | PASS |
| S24-D0-G3 | Gate config: 100/100 deterministic | 100/100 at 0.47740927 | PASS |
| S24-D0-G4 | hist_ms ratio ≥ 0.45× (kill-switch) | 0.959× | PASS |

### R8 collapse: 1.90× → 1.01×

| Metric | S23 D0 (T2 v4, non-det.) | S24 D0 (T2 v5, det.) |
|--------|:------------------------:|:--------------------:|
| hist_ms (gate config) | ~6.85 ms (0.317× T1) | ~20.75 ms (0.959× T1) |
| e2e speedup vs S16 baseline | **1.90×** | **~1.01×** |
| Verstappen ≥1.5× | cleared by 40 pp | **FAILED retroactively** |

T2's speed advantage was contingent on its sort-based accumulation having a different reduction
topology from T1. The topology difference is also the root cause of DEC-023. These are not
separable: fixing the topology eliminates the speed. The 1.90× record is superseded. Honest
post-S24 position: 1.01×.

### Decisions updated

| Decision | Change |
|----------|--------|
| DEC-023 | RESOLVED 2026-04-21. Close-commit `784f82a891`. 4/4 gates PASS. R8 consequence appended. |
| DEC-026 | NEW — OPEN (S25 research). Cascade-robust GAIN comparison research track. |

### KNOWN_BUGS.md updated

BUG-T2-001: marked RESOLVED 2026-04-21. Fix summary and forward pointer to DEC-026 prepended.
Sibling S-1 (`kHistOneByte` writeback race) still latent, still guarded by NIT-4 CB_ENSURE —
no change to S-1 status.

### Championship benchmark

Not run. Campaign retreated before suite started. S24-BENCH-G1 NOT RUN.

### Sprint 25

DEC-026 cascade-robust GAIN research opens. @research-scientist leads epsilon calibration study
(DEC-026-G1). If viable ε identified, research proceeds through T2 Path 5 rebuild and 5-gate
acceptance suite. If no viable ε, DEC-026 is falsified and R8 stays at 1.01×. Not a guaranteed
delivery. See `docs/sprint25/README.md` and `DECISIONS.md DEC-026`.

---

## Sprint 23 closed — T2 scratch→production promotion + NIT cleanup + tree-search research (2026-04-21, CLOSED)

**Branch**: `mlx/sprint-23-t2-promotion` (cut from Sprint 22 tip `73baadf445`)
**Campaign**: Operation Verstappen — battle 8 of 9
**Sprint verdict**: PASS with pre-existing-bug footnote. R8 = **1.90×** (unchanged through S23). Verstappen ≥1.5× gate remains cleared by 40 pp. PR #15 pending (Ramos opens, stacked on #14).

### 8 commits landed

| Commit | Role | Verdict |
|--------|------|---------|
| `4d1eda1f4c` | D0 Commit 1 — `kT2SortSource` + `kT2AccumSource` into `kernel_sources.h`; NIT-1/2/7 applied | PASS |
| `2df0bb1aed` | D0 Commit 2 — `DispatchHistogramT2` promoted into `histogram.cpp`; CB_ENSURE API | PASS |
| `eaf05bc21d` | D0 Commit 3 — `CATBOOST_MLX_HISTOGRAM_T2` flag removed; T2 default; NIT-3/4/5 applied | PASS |
| `84529b47ed` | D0 Commit 4 — parity re-verify post-promotion; 17/18 ULP=0 | PASS (kill-switch tripped) |
| `dd1c9e0a6e` | D0 close-out — bimodality pre-existing verdict; `d0_bimodality_verification.md` | DONE |
| `be530059da` | D0 records correction — S22 D3 errata; DEC-022 scope qualifier; DEC-023 opened; KNOWN_BUGS.md; S24 scaffold | DONE |
| `441f632b10` | R1 doc — `r1_evalatboundary.md`; DEC-024 DEFERRED | DONE (no-op) |
| `5b9827ad93` | R2 doc — `r2_dispatch_inversion_spike.md`; DEC-025 FALSIFIED | DONE (no-op) |

### D0 — T2 promotion (4 commits + 2 close-out commits)

Kernel sources and host dispatch promoted from scratch form to production. NIT-1 through NIT-7 (minus NIT-6, removed in S22 audit) applied across Commits 1 and 3. `CATBOOST_MLX_HISTOGRAM_T2` compile-time flag removed; T2 is now the unconditional default dispatch path.

**Kill-switch**: TRIPPED at config #8 (N=10000/RMSE/128b, 105 ULP gap, ~50/50 bimodal between 0.48231599 and 0.48231912). **Verdict: PRE-EXISTING.** The bimodality is present in S22 D2/D3 tip `73baadf445`. Root cause: features 1-3 `atomic_fetch_add_explicit(memory_order_relaxed)` on float; non-associative accumulation under non-deterministic Metal scheduling. S22 D3's 1-run-per-config parity sweep had 50% miss probability for a ~50/50 race. The promotion is innocent.

**Gate config #14** (50k/RMSE/128b): 100/100 deterministic at 0.47740927. R8 1.90× record unaffected.

**D0 exit gates**:

| Gate | Criterion | Verdict |
|------|-----------|---------|
| S23-D0-G1 | 18/18 ULP=0 post-promotion (≥ S22 D3 standard) | PASS with errata — 17/18 ULP=0; G1 satisfied pending DEC-023 at S24 D0 |
| S23-D0-G2 | iter_total_ms ≤ 19.5 ms at gate config | PASS — unchanged at 19.098 ms |
| S23-D0-G3 | T2 in `kernel_sources.h`; inline T2 removed from `bench_boosting.cpp`; flag removed | PASS |
| S23-NIT-G | All 6 deferred nits addressed | PASS |

### R1 — EvalAtBoundary readback elimination (DEFERRED)

Sites A/B/C in `structure_searcher.cpp` (`:290`, `:609`, `:705`) are on Depthwise/Lossguide paths only. Gate config runs SymmetricTree (oblivious). `bench_boosting` never calls `structure_searcher.cpp`. The ~0.3 ms/iter estimate was a theoretical S16 cost-class projection, not a measured value. 0/3 sites are reachable from the gate path. Zero code changes. R8 = 1.90× unchanged. DEFERRED, not retired — re-entry requires `bench_boosting --grow-policy` flag or a separate Depthwise/Lossguide harness. See DEC-024 and `docs/sprint23/r1_evalatboundary.md`.

### R2 — Dispatch inversion spike (FALSIFIED)

Proposal: replace partition-fragmented 1664-TG dispatch with a single all-docs histogram over `(feature × stat × bin)`, recovering per-partition bin sums at scoring time. Structural algebraic blocker: `H[f][b] = Σ_p h_p[f][b]` is not invertible. All five candidate mask mechanisms (A through E) are algebraically or empirically rejected — each either performs equivalent work to the current per-partition histogram or blows the 5.82 ms headroom budget. Atomic contention under inversion is 64× worse than the DEC-023 trigger. Mechanism E (the only variant retaining the 195 docs/thread shape) is DEC-017 T3b without the CAS — the same +42.3% regression is the predicted outcome. Day-1 kill-switch invoked; Day 2 not exercised. FALSIFIED permanently. See DEC-025 and `docs/sprint23/r2_dispatch_inversion_spike.md`.

### Records corrected

- **S22 D3 parity verdict**: "18/18 ULP=0 bit-exact" corrected to **17/18 ULP=0 + 1 latent bimodal** (config #8). Errata prepended to `docs/sprint22/d3_parity_gate.md` and `docs/sprint22/d2_t2_fix_verified.md`.
- **S22 D2 determinism claim**: "10/10 determinism" was at gate config only; config #8 was not tested.
- **DEC-022 scope qualifier**: "bug β does not exist" scoped to gate config; race fires at N=10000. Original retirement of Kahan concern remains valid at gate.
- **DEC-020 footnote**: corrects the "18/18" claim and points to DEC-023.

### New decisions

| Decision | Status | Summary |
|----------|--------|---------|
| DEC-023 | OPEN (S24) | Features 1-3 atomic-float race; fix options: threadgroup-local reduce (preferred), int-atomic fixed-point, Kahan (insufficient standalone) |
| DEC-024 | DEFERRED | S23-R1 EvalAtBoundary elimination; blocked by harness gap; not retired |
| DEC-025 | FALSIFIED | S23-R2 dispatch inversion; structural algebraic blocker; do not re-enter |

### KNOWN_BUGS.md

BUG-T2-001 created: features 1-3 atomic-float race, config #8 bimodal, DEC-023 fix target S24 D0. Sibling latent race S-1 (`kHistOneByte` writeback, currently dead code, guarded by NIT-4 CB_ENSURE) documented.

### Parity-sweep protocol standing order (carried forward)

Minimum 5 runs per non-gate config + 100 runs at gate unconditionally. Effective from S23 D0 forward.

---

## Sprint 23 D0 — T2 scratch→production promotion; kill-switch tripped (pre-existing bimodal at config #8); S22 records corrected; DEC-023 opened (2026-04-20, D0 COMPLETE)

**Branch**: `mlx/sprint-23-t2-promotion` (cut from Sprint 22 tip `73baadf445`)
**Campaign**: Operation Verstappen — battle 8 of 9
**D0 verdict**: PASS (kill-switch tripped on pre-existing bug; proceed to R1/R2)

### 4 commits landed (D0 promotion arc)

| Commit | Content |
|--------|---------|
| (Commit 1) | Kernel sources promotion: `kT2SortSource` + `kT2AccumSource` into `kernel_sources.h`; NIT-1/NIT-2/NIT-7 applied |
| (Commit 2) | Dispatch promotion: `DispatchHistogramT2` into `histogram.cpp`; production API with CB_ENSURE |
| (Commit 3) | Flag removal + default flip: `CATBOOST_MLX_HISTOGRAM_T2` removed; T2 is default; NIT-3/NIT-4/NIT-5 applied |
| `84529b47ed` | Parity re-verify post-promotion (tip commit) |

### Kill-switch trip: config #8 bimodal

Parity sweep result: **17/18 ULP=0 deterministic + 1 latent bimodal** (config #8: N=10000/RMSE/128b, 105 ULP gap, ~50/50 between 0.48231599 and 0.48231912). Kill-switch tripped.

**Verdict: PRE-EXISTING.** The bimodality is present in S22 D2/D3 tip `73baadf445` and was not introduced by promotion. Root cause: features 1-3 `atomic_fetch_add_explicit(memory_order_relaxed)` on float is non-associative; non-deterministic Metal thread scheduling produces 1-2 ULP bin drift that cascades to 105 ULP over 50 iterations at this config's dispatch shape. See `docs/sprint23/d0_bimodality_verification.md`.

S22 D3's 1-run-per-config parity sweep had a 50% probability of missing a 50/50 bimodal race per config; the miss was statistically expected, not unlucky.

Gate config #14 (N=50000/RMSE/128b): **100/100 deterministic** at 0.47740927. R8 1.90× record unaffected.

### Records corrected

- **S22 D3 parity verdict**: "18/18 ULP=0 bit-exact" → corrected to **17/18 ULP=0 + 1 latent bimodal** (config #8). Errata prepended to `docs/sprint22/d3_parity_gate.md`.
- **S22 D2 determinism claim**: "10/10 determinism" was at gate config only; config #8 not tested. Errata prepended to `docs/sprint22/d2_t2_fix_verified.md`.
- **DEC-022**: Scope qualifier added — "bug β does not exist" scoped to gate config; race fires at N=10000.
- **DEC-020**: Footnote added correcting the 18/18 claim and pointing to DEC-023.

### DEC-023 opened

Features 1-3 atomic-float race documented in `DECISIONS.md DEC-023` as OPEN (S24 scope). Three fix options enumerated. Fix budget: S24 D0, 1-2 days.

### Parity sweep protocol standing order

Minimum 5 runs per non-gate config (97% detection probability for 50/50 race); 100 runs at gate config unconditionally. Documented in `docs/sprint23/README.md §5`.

---

## Sprint 22 — T2 sort-by-bin SHIPPED; Option III fix; Verstappen ≥1.5× gate CLEARED; R8 1.90× (2026-04-20, CLOSED)

**Branch**: `mlx/sprint-22-t2-integration` (cut from Sprint 21 tip `a7a206b90d`)
**Campaign**: Operation Verstappen — battle 7 of 9
**Verdict**: **CLOSED. 4/4 exit gates PASS. T2 sort-by-bin validated. Cumulative R8 = 1.90×. Verstappen ≥1.5× gate CLEARED by 40 pp.**

### Sprint arc: D0 PASS → D1 parity failure → four-phase diagnostic → Option III fix → 4/4 gates PASS

Sprint 22 began with an in-situ T2 integration probe (D0) that passed its kill-switch at 0.328× ratio — inside the optimistic band. D1 parity sweep then failed 18/18 configs (ULP 1,327–2,583,206), triggering a four-phase diagnostic arc:

- **D1a**: blit-ordering hypothesis (fill_gpu pool reuse) — REFUTED (fill_gpu is compute; eval barriers did not fix parity)
- **D1b**: depth-parity indexing hypothesis — REFUTED (even-depth pattern explained by split-distribution artifact)
- **D1c**: root cause identified — `bench_boosting.cpp:526` `maxPartDocs = ceil(numDocs / numActiveParts)` uniform-partition assumption. Under real argsort-permuted splits at depth 1 on 50k docs, partitions are [442, 49558] vs `maxPartDocs=25000`; 24558-doc overflow into the neighboring TG's `sortedDocs` slot corrupted histograms. `iters=1` always passed (depth=0 → single partition, no overflow possible).
- **D2**: Option III fix (slab-by-partOffsets). `sortedDocs` reorganized to per-(groupIdx, statIdx) slabs of size `numDocs` indexed by `partOffsets[partIdx]`. Overflow structurally impossible since `sum(partSizes) == numDocs`. Buffer 5.2 MB at gate config vs 333 MB worst-case for Option I one-line fix.

Side-finding: bug β (atomic-scatter float drift, S21 D1-R4 §3 risk) does not exist. 10/10 and 100/100 determinism confirmed post-fix. Kahan compensation concern retired (DEC-022).

### Commits landed (2 kernel/state commits)

| Commit | Content | Verdict |
|--------|---------|---------|
| `4333c82a7e` | D0 in-situ T2 probe at production shape | PASS — ratio 0.328× (optimistic band) |
| `73baadf445` | D1+D1a+D1b+D1c+D2 Option III fix + D3/D4/D5/D6 gate reports | 4/4 GATES PASS |

### Exit gates

| Gate | Criterion | Verdict |
|------|-----------|---------|
| D3 parity | 18/18 DEC-008 ULP=0; 100/100 determinism; EC-1–EC-5 all ULP=0 | **PASS** |
| D4 perf | Ratio 0.317× cross-session; cumulative R8 = 1.90×; gate cleared +40 pp | **PASS** |
| D5 code review | 0 blockers, 6 nits deferred to S23 | **PASS** |
| D6 security audit | 0 CRITICAL/HIGH; overflow class structurally eliminated; max-safe-N 14.3M | **PASS** |

### Final numbers

| Metric | Value |
|--------|-------|
| T2/T1 hist_ms ratio (gate config) | 0.317× cross-session (band 0.315–0.319×) |
| S22 e2e multiplier | 1.778× (33.958 ms → 19.098 ms iter_total) |
| Cumulative R8 post-S22 | **1.07 × 1.778 = 1.90×** |
| Verstappen gate (≥1.5×) | **CLEARED +40 pp** |
| Parity | 18/18 ULP=0; 100/100 determinism; BENCH_FINAL_LOSS T1=T2=0.47740927 |

### Decisions recorded

- **DEC-020**: status advanced from VIABLE → **SHIPPED / VALIDATED**
- **DEC-021**: Option III slab-by-partOffsets layout chosen over Option I (5.2 MB vs 333 MB; overflow structurally eliminated; 1.6 pp perf headroom vs D0)
- **DEC-022**: Kahan/compensated-summation concern RETIRED — bug β does not exist (10/10 + 100/100 determinism post-fix)

### PR #14 target

`RR-AMATOK/catboost-mlx` — stacked on PR #13 (Sprint 21). Ramos opens. Title: `[mlx] sprint-22: T2 sort-by-bin — Option III fix, 4/4 gates PASS, R8 1.90×`.

### Sprint 23 backlog (from S22 closeout)

D0 task: T2 scratch→production promotion (move `kernel_sources_t2_scratch.h` → `kernel_sources.h`, `DispatchHistogramT2` → `histogram.cpp`). 6 deferred NIT catalog items. Tree-search restructure research track (S23-R1 EvalAtBoundary readback, S23-R2 dispatch inversion spike).

---

## Sprint 21 — A1 measurement sprint; L2 FALSIFIED; T2 VIABLE; variant A RETIRED; 0× perf shipped (2026-04-20, CLOSED)

**Branch**: `mlx/sprint-21-hist-tg-reduction` (cut from Sprint 20 tip `85b6362b6e`)
**Campaign**: Operation Verstappen — battle 6 of 9
**Verdict**: **CLOSED via A1 measurement sprint.** 6/6 A1 exit gates PASS. 0× net perf delta shipped (A1-G6 discipline — no production source modified). Two levers retired; one promoted to viable-set.

### A1 pivot rationale

Sprint 21 was planned as a TG-count reduction (variant A) integration sprint. D0 kill-switch fired on day 1: fixed per-TG overhead at depth 6 = 2.5% ± 1.3% (R²=0.9989 depth regression), far below the ≥10% gate. A specification error was discovered: the D0 gate tested T1 fixed-overhead amortization as a proxy for variant A's actual mechanism (T3b shape restoration at 195 docs/thread). Ramos chose option (a): honor the kill-switch strictly. Sprint 21 retargeted to A1 — a measurement-only sprint producing production-shape evidence for two lever candidates. Generalizable lesson encoded in `feedback_ultrathink_task_planning.md`.

### Commits landed (5, all docs/instrumentation — zero kernel changes)

| Commit | Content | Verdict |
|---|---|---|
| `a0c473e3b7` | D0 kill-switch: depth-sweep regression, fixed overhead = 2.5% ± 1.3% | FIRED — variant A RETIRED (DEC-018) |
| `ac378d8de6` | D1-R3 per-kernel-profile instrumentation in `bench_boosting.cpp` | DONE — stable, stdev < 5% of mean |
| `fedf9d5348` | D1-R1 L2 direct mechanism test (`stat = 1.0f` zero-gather at 1664-TG depth-6) | FALSIFIED — +2.61% slower (DEC-019) |
| `13322feaca` | D1-R2 T2 sort-by-bin production-shape micro-bench (sort+accum, 1664-TG shape) | VIABLE — −64.8% (DEC-020) |
| `a7a206b90d` | D1-R4 synthesis + Sprint 22 kickoff plan (`docs/sprint21/d1r4_synthesis.md`) | DONE — mechanism-direct gates; R8 ledger |

### Two decisions retired

- **DEC-018 TG-count reduction variant A — RETIRED** (was DRAFT-S21, never activated). D0 kill-switch fired (2.5% << 10% gate). Specification error captured: gate tested T1 amortization proxy, not the T3b shape-restoration mechanism that was the actual savings source. `docs/sprint21/d0_attribution.md §6.2`.
- **DEC-019 L2 stats pre-permute — FALSIFIED**. Zero-gather upper bound (stat=1.0f): +2.61% slower at 1664-TG depth-6 production shape. 12.6 pp below 10% gate. AGX out-of-order execution + hardware L2 prefetcher fully hide the stats gather. Generalizes S19-01c probe D single-TG finding to multi-TG depth-6. `docs/sprint21/d1r1_l2_attribution.md`.

### One decision promoted

- **DEC-020 T2 sort-by-bin — VIABLE (pending Sprint 22 D0 in-situ)**. D1-R2 at 1664-TG production shape: −64.8% histogram_ms (band 63.6–66.7%, 2σ ±2.7–4.4%), clearing 50% gate by 28–34 pp. Gate B parity: max ULP 64, mass conservation 0 ULP across 812,800 bins. Enters Sprint 22 viable-set rank #1. Ratio-transfer risk (synthetic identity-permuted → production argsort-permuted) unproven; Sprint 22 D0 tests directly with kill-switch at ratio > 0.60. `docs/sprint21/d1r2_t2_microbench.md`.

### R8 — honest

- Sprint 21 contribution: **0× by design** (A1 measurement sprint; no perf change intended or shipped)
- Cumulative through Sprint 21: **~1.07× over Sprint 16-class baseline** (from S17/S18/S19 kernel improvements only)
- Gap to Verstappen 1.5× gate: **40% residual** — reachable iff T2 clears Sprint 22 D0 at ratio ≤ 0.60

### Sprint 21 exit gates

| Gate | Criterion | Status |
|---|---|---|
| A1-G1 | D0 kill-switch executed with production-shape evidence | PASS (`a0c473e3b7`) |
| A1-G2 | D1-R3 per-dispatch timings stable (stdev < 5% of mean) | PASS (`ac378d8de6`) |
| A1-G3 | D1-R1 binary L2 verdict at production shape | PASS — FALSIFIED (`fedf9d5348`) |
| A1-G4 | D1-R2 binary T2 verdict at production shape (sort-inclusive) | PASS — VIABLE (`13322feaca`) |
| A1-G5 | D1-R4 Sprint 22 plan has mechanism-direct gates | PASS (`a7a206b90d`) |
| A1-G6 | No kernel source committed on Sprint 21 branch | PASS (zero production source diffs) |

### PR #13 target

`RR-AMATOK/catboost-mlx` — stacked on PR #12 (Sprint 20). Ramos opens. Title: `[mlx] sprint-21: A1 measurement sprint — L2 falsified, T2 viable, variant A retired`.

---

## Sprint 20 — T3b atomic-CAS FALSIFIED at D2; DEC-017 RETIRED; 0× ship, empirical record + Sprint 21 redesign (2026-04-19, CLOSED via falsification)

**Branch**: `mlx/sprint-20-hist-atomic-cas` (cut from Sprint 19 tip `4113200529`)
**Campaign**: Operation Verstappen — battle 5 of 9 — L_accum lever (T3b variant)
**Verdict**: **FALSIFIED.** Toy-kernel −84.4% single-TG accumulation did not translate to production partition-fragmented dispatch. D2 integration measured +42.3% regression at gate config (50k/RMSE/d6/128b), far outside the stop-bound of [9.0 ms, 21.1 ms]. Kernel + host changes reverted pre-commit per standing orders. DEC-017 RETIRED. **0× net perf delta shipped this sprint.** PR #12 ships the empirical record and Sprint 21 redesign plan.

### Commits landed (3, all docs/state)

1. **`9216f4941c`** — D1 parity sweep. T3b 18/18 configs bit-exact vs T0 production kernel (ULP = 0 everywhere, stronger than DEC-008 envelope). 100-run determinism at gate config produced a single unique BENCH_FINAL_LOSS. **Critical CRITIQUE catch during implementation**: the T0 baseline in `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp` originally omitted the DEC-009 cross-SIMD fold — without that correction the T3b vs T0 ULP would have been spuriously non-zero and masked a real D2 green-light. Harness in `docs/sprint20/scratch/microbench_parity.cpp` (905 lines); results in `docs/sprint20/d1_parity.md`.
2. **`9079ad3873`** — D2 falsification record. Three independent warm runs at gate config: D2 = 45.3 ms, S19-tip = 31.87 ms → +42.3% regression. Stage attribution via `bench_boosting --stage-profile`: derivatives 0.5 ms (0%), **tree search 41.7 ms vs 29.4 ms (+42%)**, leaf estimation 2.5 ms (0%). 100% of regression lives in the histogram kernel. Root-cause analysis in `docs/sprint20/d2_results.md`.
3. **`78697fff79`** — D2b design + DEC-017 retirement (single commit per user option A). `docs/sprint20/d2b_design.md` (229 lines, 7 sections): abandon verdict for Sprint 20, Sprint 21 lever scoping (TG-count reduction via partition-batching), R8 projection ≥1.08× gate / ≥1.10× best, Sprint 21–22–23 pipeline (midpoint 1.27×, upper bound 1.46×; 1.5× not credibly reachable and flagged honestly). DECISIONS.md DEC-017 flipped from `ACTIVE-PENDING-D3` to `RETIRED — SUPERSEDED BY EMPIRICAL FALSIFICATION` with post-mortem banner and dispatch-shape root cause math. Original DRAFT-S20 text preserved below banner per DEC-013/14/15 pattern.

### Root cause — dispatch-shape mismatch (locked as campaign-level standing warning)

Toy kernel (Sprint 19 ablation): 1 TG × 256 threads × 50k docs single partition, ≈195 docs/thread. T3b's fixed per-TG overhead (1024-slot `atomic_uint` zero-init + writeback read = 8 memory ops per thread) amortizes to ≤1% of per-TG work; accumulation gain dominates; −84.4% valid for this shape only.

Production depth-6 dispatch: 13 feature groups × 63 partitions × 2 stats = **1638 TGs**. Per TG: ~50000 / 64 partitions ≈ 781 docs → 781 / 256 ≈ **3 docs/thread**. Fixed overhead now 8 memory ops vs 12 CAS ops = **67% of per-TG work**. CAS atomics cannot pipeline like simd_shuffle chains (each CAS is a read-modify-write with conditional retry that must see the result before the next iteration). Net: the fixed-cost structure of T3b is incompatible with the production partition count.

**Standing warning (campaign-level, encoded in DECISIONS.md DEC-017 post-mortem)**: toy-kernel ablations at single-TG root shape do not predict production partition-fragmented dispatch. Any future lever whose benefit comes from amortization across many docs/thread must be validated against the production TG × docs/thread shape *before* integration commit. This is the fifth analytical/toy-kernel model falsified this campaign — the pattern is now locked and the validation gate is mandatory for Sprint 21+.

### Sprint 20 exit gates

| Gate | Criterion | Status |
|---|---|---|
| G1 | `histogram_ms` ≤ 4 ms on gate | **FAIL** (measured +42%) |
| G2 | No 18-config regression > 5% | N/A (no kernel change shipped) |
| G3 | Parity 108/108 | **PASS** (D1 18/18 + 100/100 determinism) |
| G4 | `iter_total_ms` ≤ 10.5 ms | **FAIL** (tied to G1) |
| G5 | Non-histogram stages ≤ 10% | **PASS** (derivatives & leaf unchanged) |
| G6 | CI green | **PASS** (no kernel change) |

Sprint exits via empirical falsification, not a perf gate. PR #12 body records the gate table unchanged.

### R8 status — honest

- Sprint 20 target: ≥2.0× e2e (projected from toy −84.4%).
- Sprint 20 delivered: **0.704× gate** (+42% regression) — falsified before commit.
- **Sprint 21 target reset: ≥1.08× e2e** (TG-count reduction lever, scoped in d2b_design.md §3).
- **Campaign ≥1.5× e2e kept** per user's explicit decision. Sprint 21–22–23 pipeline midpoint 1.27×, upper bound 1.46×. **1.5× not credibly reachable on current kernel structure and is flagged honestly.**

### PR #12 — opened

`https://github.com/RR-AMATOK/catboost-mlx/pull/12` — stacked on PR #11 (Sprint 19). Ships the empirical record, not performance. Merge order: #9 → #10 → #11 → #12.

---

## Sprint 19 — T1 fuse-valid (DEC-016) shipped; DEC-014/015 REJECTED empirically; S19-13 envelope guard + exit gates (2026-04-17 → 2026-04-19, EXIT-GATES PASSED)

**Branch**: `mlx/sprint-19-hist-writeback`
**Campaign**: Operation Verstappen — battle 4 of 9 — L_accum lever (pivoted from L_writeback)
**Verdict**: T1 (DEC-016) shipped at −1.76% e2e on gate config, bit-exact, deterministic, guarded. R8 ≥1.07× NOT met (1.018× actual on gate / 1.033× best) — deferred to Sprint 20 via DEC-017 (T3b atomic-CAS).

### Day 4 evening (2026-04-19) — Exit gates + S19-13 envelope guard

Five exit-gate agents launched after commit `0f992cf863`. Two completed with empirically-backed sign-offs; two returned plan-only outputs (sandbox constraints); one flagged a BLOCKER on the T1 MSB-sentinel that was then fixed in S19-13.

**S19-07 code review — BLOCKER then resolved via S19-13.** Reviewer found that `compressedIndex[...] | VALID_BIT` in `kernel_sources.h` is unsafe whenever slot-0 holds a bin value ≥ 128. The packer (`csv_train.cpp::PackFeatures`) uses 8-bit slots, so slot-0 occupies bits [24..31] — bit 31 aliases bin 128. With default `MaxBins = 255` or the `bins = 128 + NaN offset` case, the path is reachable and `p_clean = p_s & 0x7FFFFFFFu` silently rewrites bins 128..255 → 0..127. The DECISIONS.md rationale claim "Safe at ≤128 bins because packed holds four 8-bit values in bits 24–30" was off by one.

**S19-13 fix** (landed in this session, single commit):
- `catboost/mlx/methods/histogram.cpp::ComputeHistogramsImpl` — computes `maxFoldCount` during foldCountsFlatVec construction and enforces `CB_ENSURE(maxFoldCount ≤ 127u, …)` before dispatch, with diagnostic message naming DEC-016 envelope and Sprint 20 DEC-017 as the wider-envelope follow-up. Include of `<catboost/libs/helpers/exception.h>` added.
- `catboost/mlx/tests/bench_boosting.cpp::DispatchHistogram` — mirror of the host-side guard via `std::fprintf(stderr, …)` + `std::exit(1)` (CB_ENSURE header is not available in the standalone bench build path).
- `catboost/mlx/tests/bench_boosting.cpp::GenerateSyntheticDataset` — `folds = isOneHot ? (…) : cfg.NumBins − 1` for ordinals. Aligns bench's Folds with real-quantize (`csv_train.cpp::Quantize` sets `folds = numBorders` for no-NaN features). Previously bench stored `Folds = cfg.NumBins` which over-reported by 1 and caused the guard to false-trip on `--bins 128` despite actual bin values staying in [0, 126].
- `catboost/mlx/kernels/kernel_sources.h:175–182` — inline comment rewritten to state the true invariant ("Safe ONLY when every feature's fold count ≤ 127") and cross-reference the host-side guard.
- `.claude/state/DECISIONS.md::DEC-016` — rationale + scope-limit corrected, S19-07 cross-reference added.

**S19-04 parity + determinism — PASS.** 18 configs × 3 runs each on `bench_boosting_ref` (kernel `020eacfb4c` pre-T1, HEAD elsewhere) vs `bench_boosting_t1` (HEAD + S19-13). All 18 produce bit-exact `BENCH_FINAL_LOSS` across ref and t1 (ulp = 0 in all cases, DEC-008 envelope satisfied at the strictest level). 100-run determinism on 50k/RMSE/d6/128b/seed42 returns a single unique loss (0.47740927 post-S19-13) — BUG-001 structural guard holds.

**S19-05 perf delta — PASS G2.** 3-run warm-mean deltas: best −3.23% (50k/Logloss/128); gate config (50k/RMSE/128) −1.76%; worst regression +1.39% at 1k/RMSE/128 (within 3-run noise floor ±2%). No config regresses > 5%. Delivered R8 factor on gate: **1.018×**. Honest accounting preserved. Per-config JSONs written to `.cache/profiling/sprint19/after_t1/*.json` (18 files).

**S19-08 security — PASS (APPROVED).** 5-commit diff audit: no kernel-source injection surfaces, no new buffer-size surfaces, no TOCTOU from EvalAtBoundary removal (MLX host-pointer ctor copies synchronously), no subprocess/eval/pickle in `check_histogram_gate.py`, no secrets, no dependency drift. One defense-in-depth suggestion ("add bins ≤ 128 assertion") — absorbed into S19-13.

**S19-09 post-fix MST — DEFERRED.** `xcrun xctrace` remains sandbox-blocked (same condition as S18-09). Analytical stage decomposition appended to `docs/sprint19/results.md §S19-09`: first-principles probe-A projection (−19.5% e2e) vs measured (−1.76%) is an ~11× over-projection, consistent with probe-A's 86.2% being a depth-0 single-TG attribution that does not multiply cleanly across 1575 TGs × 6 depths. Pattern: fifth analytical model under-predicts the projection-to-production gap. MST capture carried to Sprint 20 under Instruments availability.

**Docs landed:** `docs/sprint19/results.md` (executive summary + per-gate detail + honest R8 accounting).

### Day 4 (2026-04-19) — Path 3 close-out: Commits 1+2 shipped, A1 empirically dropped, parallel tracks

### Day 4 (2026-04-19) — Path 3 close-out: Commits 1+2 shipped, A1 empirically dropped, parallel tracks

**Three DEC-012 kernel commits landed** on `mlx/sprint-19-hist-writeback`:

1. **`77db8b5631`** — Commit 1: extract DEC-015 side-fix. Reverted col-major layout changes in `compressed_index.h`, `kernel_sources.h`, `bench_boosting.cpp`, `csv_train.cpp`. Kept the `DispatchHistogramBatched` per-group variable correction (`featureColumnIndices`+`numGroups` replacing scalar `featureColumnIdx`) in `histogram.cpp` — a pre-existing correctness fix that would have shipped regardless.
2. **`7387814dd6`** — S19-06 CI gate widening. `benchmarks/check_histogram_gate.py` updated from `sprint17/10k` to `sprint19/baseline/50000_rmse_d6_128bins.json`. Dropped min-reduction flag; sprint-neutral messages. Dry-run triggers at +6.1% delta.
3. **`020eacfb4c`** — S19-11 scope-reduced. Removed `TMLXDevice::EvalAtBoundary(result.LeafDocIds)` at `structure_searcher.cpp:738` — a no-op flush since MLX constructor copies data into the GPU buffer synchronously. Other 3 `EvalAtBoundary` calls on that path (lines 290, 609, 705) are legitimate pre-`.data<T>()` guard-syncs, left intact. Bit-exact pre/post at 50k/RMSE/d6/128b = 0.48047778 (3 runs each).
4. **`92f3832169`** — Commit 2: DEC-016 T1 fuse-valid simd_shuffle reduction. Pack the valid flag into the MSB of `packed` at load time (`packed |= VALID_BIT` where `VALID_BIT = 0x80000000u`); derive validity from `(p_s & VALID_BIT)` inside the src broadcast loop; mask via `p_clean = p_s & 0x7FFFFFFFu` before bin extraction. Drops one `simd_shuffle` per src iteration (3 → 2). **Measurements (50k/RMSE/d6/128b, 3-run warm mean):** pre-edit 32.47 ms, post-edit 31.73 ms → **−2.3% e2e**. **Parity bit-exact at 3 configs** (50k/RMSE=0.48047778, 10k/RMSE=0.48016092, 50k/MultiClass=0.94424933). Safe at ≤128 bins (packed holds four 8-bit values in bits [0..30]; bit 31 always zero on load).

**Commit 3 (DEC-014 A1 BATCH_DOCS=64) DROPPED** per plan clause "if not reproducible, drop":
- A1 variant added to `docs/sprint19/scratch/algorithmic/microbench_algorithmic.cpp` as `kA1Source`. Toy measurement (3 runs, post-T1): A1 vs T1 mean = **−1.9%** (noise-marginal; stdev ~1%).
- Production port (lo/hi slab state in lane registers, outer stride doubled, 2-slab inner shuffle loop). Parity bit-exact (0.48047778) but **warm-mean e2e +9.4% REGRESSION** (T1-only 31.7 ms vs T1+A1 34.7 ms, 3 runs each). Register pressure from lo/hi slab state dominates the halved outer-loop saving — AGX VGPR spill hypothesis.
- A1 reverted in `kernel_sources.h`; A1 variant kept in `microbench_algorithmic.cpp` for future reference.
- Full disposition: `docs/sprint19/scratch/algorithmic/a1_empirical_drop.md`.

**Pattern note: fourth analytical model falsified this sprint.** DEC-013 writeback plurality → SUPERSEDED. DEC-014 original gather sub-phase → INVALIDATED. DEC-015 col-major layout → REJECTED (measured 0.98× vs projected 2.13×). DEC-014 (A1 BATCH_DOCS=64) → REJECTED (measured +9.4% regression vs projected −4%). Sprint 19 lesson, locked: analytical reasoning about AGX cache/register behavior is unreliable — empirical micro-bench backing is required before committing any production kernel change, and toy-kernel signal must be validated against production integration before shipping.

**R8 accounting (honest per "do not soften" standing order):**
- R8 revised mid-sprint from aggressive 1.5–1.8× e2e to **≥1.07× e2e** after S19-01 ground-truth falsified the writeback plurality model.
- Delivered: **1.023× e2e** on 50k/RMSE/d6/128b via T1 alone.
- R8 NOT met. Deferred to Sprint 20 via DEC-017 T3b atomic-CAS (toy measured −84.4% accumulation; full DEC-008 parity sweep is the Sprint 20 D1 gate).

**Documentation landed (S19-10 technical-writer pass):**
- `docs/sprint19/algorithmic_ablation.md` — T0/T1/T2/T3/T3b ablation with measured toy-kernel deltas.
- `docs/sprint20/README.md` — Sprint 20 D1–D4 plan (T3b parity sweep, production integration, full-grid scaling, MultiClass drift analysis).
- DECISIONS.md updated: DEC-014 REJECTED, DEC-015 REJECTED, DEC-016 ACTIVE, DEC-017 DRAFT-S20.
- HANDOFF.md updated with close-out status and R8 deferral.

**Exit gates PENDING (parallel tracks, unblocked):** S19-04 parity grid + 100-run determinism, S19-05 18-config perf delta + 50k MST, S19-07 code review, S19-08 security pass, S19-09 post-fix MST.

---

## Sprint 19 — Accumulation Redesign (PIVOTED from Two-Phase Writeback) (2026-04-17, in progress)

**Branch**: `mlx/sprint-19-hist-writeback` (name reflects original scope — history over cosmetics)  
**Campaign**: Operation Verstappen — battle 4 of 9 — L_accum lever (pivoted from L_writeback)  
**Verdict**: IN PROGRESS

### Day 3 (2026-04-18) — DEC-015 col-major layout: correct, but performance gate not met (BLOCKER)

**S19-03 Commit 1 (DEC-015) — BLOCKED, NOT COMMITTED**

Implementation completed across 5 files. Status: parity-clean, determinism-clean, performance gate not met.

**Changes in working tree (not committed):**
- `catboost/mlx/gpu_data/compressed_index.h` — Added `CompressedDataTransposed_` member (`[numUi32PerDoc * numDocs]` uint32, col-major). Built in `Build()` via `mx::copy(mx::transpose(CompressedData_, {1,0}))` → `mx::reshape(..., {-1})` → `EvalAtBoundary`. One-time materialisation at load time. Added `GetCompressedDataTransposed()` accessor.
- `catboost/mlx/kernels/kernel_sources.h` — Changed compressedIndex load address from row-major `compressedIndex[docIdx * lineSize + featureColumnIdx]` to col-major `compressedIndex[featureColumnIdx * totalNumDocs + docIdx]`.
- `catboost/mlx/methods/histogram.cpp` — Rewrote `DispatchHistogramGroup()` (scalar per-group dispatch with broken variable name mismatch) to `DispatchHistogramBatched()` (correct batched dispatch matching `bench_boosting.cpp`/`build_verify_test.cpp`). Input names now match kernel body: `featureColumnIndices` (array) + `numGroups` (scalar). Passes `compressedDataTransposed` from `GetCompressedDataTransposed()`.
- `catboost/mlx/tests/bench_boosting.cpp` — Pre-computes `compressedDataTransposed` once before training loop; passes as parameter to `RunIteration()` → `DispatchHistogram()`.
- `catboost/mlx/tests/csv_train.cpp` — Pre-computes `compressedDataTransposed` once in `RunBoosting()` before training loop; passes to all 3 `DispatchHistogram()` call sites.

**Bugs fixed along the way (would have shipped regardless):**
- Pre-existing `histogram.cpp` kernel variable name mismatch: old code used `featureColumnIdx` (scalar 0-dim) but kernel body referenced `featureColumnIndices[groupIdx]` (array). Metal compile would have errored. Fixed as part of DEC-015 rewrite.
- Stale S18 parity reference (8/18 FAIL on first run): S18 parity table was from older D1c binary. Rebuilt reference binary from pre-DEC-015 stash. Result: 18/18 PASS, 0 ULP.
- Per-call transpose overhead: initial attempt placed `mx::copy(mx::transpose(...))` inside `DispatchHistogram()`, causing 6× GPU copies per iteration. Moved to pre-training-loop.

**Gate measurements (50k/RMSE/d6/128b, 5 warm runs each):**
- `bench_boosting_ref` warm mean: 33.7–34.2 ms (5 runs, σ ≈ 0.3 ms)
- `bench_boosting` (DEC-015) warm mean: 34.3–35.7 ms (5 runs, σ ≈ 0.5 ms)
- Speedup: **~0.98× e2e** (effectively 0, within noise)
- Expected from S19-01b model: 2.13× e2e (`histogram_ms` 15.43 → 4.17 ms)
- **Gate: NOT MET. BLOCKER.**

**Implication for S19-01b:** The analytical model (25 CL per 32-doc batch → 4 L2 stall rounds → 12.78 ms CI gather latency) is not validated by direct measurement. The DEC-015 layout change is the most direct test of that model's core prediction. The 0.98× result implies the model's latency estimate or access-pattern description is incorrect for this hardware. A hardware-controlled micro-benchmark (isolated kernel, swept N/lineSize, both layouts) is needed before the next intervention.

### Day 2 (2026-04-17) — Ground-truth falsifies writeback hypothesis; pivot to accumulation redesign

- **S19-01** (commit `d7ea14e28c`, `docs/sprint19/attribution.md`): Ground-truth Metal System Trace attribution on 50k/RMSE/d6/128b gate config. **Writeback = 0.79 ms (5%)** of steady-state `histogram_ms`. **Accumulation = 14.30 ms (93%)**. The "~15 ms writeback floor" from S18 was a mis-scaling of N=10k numbers to N=50k. R8 fired: writeback elimination projects 1.02–1.04× e2e (below the 1.5× aggressive target). Evidence correct; premise (writeback as plurality) falsified.
- **S19-02** (commit `fb05205ec0`, `docs/sprint19/ablation.md`): @research-scientist wrote a clean DEC-013 draft for two-phase writeback reduction. Variant (c) projected 3.0 ms reduction. Premise immediately invalidated by S19-01 — secondary effects ground truth does not support the projection. DEC-013 draft stands as historical artifact; not implemented.
- **R8 result**: writeback elimination → 1.02–1.04× e2e. Does not meet the 1.5× aggressive gate.
- **Ramos decision**: Option 2 — pivot Sprint 19 to accumulation redesign. Option 1 (ship weak writeback) and Option 3 (cleanup-only demote) rejected.
- **DEC-013 SUPERSEDED** by DEC-014 (see `.claude/state/DECISIONS.md`). DEC-013 entry preserved as audit trail.
- **DEC-014 DRAFT added**: accumulation redesign over writeback rewrite. 4 candidate variants (A: wider batch, B: coalesced TG staging, C: per-feature specialization, D: different ownership granularity). Projection: 30–50% `histogram_ms` reduction → 1.25–1.50× e2e. Locks at S19-02b close.
- **Day 2 kickoff**: @performance-engineer running S19-01b (accumulation sub-phase attribution); @research-scientist running S19-02b (accumulation redesign ablation + DEC-014 lock). Both in parallel.
- Sprint length bumped Day 5 → **Day 6** (pivot cost one day).
- G1 gate revised: `histogram_ms` −40% → **−30% min** (accumulation = 93%; 32% accumulation reduction ≈ 30% histogram_ms).

### Day 0 (2026-04-17) — Branch cut and scaffold

- S19-00: Branch cut from `mlx/sprint-18-hist-privhist-tile@463de74efa`. Sprint 18 after-profiles copied to `.cache/profiling/sprint19/baseline/` (18 JSONs, identical to S18 after). Gate config shift: 10k/RMSE/128b → **50k/RMSE/128b** (writeback lever has force at large N). Steady-state baselines — gate config: `histogram_ms` 15.52 ms (mean), `iter_total_ms` 21.12 ms. State files scaffolded (HANDOFF S19 rewrite, TODOS S19 section, DECISIONS DEC-013 placeholder, CHANGELOG S19 header). `docs/sprint19/README.md` scaffold created with campaign context, lever description, gates table, and projection table. DEC-013 DRAFT: two-phase on-chip reduction over batched-atomic (Ramos: "whatever is more robust"). PR #10 (Sprint 18) remains OPEN, unblocked.

---

## Sprint 18 — Histogram Accumulator Re-architecture (L1a) (2026-04-17)

**Branch**: `mlx/sprint-18-hist-privhist-tile`  
**Campaign**: Operation Verstappen — second structural kernel rewrite  
**Verdict**: **All gates PASS.** Cleared for merge.

- S18-00: Branch cut from `mlx/sprint-17-hist-tree-reduce`; Sprint 17 after-profiles copied to `.cache/profiling/sprint18/` as baselines.
- S18-01 (`attribution.md`): Ground-truth post-S17 attribution by linear regression on steady-state per-depth `histogram_ms` breakdown. Accumulation = 6.4 ms (27% of SS), zero-init = 4.0 ms (17%), D1c reduction = 3.0 ms (13%), writeback = 5.0 ms (21%), JIT = 5.3 ms. Plan's 52–59% accumulation estimate refuted (actual 27%); D1c had already eliminated the device-memory re-read cost conflated in the Sprint 16 baseline. Gate revised from ≥50% to ≥35% (≤18.7 ms) with Ramos Day-1 approval.
- S18-02: Ablation sweep L1a / L1b / L1c / L1d. L1a is the only variant with error-envelope gate clearance (worst case 17.3 ms vs 18.7 ms gate; L1b/c miss upper bounds). Ramos approved L1a Day 2. See `docs/sprint18/ablation.md`.
- S18-03 (`abc4c229f9` → `19fa5ce6cc`): L1a implementation. **Pivot**: initial kernel (commit `abc4c229f9`) failed all 18 parity configs by 6 orders of magnitude (BUG-S18-001). Two compounding structural flaws: (1) 1/32 doc-inclusion rate from stride/ownership mismatch; (2) 32× butterfly amplification from applying D1c's intra-SIMD `simd_shuffle_xor` butterfly to shared `simdHist` slots. Fixed at commit `19fa5ce6cc`: replaced accumulation with cooperative 32-doc batch loop using `simd_shuffle` broadcast (every doc contributes exactly once, no atomics); removed intra-SIMD butterfly entirely (`simdHist[g][bin]` is already the full per-SIMD-group sum). See `docs/sprint18/bug_s18_001.md` for post-mortem.
- S18-04a (initial, commit `abc4c229f9`): Parity FAIL — 4–20M ULP all 18 configs. Determinism PASS (consistent wrong answer).
- S18-04b (`7ab4e8e804`): Parity re-run on fixed kernel. **108/108 checkpoints bit-exact (ULP = 0 all loss types). 100/100 determinism runs bit-exact.** Cleaner than Sprint 17's 35/36 outcome. S18-G3 hard merge gate CLEARED.
- S18-05b (`da303866ef`): 18-config stage-profiler delta. Gate config (N=10k, RMSE, d6, 128b): **28.75 → 9.56 ms (-66.8%)**. S18-G1 (≥35%) **PASS** — 9.1 ms margin above target. Full range: -56.6% to -85.5%. All 18 configs improved, no regressions. Non-histogram stages all improved or unchanged. S18-G2, S18-G4 PASS. Sprint 19 floor visible: N=50k configs converge to ~15 ms (writeback-dominated). See `docs/sprint18/results.md`.
- S18-06: CI gate `benchmarks/check_histogram_gate.py` baseline updated to Sprint 17 after-JSON. S18-G5 PASS.
- S18-07: Code review PASS — barrier correctness, threadgroup-memory bound, stride-partition ownership.
- S18-08: Security audit PASS — no new exploitable surfaces.
- S18-09: Metal System Trace re-capture confirms `simdHist` on-chip residency; accumulation phase below 5 ms target. Appendix in `docs/sprint18/results.md`.
- S18-10: Docs — `bug_s18_001.md` post-mortem, `design.md` updated with final kernel structure and BUG-S18-001 root cause diagram, `ablation.md` post-ship actual vs projected section, `README.md` verdict banner, DEC-011 + DEC-012 in `DECISIONS.md`, `ARCHITECTURE.md` histogram section refreshed, `CHANGELOG.md` user-facing entry.

**Kernel change summary** (`catboost/mlx/kernels/kernel_sources.h`, commit `19fa5ce6cc`):
- `float privHist[HIST_PER_SIMD]` (4 KB/thread, 1 MB/threadgroup device-memory spill) → `threadgroup float simdHist[8][1024]` (32 KB, on-chip, at Apple Silicon limit).
- Zero-init loop eliminated (implicit for threadgroup memory).
- Per-thread stride accumulation → cooperative 32-doc batch loop with `simd_shuffle` broadcast and stride-partition ownership.
- D1c intra-SIMD butterfly removed (DEC-012). Cross-SIMD 8-term linear fold (DEC-009) unchanged.
- Barriers: 9 → 6 per dispatch.
- Reduction depth: γ_12 (S17) → γ_7 (S18). Higham bound improves ~7.2e-7 → ~4.2e-7.

**Sprint 19 carry-forward lever**: writeback (global-atomic) phase at ~15 ms for N=50k configs is now the floor. Batched-atomic writeback or shared-memory prefix-scan reduction of per-SIMD histograms before global writeback is the likely S19 L1. Scope constraint: results bounded to DEC-008 envelope (`approxDim ∈ {1, 3}`, `N ≤ 50k`, depth 6, 50 iterations).

---

## Sprint 17 — Histogram Tree Reduction (D1c) (2026-04-17)

**Branch**: `mlx/sprint-17-hist-tree-reduce`
**Campaign**: Operation Verstappen — headline performance lever
**Verdict**: **All gates PASS.** Cleared for merge.

- S17-00: Branch cut from master; 18 Sprint 16 baselines copied to `.cache/profiling/sprint17/` as before-snapshots.
- S17-01 (`5b4a8206bc`): D1c kernel — replaced 255-step serial threadgroup reduction in `catboost/mlx/kernels/kernel_sources.h:160–181` with 5-round `simd_shuffle_xor` intra-SIMD butterfly (xor 16/8/4/2/1) + 8-term linear cross-SIMD fold. Barriers 255 → 8, threadgroup memory 12KB (25% of 32KB limit). 95 lines changed.
- S17-02 (`1ce1ea6ee1`): Ablation verdict D1c over D1a (D1a structurally infeasible — ~9,216 barriers from orthogonal axes). Higham γ_8 FP32 bound analysis documented in `docs/sprint17/ablation.md`. Sprint 18 prior in `docs/sprint18/plan_prior.md`.
- S17-03 (`26fbabe932`): 18-config perf capture. `histogram_ms` reduced **89.4–93.0%** (308.20→28.75 ms on gate config, -90.7%). `iter_total_ms` reduced 84.4–92.4%. Secondary stages (suffix_scoring, leaf_sums, cpu_readback) improved 10–30% from pipeline backpressure unblocking. Full table in `docs/sprint17/results.md`.
- S17-04 (`26fbabe932`): Parity matrix — 35/36 checkpoints bit-exact across 18 configs × 6 checkpoints. Final-iteration ulp=0 for all 18 configs. One transient 17-ulp spike at iter=10 of 10k/MultiClass/32 healed to 0 by iter=20 — within Higham γ_8 bound. See `docs/sprint17/parity_results.md`.
- S17-05 (`afded6c4e5`): CI gate `benchmarks/check_histogram_gate.py` (15 tests, all pass). `.github/workflows/mlx-perf-regression.yaml` wired to block >5% histogram regression.
- S17-06: Code review PASS. Three should-fix items addressed in a follow-up: (1) stale "left for S17-06 code review" comment → "deferred to Sprint 18"; (2) scope caveats added to results.md and parity_results.md bounding findings to `approxDim ∈ {1,3}`, `N ≤ 50k`; (3) DECISIONS.md updated with DEC-008 (parity envelope), DEC-009 (linear 8-term choice), DEC-010 (Sprint 18 L1 lever).
- S17-07: Security audit PASS — no exploitable findings, 2 info-level hardening suggestions (SHA-pin actions, add `permissions: read`). Metal shader bounds provable from compile-time constants; CI gate parser uses only argparse+json.load; workflow is `pull_request` (safe) with no secret interpolation.
- **Sprint 18 headline lever identified**: steady-state histogram is still ~175× above memory-bandwidth floor. `privHist[1024]` register spill is the top ceiling. Tiled accumulation (256-lane × 4-pass fold) is the Sprint 18 L1.

## Sprint 16 — Performance Diagnosis & First Cut (2026-04-15, in progress)

**Branch**: `mlx/sprint-16-perf-diagnosis`
**Campaign**: Operation Verstappen — performance domination push
- Restored `.claude/state/` files (HANDOFF, TODOS, MEMORY, DECISIONS, CHANGELOG-DEV)
- S16-05: Extended `benchmarks/bench_mlx_vs_cpu.py` with `--bins`, `--mlx-stage-profile`, `--save-baseline` flags; CPU-parity runner with side-by-side JSON; new `ParityResult` data class; JSON schema with `meta`+`runs[]` including `bins`, `stage_timings`, `cpu_baseline`, `mlx_baseline`
- S16-06: Created `.github/workflows/mlx-perf-regression.yaml` — CI gate on 50k RMSE 128-bin benchmark, 5% threshold, step summary table, `macos-14` only
- S16-02 (baseline support): Regenerated `.cache/benchmarks/sprint16_baseline.json` with accurate Sprint 15 numbers — old phase_a data was stale (from early-sprint code). True MLX/CPU gap is 100–300x, not 10–24x
- S16-07: Sync-storm elimination — removed all 18 `EvalNow` from `pointwise_target.h`, 3 per-depth `EvalNow` from `structure_searcher.cpp`, added `EvalAtBoundary` at iteration boundary. Validated: bit-exact loss across 9 test combos, zero perf regression
- S16-08: Numerical parity validated — RMSE/Logloss/MultiClass × 1k/10k/50k all bit-exact between Sprint 15 and Sprint 16 binaries
- Fixed `bench_mlx_vs_cpu.py` bug: `n_bins=` → `bins=` (API param name mismatch)
- Key finding: per-iteration cost barely scales with N (300ms at 1k, 323ms at 10k, 487ms at 50k with 50 features) — confirms histogram occupancy (`maxBlocksPerPart=1`) as dominant bottleneck
- Stage profiler code drafted by @performance-engineer (pending write to disk)

---

## Sprint 15 — Upstream Submission Prep and Release Packaging [from git log]

**Commit**: `74f2ba63d4` | **Merge**: `165f2bc706`
- Upstream submission preparation
- Release packaging

## Sprint 14 — CI/CD Workflows and Performance Benchmarks [from git log]

**Commit**: `7b36f60a82` | **Merge**: `97a069c93a`
- CI/CD workflow setup
- Performance benchmark infrastructure

## Sprint 13 — Library Path Feature Parity [from git log]

**Commit**: `f1d6b00b20` | **Merge**: `1a2dd61ea2`
- Library path feature parity with CPU CatBoost

## Sprint 12 — Docs Refresh, Ranking Hardening, Upstream Prep [from git log]

**Commit**: `0ec8754c82` | **Merge**: `46ba563172`
- Documentation refresh
- Ranking hardening
- Upstream prep
- BUG-007 found: nanobind path doesn't sort group_ids

## Sprint 11 — Nanobind Python Bindings [from git log]

**Commit**: `3722eb9f95` | **Merge**: `7f7d540276`
- Nanobind in-process GPU training bindings
- CUDA coexistence specification

## Sprint 10 — Lossguide, Model Versioning, PyPI 0.3.0 [from git log]

**Commit**: `d8e3e7ba7b` | **Merge**: `8641eee078`
- Lossguide grow policy (best-first leaf-wise construction)
- Model format versioning (format_version=2)
- PyPI packaging
- User-facing README and quickstart
- `bench_mlx_vs_cpu.py` benchmark script
- BUG-006 fix: scope max_leaves validation to Lossguide only

## Sprint 9 — Depth>6, Depthwise Policy, MLflow, 16M Fix [from git log]

**Commit**: `b8a0ab258a` | **Merge**: `445f55c20a`
- `max_depth > 6` via chunked multi-pass leaf accumulation
- Depthwise grow policy (per-leaf splits at each depth level)
- Deferred histogram EvalNow — reduced CPU-GPU syncs to 5 remaining
- Optional MLflow logging
- bench_boosting CI regression check
- int32 accumulator in ComputePartitionLayout (DEC-003)
- BUG-005 fix: validate grow_policy in _validate_params
- 66 new tests, 789 total

## Sprint 8 — Housekeeping, Poisson/Tweedie/MAPE Losses [from git log]

**Commit**: `1d1e25321f` | **Merge**: `9d9d645430`
- Poisson, Tweedie, MAPE loss functions (library path)
- BUG-004 fix: strip variance_power= prefix in loss param validation
- 39 QA tests for new losses

## Sprint 7 — Multiclass Fuse, Partition Kernel Output, BUG-002 [from git log]

**Commit**: `cd239c84d1` | **Merge**: `7b483ad631`
- Fused multiclass leaf computation — eliminated K EvalNow calls per iteration
- Partitions output from tree_applier kernel — deleted O(depth) recompute
- BUG-002 fix: threshold comparison in bench_boosting

## Sprint 6 — CI Infra, bench --onehot, Tree Applier Metal Kernel [from git log]

**Commit**: `44ac16d66d` | **Merge**: `c7b478f352`
- Tree applier ported to Metal kernel dispatch
- bench_boosting `--onehot` flag
- CI workflow: bench_boosting compile step
- ARCHITECTURE.md deep-dive added
- CONTRIBUTING.md, CHANGELOG.md, Known Limitations docs

## Sprint 5 — BUG-001 Fix + Lint Cleanup [from git log]

**Commit**: `ee617527e3` | **Merge**: `0d2e97f914`
- Deterministic suffix-sum scan (BUG-001 fix)
- Ruff lint cleanup across test and source files
- Parallel SIMD scan for suffix_sum_histogram
- bench_boosting library-path harness
- 16M-row float32 limit documented in DECISIONS.md

## Sprint 4 — GPU Partition Layout [from git log]

**Commit**: `591822a51e` | **Merge**: `fff9f02b7b`
- ComputePartitionLayout ported to GPU
- 16M-row float32 safety guard
- Sprint branch convention established (DEC-004)

## Sprint 3 — Leaf Estimation, Score Splits, Loss Functions [from git log]

**Commits**: `928c7ff4d1` through `38f963cd4a`
- MAX_LEAVES=64 runtime enforcement
- Bin-to-feature lookup table for score_splits
- Fused leaf sum dispatch
- MAE/Quantile/Huber losses wired into dispatch
- Loss function validation tests

## Sprints 0–2 — Foundation [from git log]

**Commits**: `b78d428f58` through `edf8a97ba5`
- Initial Metal kernels for histogram, scoring, leaf accumulation
- Multi-block histogram dispatch (1.2x speedup)
- Feature group batching (1.6x speedup)
- In-process tree evaluation for predict (5–25x faster)
- CBMX binary format (200x faster I/O)
- MVS sampling, base prediction (boost from average)
- Input validation, accuracy bug fixes
- Multiclass fix (off-by-one, 2-class crash)
- random_strength, performance profiling
