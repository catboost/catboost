# Sprint 24 D0 — DEC-023 Close-out: Full Diagnostic History + v5 Ship

**Branch**: `mlx/sprint-24-dec023-fix`
**Date**: 2026-04-21
**Tip commit**: `784f82a891`
**Status**: RESOLVED — DEC-023 closed via v5 accumulation-topology match; all four acceptance criteria PASS; R8 collapses from 1.90× to 1.01×

---

## §1 Background

DEC-023 was opened in Sprint 23 D0 during the T2 scratch→production promotion parity sweep.
Config #8 (N=10000/RMSE/128b/depth=6/iters=50/seed=42) produced bimodal output, ~50/50
between 0.48231599 and 0.48231912 (105 ULP gap). Gate config #14 (N=50000/RMSE/128b) was
unaffected — 100/100 deterministic at 0.47740927. The other 16 DEC-008 configs were also clean.

Root cause was classified as: features 1-3 `atomic_fetch_add_explicit(memory_order_relaxed)` on
`device atomic_float` in `kT2AccumSource` — FP32 addition is non-associative, and
non-deterministic Metal GPU thread scheduling produces 1-2 ULP drift in histogram bins. That drift
can flip a near-tie GAIN comparison early in training and cascade to 105+ ULP in the final loss
at iters=50. Config #8's partition sizes and bin distributions create exactly the near-tie
sensitivity that makes the cascade fire. Config #14's larger partitions resolve additions in a
consistent order and are not sensitive.

Sprint 24 D0 scope: fix DEC-023 and re-run the 18-config parity sweep. Sprint 24 was also
supposed to run a championship benchmark suite (§3 of the sprint README). That suite was never
started — the campaign retreated at DEC-023.

### Fix-option matrix (from sprint README)

| Option | Mechanism | Deterministic | Perf risk |
|--------|-----------|:-------------:|:---------:|
| 1 — TG-local reduce + single-thread commit | Each TG accumulates feat 1-3 into threadgroup memory; single thread writes to global | YES | Low |
| 2 — Int-atomic fixed-point | `atomic_uint` with fixed-point encoding; deterministic by integer arithmetic | YES | Medium |
| 3 — Kahan/Neumaier compensated summation | Running compensation per bin | NO | Low |

Kill-switch: if the chosen fix degrades `hist_ms(T2)/hist_ms(T1)` at gate config below 0.45×,
escalate to structural redesign rather than shipping a fix that collapses the 1.90× record.

---

## §2 Path 5 Attempt: Deterministic T2-Sort Prefix-Sum Scatter + Int64 Fixed-Point — FALSIFIED

### Mechanism

Path 5 combined two sub-fixes:

1. **Deterministic T2-sort**: replaced the non-deterministic parallel cursor scatter (S-3 race,
   within-bin doc order in `sortedDocs[]` varies run-to-run due to `threadgroup atomic_uint`
   cursor advances from 256 threads) with a deterministic prefix-sum scatter. Goal: make
   feature-0's bin-range scan produce a fixed, reproducible within-bin doc ordering.

2. **Int-atomic fixed-point for features 1-3**: replaced `atomic_fetch_add` on float with
   `atomic_fetch_add` on `atomic_uint` using a 24.8 fixed-point encoding. Goal: eliminate
   S-5's FP non-associativity for features 1-3.

The intent was to address both S-3 and S-5 simultaneously while preserving T2-sort's structural
advantage (feature-0 bin-range scan over sorted docs — the source of T2's 0.317× hist_ms ratio).

### Falsification result

All Path 5 variants that retained feature-0's bin-range scan over `sortedDocs` pinned to
Value B (0.48231912) deterministically across 10/10 runs at config #8 — 105 ULP off T1's
Value A (0.48231599).

**Root cause of the persistent 105 ULP offset**: The bin-range scan over `sortedDocs` applies
FP additions in within-bin doc order (determined by the sort). T1's accumulation applies FP
additions in SIMD-group batch order, driven by `docIndices` and the 8-SIMD linear fold. These
two reduction topologies are algebraically different — even when summing identical values, the
FP addition order differs, and at 50 iterations the 1-2 ULP per-bin difference cascades to 105
ULP via the near-tie GAIN flip at config #8.

Integer fixed-point accumulation eliminated the S-5 non-associativity correctly — features 1-3
became deterministic. But determinism is not the same as matching T1. Feature-0's bin-range scan
still used a different reduction topology than T1, so the result was deterministic at Value B
rather than bimodal between A and B.

Making the T2-sort ordering more deterministic (prefix-sum scatter vs cursor scatter) fixed
S-3's non-determinism but did not change the reduction topology. Any design where feature-0
reads from `sortedDocs` uses a reduction topology incompatible with T1's SIMD accumulation.

**Conclusion**: Path 5 is FALSIFIED. The 105 ULP offset is not an associativity bug — it is a
reduction-topology structural difference between T2-sort's scan and T1's SIMD fold. No
within-design variation of Path 5 that retains `sortedDocs` can produce Value A.

The archived diagnostic record for Path 5's false-positive halt is at branch
`archive/s24-d0-v5-retreat`.

---

## §3 Path X: CPU Anchor Measurement — INCONCLUSIVE

### What was measured

Ran CPU CatBoost at config #8 (N=10000/RMSE/128b/depth=6/iters=50/seed=42) with byte-matched
synthetic data. CPU landed at 0.068, approximately 24M ULP from both Value A (0.48231599) and
Value B (0.48231912).

### Why this is inconclusive

`bench_boosting` is not a CatBoost conformance harness. It is a simplified GPU-kernel speed
benchmark:

- No `boost_from_average` initialization (CatBoost default: initializes leaf values to the mean
  of the target, shifting all predictions before iteration 1).
- Intentionally simple split loop (oblivious tree with inline `RunIteration`, not
  `GreedyTensorSearch`).
- Synthetic data with a fixed random seed rather than a real dataset.

The 24M ULP gap between CPU=0.068 and GPU Values A/B is expected — the two harnesses compute
different things. CPU CatBoost's 0.068 is the loss value for a full CatBoost training run on
synthetic data with `boost_from_average` enabled; A/B are the loss values for the bench_boosting
GPU kernel harness on the same data without that initialization.

### Reframe

**R8 1.90× is a GPU-kernel-speed metric, not a training-correctness metric.**

`bench_boosting` measures how fast the GPU histogram kernel is relative to itself across
configurations. The T1 reference value at config #8 (0.48231599 = Value A) is the declared
parity anchor for the DEC-008 envelope. It is the anchor because T1 is the production kernel
at the start of Sprint 22, not because it matches CPU CatBoost. Path X does not falsify Value A
as the anchor — it confirms that `bench_boosting` is a standalone kernel harness.

---

## §4 Off-by-One Cascade Retest — FALSE POSITIVE

### Hypothesis

A Path X diagnostic agent proposed that config #8's 105 ULP cascade might be an artifact of an
off-by-one between the scoring kernel and `ApplySplitToPartitions`: the scoring kernel was
described as using "bin ≥ b right" while the apply path was described as using "bin > b right".

### Retest result

Code audit confirmed both paths encode `raw_bin > splitIdx` (strict greater than), consistent
with CatBoost canonical `IsTrueHistogram(bucket, splitIdx) = bucket > splitIdx`
(`catboost/libs/model/split.h`). The apparent discrepancy was a coordinate-system labeling
mismatch — the scoring kernel's "histogram slot b" equals raw_bin > b (because the suffix-sum
kernel transforms the histogram so slot b holds the sum for raw bins b+1..folds), and the apply
path's "raw_bin > b" is the same condition expressed in raw bin units. No functional mismatch
exists. The off-by-one is not present in the code.

**Full diagnostic trace at**: `docs/sprint24/d0_offby1_cascade_retest.md`.

---

## §5 v5: All-Feature T1-Style Accumulation — CORRECT FIX

### Rationale

Path 5 established that any T2-accum design retaining feature-0's bin-range scan over
`sortedDocs` produces Value B. The only way to produce Value A deterministically is to match T1's
reduction topology exactly: all features must use T1-style SIMD-shuffle accumulation reading from
`docIndices`.

v5 implements exactly this. T2-accum is rewritten so all four features (0-3) use T1-style
SIMD-shuffle accumulation reading from `docIndices`. T2-sort is removed from the dispatch path
(no longer needed — feature-0 no longer scans `sortedDocs`).

### Kernel changes (v5 — commit `784f82a891`)

**`kT2AccumSource`**:
- All features 0-3: T1-style SIMD-shuffle + linear fold + writeback
- Feature-0: bin mask `(p_clean >> 24u) & 0x7Fu` (7-bit, matching T2-sort step 1)
- `sortedDocs` and `binOffsets` removed from kernel inputs
- `partSizes` added to supply `totalDocsInPart` directly
- `simdHist[8][1024]` = 32 KB threadgroup memory; SIMD-owned stride accumulation
- Linear fold across 8 SIMD groups per feature (f=0..3); writeback via atomic_fetch_add

**`DispatchHistogramT2`**:
- `GetT2SortKernel()` removed from anonymous namespace (T2-sort no longer dispatched)
- `GetT2AccumKernel()` kernel name: `t2_accum_s24d0_v5` (invalidates prior cache entries)
- `sortedDocs` and `binOffsets` removed from accum input list
- `partSizes` added to accum input list
- Single-kernel dispatch: only T2-accum fires

### Why ULP=0 is structural, not empirical

v5's T2-accum executes the identical FP computation as `kHistOneByteSource` (T1): same SIMD-group
batch order reading from `docIndices`, same linear fold across 8 SIMD groups, same writeback.
Bit-exact agreement follows from execution identity — it is not a coincidence of a particular
config or seed.

---

## §6 Acceptance Criteria Results

### G1 — Config #8: 10/10 deterministic

```bash
for i in $(seq 1 10); do
  /tmp/bench_s24 --rows 10000 --features 50 --classes 1 \
    --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42
done
```

| Runs | Distinct values | Value | ULP vs T1 |
|------|----------------|-------|-----------|
| 10/10 | 1 | 0.48231599 | **0** |

**PASS**

### G2 — 18/18 DEC-008 parity sweep (5 runs each)

| # | N     | Loss       | Bins | T1 ref       | T2 v5        | ULP | Runs | Verdict |
|---|------:|:-----------|-----:|-------------:|-------------:|----:|-----:|:-------:|
| 1 |  1000 | RMSE       |   32 | 0.40689126   | 0.40689126   |   0 |  5/5 | **PASS** |
| 2 |  1000 | RMSE       |  128 | 0.46936080   | 0.46936080   |   0 |  5/5 | **PASS** |
| 3 |  1000 | Logloss    |   32 | 0.34161490   | 0.34161490   |   0 |  5/5 | **PASS** |
| 4 |  1000 | Logloss    |  128 | 0.61407095   | 0.61407095   |   0 |  5/5 | **PASS** |
| 5 |  1000 | MultiClass |   32 | 0.61065382   | 0.61065382   |   0 |  5/5 | **PASS** |
| 6 |  1000 | MultiClass |  128 | 0.99084771   | 0.99084771   |   0 |  5/5 | **PASS** |
| 7 | 10000 | RMSE       |   32 | 0.44631991   | 0.44631991   |   0 |  5/5 | **PASS** |
| 8 | 10000 | RMSE       |  128 | 0.48231599   | 0.48231599   |   0 |  5/5 | **PASS** |
| 9 | 10000 | Logloss    |   32 | 0.30072498   | 0.30072498   |   0 |  5/5 | **PASS** |
|10 | 10000 | Logloss    |  128 | 0.60412812   | 0.60412812   |   0 |  5/5 | **PASS** |
|11 | 10000 | MultiClass |   32 | 0.57359385   | 0.57359385   |   0 |  5/5 | **PASS** |
|12 | 10000 | MultiClass |  128 | 0.95665115   | 0.95665115   |   0 |  5/5 | **PASS** |
|13 | 50000 | RMSE       |   32 | 0.44676545   | 0.44676545   |   0 |  5/5 | **PASS** |
|14 | 50000 | RMSE       |  128 | 0.47740927   | 0.47740927   |   0 |  5/5 | **PASS** |
|15 | 50000 | Logloss    |   32 | 0.30282399   | 0.30282399   |   0 |  5/5 | **PASS** |
|16 | 50000 | Logloss    |  128 | 0.60559267   | 0.60559267   |   0 |  5/5 | **PASS** |
|17 | 50000 | MultiClass |   32 | 0.56538904   | 0.56538904   |   0 |  5/5 | **PASS** |
|18 | 50000 | MultiClass |  128 | 0.94917130   | 0.94917130   |   0 |  5/5 | **PASS** |

**18/18 PASS — all ULP=0, all 5/5 deterministic.**

### G3 — Gate config #14: 100/100 deterministic

```bash
for i in $(seq 1 100); do
  /tmp/bench_s24 --rows 50000 --features 50 --classes 1 \
    --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42
done
```

| Runs | Distinct values | Value |
|------|----------------|-------|
| 100/100 | 1 | 0.47740927 |

**PASS**

### G4 — hist_ms ratio ≥ 0.45× (kill-switch)

Measured at gate config with `--per-kernel-profile` (3 independent sessions):

| Session | T2 v5 hist_ms | T1 ref hist_ms | Ratio   |
|---------|:-------------:|:--------------:|:-------:|
| 1       | 20.277 ms     | 21.639 ms      | 0.937×  |
| 2       | 21.150 ms     | 21.639 ms      | 0.978×  |
| 3       | 20.821 ms     | 21.639 ms      | 0.962×  |
| **Mean** | **20.749 ms** | **21.639 ms**  | **0.959×** |

Ratio 0.959× >> 0.45× kill-switch threshold.

**PASS** — kill-switch does not fire.

---

## §7 R8 Consequence: 1.90× → 1.01×

### Numbers

| Metric | S23 D0 (T2 v4, non-deterministic) | S24 D0 (T2 v5, deterministic) |
|--------|:---------------------------------:|:-----------------------------:|
| hist_ms (gate config) | ~6.85 ms (0.317× T1) | ~20.75 ms (0.959× T1) |
| iter_total_ms warm mean (gate config) | ~17.3 ms | ~33–35 ms |
| e2e speedup vs T1 iter_total (33.96 ms) | **1.90×** | **~1.01×** |

**The 1.90× R8 record is superseded. Honest current position: ~1.01×.**

The Verstappen ≥1.5× gate criterion is not met at S24 D0 v5. This is the correct honest record.

### Why the speedup collapsed

T2's 0.317× hist_ms ratio derived from T2-sort's sort-by-bin pre-pass: by sorting docs by their
feature-0 bin, the bin-range scan reads consecutive docs within the same bin, enabling coalesced
memory access and eliminating the SIMD-shuffle broadcast overhead for feature-0. The elimination
of that shuffle chain was T2's structural advantage.

v5 removes the bin-range scan and replaces it with T1's SIMD-shuffle pattern. The T2-accum
kernel now executes the same operations as T1 for all four features. The 0.317× ratio
(T2 vs T1) becomes ~0.959× (T2 vs T1 at essentially the same speed, with trivially different
dispatch paths).

The sort-by-bin speedup was contingent on feature-0 using a reduction topology different from
T1's. Making T2 deterministic requires matching T1's topology exactly. These two requirements
are in irreconcilable conflict: T2-sort's bin-range scan cannot reproduce T1's SIMD partial-sum
fold order while being faster than it.

### Verstappen campaign status

**Operation Verstappen: battle 9 CLOSED. ≥1.5× gate failed retroactively at S24 D0. R8 post-fix: 1.01×.**

The 1.90× record that cleared the Verstappen gate was predicated on the bimodal T2 kernel.
Making T2 deterministic collapses R8 to 1.01×. This is an honest retreat, not a managed
tradeoff.

---

## §8 Forward Work: DEC-026 (S25 Research Track)

The 1-2 ULP/bin accumulation-topology difference between T2's sort-based scan and T1's SIMD fold
cascades to 105 ULP at iters=50 through a near-tie GAIN flip. This cascade is config-specific:
only config #8 has a GAIN comparison close enough to flip on 1-2 ULP input differences.

A deterministic tiebreak at near-tie GAIN comparisons — lexicographic on (featureIdx, binIdx)
when `|GAIN_A - GAIN_B| < ε` — could prevent the cascade flip. If the tiebreak allows T2's
Value B to produce the same tree structure as T1's Value A at every iteration, T2's structural
speedup becomes shippable again.

This is a research question:
- **Epsilon calibration**: what ε catches genuine near-tie flips without false-positive
  tiebreaks at legitimate GAIN gaps?
- **Model-quality validation**: does the tiebreak change tree structure in a way that degrades
  AUC/RMSE across the full DEC-008 envelope?
- **T2 rebuild**: T2-sort + int-atomic fixed-point accumulation on top of the tiebreak.

DEC-026 is opened for Sprint 25 research. It is not a guaranteed delivery — it is a research
question with falsification checkpoints. See `DECISIONS.md DEC-026`.

---

## §9 Files Modified

- `catboost/mlx/kernels/kernel_sources.h` — `kT2AccumSource` v5 (all features T1-style; `kT2SortSource` retained for reference)
- `catboost/mlx/methods/histogram_t2_impl.cpp` — `GetT2SortKernel()` removed; `GetT2AccumKernel()` updated to v5; `DispatchHistogramT2` updated (T2-sort dispatch removed; `partSizes` added to accum inputs)
- `catboost/mlx/methods/histogram.h` — doc comment updated for v5 architecture
