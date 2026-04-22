# DEC-026-G1: ε-Calibration Sweep — Verdict Document

**Checkpoint:** G1 / Checkpoint 4 (analyzer + verdict)
**Date:** 2026-04-21
**Sweep:** 18 configs × 5 runs × 2 kernels = 180 runs, 5 min 04 s wall
**Status:** **FALSIFIED**

---

## 1. Summary

The G1 sweep ran 50-iter training to completion on 18 configs covering
`{N=1k, 10k, 50k} × {bins=32, 128} × {classes=1, 2, 3}` with two kernels:
**T1** (DEC-008 reference; SIMD histogram + Kingpin K=5 topk + sort-based
serial tiebreak) and **Path 5** (T2-sort serial scatter + int-atomic
fixed-point accumulation with SCALE=2³⁰). All 180 runs are deterministic
(5/5 seeds produce bit-identical `final_loss`), T1 reproduces all 18
DEC-008 reference losses exactly, and 17/18 configs have T1 ≡ Path 5
`final_loss` bit-exact. Config #8 (N=10 k, bins=128, 1 class) is the sole
divergence: T1 = 0.48231599, Path 5 = 0.48231912 (≈ 105 ULP in float32).

Path 5 **cannot be ε-gated safely.** ε_min = 2.200 × 10⁻³ (config #8,
iter 45, split-bin flip 97 → 77, top-5 disagreement in T1's own gain
space) while ε_max = **0** (zero-gain ties at pure/terminal nodes in
configs 1, 2, 8, 14) and the positive-gap floor ε_max⁺ = 1.043 × 10⁻⁷
(config #1, iter 40, depth 3). Required ratio ≥ 2.0; observed
ratio (positive) = 4.7 × 10⁻⁵. **ε threading is impossible by more than
four orders of magnitude even under the optimistic positive-only
reading.** Decisively FALSIFIED — the killer is intrinsic to the T1/Path 5
split-gain disagreement at iter 43 depth 1 of config #8, where the two
top candidates in T1 are separated by only 5.96 × 10⁻⁸ (the same near-tie
is where Path 5 flips).

**Forward pointer:** DEC-026 FALSIFIED. R8 remains at 1.01× midpoint.
G2 (ε-gated Path 5 implementation) is CANCELLED. Path forward options
are documented in §9.

---

## 2. Methodology

### 2.1 Option A gain-dump kernel

Both T1 and Path 5 were augmented with an Option A dump kernel that, at
every NodePlaceholders evaluation, emits the top-5 (feat, bin, gain)
candidates by gain plus a rank=255 "min-non-winner" sentinel. The dump
kernel runs at every (iter, depth_level), producing 300 node evaluations
per 50-iter × depth=6 run. One run = 2,067 – 2,100 rows depending on
how often rank-255's sentinel coincides with a top-5 entry (it's then
emitted once instead of twice).

Schema:
```
config_id, run_id, kernel, iter, depth_level, feat, bin, gain, is_winner, rank
```

### 2.2 Path 5 reconstruction

Path 5 reconstructs T2's split-gain decisions via:

- **Feats 1..3:** T2-sort serial scatter over sortedDocs, int-atomic
  fixed-point accumulation with SCALE = 2³⁰ (quantizes float32 residuals
  to ≈ 9.3e-10 LSB).
- **Feat 0:** bin-range scan over sortedDocs (no atomic path — direct
  prefix-sum reduction).

SCALE = 2³⁰ places the int-atomic noise floor at ~1e-9 relative to the
largest accumulated magnitude, which is near the float32 epsilon. The
quantization is NOT the source of the flip; see §8.

### 2.3 Determinism pre-verification

Before running analysis, all 180 runs were checked for bitwise
reproducibility: 5/5 seeds produce identical `final_loss` on every
(config, kernel) pair. This matches the intent of G1: any non-determinism
would invalidate ε-calibration because flip events would depend on
scheduling.

### 2.4 180-run protocol

- 18 configs enumerated in `run_g1_sweep.sh` (N × bins × classes matrix).
- 5 seeds per (config, kernel) to lock in determinism.
- All runs emit per-(iter, depth_level) trace files to
  `benchmarks/sprint25/g1/results/traces/`.

### 2.5 Analyzer (`analyze_g1.py`)

For every node (config, run, iter, depth_level):
- Compare T1 rank=0 winner (feat, bin) with Path 5 rank=0 winner.
- **Earliest-depth-per-iter rule** (Ramos): only the SMALLEST depth_level
  with a disagreement in each (config, run, iter) is counted as a flip.
  Deeper disagreements in the same iter are downstream cascade.
- `flip_gap_t1 = gain_T1(T1_winner) − gain_T1(Path5_winner)` measured in
  T1's own gain space. If Path 5's winner is not in T1's top-5,
  `flip_gap_t1` is a LOWER BOUND (distance to T1's rank-4 entry) and
  the row is flagged `path5_winner_outside_t1_topk = true`.
- `legit_gap = gain_T1(rank=0) − gain_T1(rank=1)` for every NON-flipping
  node at depth_level STRICTLY LESS THAN the first-flip depth in that
  iter (or all depths if no flip).

ε threading:
- `ε_min = max(flip_gap_t1)` over first-flip rows.
- `ε_max = min(legit_gap)` over non-flipping rows.
- `ε_max⁺ = min(legit_gap > 0)` — strictly-positive floor.
- PASS if `ε_max / ε_min ≥ 2.0`; recommended ε = √(ε_min · ε_max).
- FALSIFY otherwise; identify killer row.

---

## 3. Sweep summary table

All losses deterministic 5/5. T1 matches DEC-008 reference 18/18. Agree
= T1 `final_loss` == Path 5 `final_loss`.

| cid | N     | bins | C | T1 loss       | Path 5 loss    | T1 det | P5 det | T1≡ref | agree |
|-----|-------|------|---|---------------|----------------|--------|--------|--------|-------|
| 1   | 1000  | 32   | 1 | 0.40689126    | 0.40689126     | 5/5    | 5/5    | YES    | YES   |
| 2   | 1000  | 128  | 1 | 0.46936080    | 0.46936080     | 5/5    | 5/5    | YES    | YES   |
| 3   | 1000  | 32   | 2 | 0.34161490    | 0.34161490     | 5/5    | 5/5    | YES    | YES   |
| 4   | 1000  | 128  | 2 | 0.61407095    | 0.61407095     | 5/5    | 5/5    | YES    | YES   |
| 5   | 1000  | 32   | 3 | 0.61065382    | 0.61065382     | 5/5    | 5/5    | YES    | YES   |
| 6   | 1000  | 128  | 3 | 0.99084771    | 0.99084771     | 5/5    | 5/5    | YES    | YES   |
| 7   | 10000 | 32   | 1 | 0.44631991    | 0.44631991     | 5/5    | 5/5    | YES    | YES   |
| **8** | **10000** | **128** | **1** | **0.48231599** | **0.48231912** | **5/5** | **5/5** | **YES** | **NO** |
| 9   | 10000 | 32   | 2 | 0.30072498    | 0.30072498     | 5/5    | 5/5    | YES    | YES   |
| 10  | 10000 | 128  | 2 | 0.60412812    | 0.60412812     | 5/5    | 5/5    | YES    | YES   |
| 11  | 10000 | 32   | 3 | 0.57359385    | 0.57359385     | 5/5    | 5/5    | YES    | YES   |
| 12  | 10000 | 128  | 3 | 0.95665115    | 0.95665115     | 5/5    | 5/5    | YES    | YES   |
| 13  | 50000 | 32   | 1 | 0.44676545    | 0.44676545     | 5/5    | 5/5    | YES    | YES   |
| 14  | 50000 | 128  | 1 | 0.47740927    | 0.47740927     | 5/5    | 5/5    | YES    | YES   |
| 15  | 50000 | 32   | 2 | 0.30282399    | 0.30282399     | 5/5    | 5/5    | YES    | YES   |
| 16  | 50000 | 128  | 2 | 0.60559267    | 0.60559267     | 5/5    | 5/5    | YES    | YES   |
| 17  | 50000 | 32   | 3 | 0.56538904    | 0.56538904     | 5/5    | 5/5    | YES    | YES   |
| 18  | 50000 | 128  | 3 | 0.94917130    | 0.94917130     | 5/5    | 5/5    | YES    | YES   |

Divergence concentration: bins=128 + single-class + N=10k is the knife
edge. At N=1k (#2) bins=128 doesn't diverge; at N=50k (#14) bins=128
doesn't diverge either. Config #8 sits in the sweet spot where N is
large enough to exercise bin 77/81/97/103 as dominant split candidates
but small enough that each bin has only a few dozen observations — the
regime where int-atomic rounding most easily reorders near-ties.

---

## 4. Flip events at config #8 (Tail 1)

35 flip events, all at config #8 (no non-#8 flips — see §5). Flips are
bit-identical across all 5 runs, so only iter/depth-unique rows are
meaningful. De-duplicated:

| iter | depth | T1 (f,b) | Path 5 (f,b) | T1 gain(T1win) | T1 gain(P5win) | flip_gap_t1 | P5-winner outside T1 top-5 |
|-----:|------:|---------:|-------------:|---------------:|---------------:|------------:|:--------------------------:|
| 43   | 1     | (0, 78)  | (0, 79)      | 0.0358914733   | 0.0358914137   | **5.96e-08** | false |
| 44   | 0     | (0, 81)  | (0, 97)      | 0.0798028     | 0.0783294     | 1.473e-03   | false |
| 45   | 0     | (0, 97)  | (0, 77)      | 0.0659876     | 0.0637881     | **2.200e-03** | false |
| 46   | 0     | (0, 81)  | (0, 90)      | 0.0574080     | 0.0564931     | 9.149e-04   | false |
| 47   | 0     | (0, 90)  | (0, 97)      | 0.0492145     | 0.0474010     | 1.813e-03   | false |
| 48   | 0     | (0, 97)  | (0, 103)     | 0.0416295     | 0.0408764     | 7.530e-04   | false |
| 49   | 0     | (0, 103) | (0, 81)      | 0.0358684     | (outside top-5)| ≥ 2.072e-03 | **true** |

Observations:
- 6 of 7 iter-unique flips are at **depth 0** (the root split for that
  iter's tree) — Path 5 disagrees on which FEATURE BIN to pick for the
  root of each tree in iters 44-49. This is catastrophic for downstream
  stability: the root split determines the whole tree topology.
- Iter 43 is the anomalous one: the flip is at depth 1, and the two
  candidates are separated by just **5.96 × 10⁻⁸** in T1's gain. This
  is within ULP-level float32 noise — a legitimate near-tie.
- Iter 45 at depth 0 sets ε_min: the largest flip_gap. To absorb all
  seven iter flips, ε must strictly exceed 2.200 × 10⁻³.
- The iter-43 flip is a **lower bound** on ε_min even by itself:
  5.96e-8 is the threshold above which the flip gets ε-gated. If we
  only cared about the big flips, ε = 2.2e-3 would still classify the
  iter-43 gap (5.96e-8) as below-ε — but so is the legit rank-0/rank-1
  gap at the same node, so the ε-gated Path 5 would STILL not reproduce
  T1 there because it would fall back to a sort or tiebreaker that
  already disagrees.

---

## 5. Flip events at non-#8 configs

**None.** All 17 non-#8 configs have zero winner disagreements across
300 nodes × 5 runs = 1,500 node-evaluations per config. T1 and Path 5
agree on the winner (feat, bin) at every node, including at zero-gain
tied nodes (configs 1, 2, 14 have 120 / 255 / 50 such ties respectively
per run × 5 runs, and both kernels pick the same deterministic fallback
— lowest feat/bin index).

This confirms that the final_loss bit-exact agreement at 17 configs is
not coincidence-cancellation of flips; it reflects ACTUAL tree-structure
bit-exact agreement.

---

## 6. Legitimate-gap floor per config (Tail 2)

Minimum T1 rank-0 minus rank-1 gain gap per config (excluding the
first-flip depth and its cascade).

| cid | non-flip nodes | zero-gain ties | min legit gap | min **positive** legit gap |
|-----|---------------:|---------------:|--------------:|---------------------------:|
| 1   | 1500 | 120 | 0          | 1.043e-07 (r1 i40 d3) |
| 2   | 1500 | 255 | 0          | 8.732e-06 (r1 i35 d2) |
| 3   | 1500 | 0   | 9.110e-03  | 9.110e-03             |
| 4   | 1500 | 0   | 2.628e-03  | 2.628e-03             |
| 5   | 1500 | 0   | 9.387e-04  | 9.387e-04             |
| 6   | 1500 | 0   | 1.984e-04  | 1.984e-04             |
| 7   | 1500 | 0   | 4.530e-06  | 4.530e-06             |
| **8** | **1295** | **130** | **0** | **9.155e-05** (r1 i14 d3) |
| 9   | 1500 | 0   | 2.532e-02  | 2.532e-02             |
| 10  | 1500 | 0   | 1.740e-03  | 1.740e-03             |
| 11  | 1500 | 0   | 1.109e-02  | 1.109e-02             |
| 12  | 1500 | 0   | 5.286e-03  | 5.286e-03             |
| 13  | 1500 | 0   | 7.630e-06  | 7.630e-06             |
| 14  | 1500 | 50  | 0          | 5.603e-06 (r1 i49 d1) |
| 15  | 1500 | 0   | 2.501e-01  | 2.501e-01             |
| 16  | 1500 | 0   | 2.403e-02  | 2.403e-02             |
| 17  | 1500 | 0   | 5.005e-01  | 5.005e-01             |
| 18  | 1500 | 0   | 1.671e-03  | 1.671e-03             |

Four configs (1, 2, 8, 14) have zero-gain ties that pin min_legit_gap
to 0. These are pure/terminal nodes where every candidate split produces
gain=0 (no improvement to the loss); the deterministic feat/bin
tiebreaker already handles them bit-exact in both kernels, so they do
NOT drive flips — but they do prevent any finite ε from being below
ε_max in the strict reading.

Even under the optimistic "positive-only" reading, ε_max⁺ = 1.043e-07
(config #1 iter 40 depth 3). No config improves on this.

---

## 7. ε threading analysis

| Quantity | Value | Source |
|---|---|---|
| ε_min | 2.200e-03 | config #8 run 1 iter 45 depth 0 (flip 97 → 77) |
| ε_max (incl. ties) | 0.0 | configs 1/2/8/14 zero-gain ties |
| ε_max⁺ (positive only) | 1.043e-07 | config #1 run 1 iter 40 depth 3 |
| safety ratio (incl.) | 0.000 | target ≥ 2.0 |
| safety ratio (positive) | **4.742e-05** | target ≥ 2.0 |

**Gap to safety:** ε_min exceeds the positive floor by a factor of
21,091×. Even if every zero-gain tie were resolved by the same
deterministic feat/bin tiebreak in both kernels (which is the case
observationally, but not a property we can prove in general), the
positive-gap floor is still four orders of magnitude below ε_min.

The **structural reason** this is impossible to thread: the same near-tie
that causes the iter-43 flip (5.96 × 10⁻⁸) is ALSO the legit rank-0/rank-1
gap at that node. That is, ε-gating cannot distinguish between "ambiguous
winner, fall back to slow path" (our intent) and "clear winner, no need
for slow path" when the two cases share the same gain separation. For
any ε that classifies iter-43's 5.96e-8 near-tie as ε-gated, a very
large number of legit nodes (8k-per-config bucket) would ALSO trip the
gate and fall back to the slow path — destroying Path 5's speed
advantage at precisely the regimes Path 5 targets.

---

## 8. Verdict

**DEC-026-G1: FALSIFIED.**

ε threading is impossible. ε_min = 2.200e-03 >> ε_max⁺ = 1.043e-07
(4.7e-5 × target safety). The killer is
`config=1 run=1 iter=0 depth_level=3` (zero-gain tie) under the strict
reading, and `config=1 run=1 iter=40 depth_level=3`
(1.043e-07 legit_gap) under the positive-only reading.

**Structural cause:** Path 5's int-atomic SCALE=2³⁰ fixed-point
accumulation introduces ULP-level gain perturbations that, at
config #8's specific N × bins × class regime, reorder the top-2 split
candidates at 7 out of 50 iterations — but the gain separations of
those flips span four orders of magnitude (5.96e-8 to 2.2e-3), which
overlaps the FULL RANGE of legitimate top-2 separations in non-#8 runs.
No ε can simultaneously:

1. be large enough to gate the 2.2e-3 flip (iter 45), AND
2. be small enough to not spuriously gate the 1.04e-7 legit separation
   (config #1 iter 40).

These two constraints are irreducibly in conflict. The test was
designed to detect exactly this conflict, and it did.

---

## 9. Forward pointer

### G2 status
**G2 (ε-gated Path 5 implementation) is CANCELLED.** The gate criterion
for G1 required safety_ratio ≥ 2.0; observed 4.7e-05. There is no
calibration that rescues G2.

### R8 status
**R8 remains at 1.01×** (Sprint 24 honest post-fix position, unchanged).
DEC-026 does not provide a speed-up path back to the pre-S24 1.90×
record. The ≥1.5× Verstappen campaign gate was retroactively failed at
Sprint 24 D0 when v5 eliminated T2's structural speed advantage for
DEC-023 parity correctness; DEC-026 investigated whether an ε-tiebreak
could unblock T2 Path 5 to recover that ceiling, and has now shown
it cannot.

### Path forward options
These are documented for the DEC-026 FALSIFIED addendum, not acted upon
here:

1. **Accept T2 exactness, drop Path 5.** Keep T2-sort serial scatter as
   the sole accumulation path. Gives up Path 5's potential 1.2–1.5×
   further speed-up but preserves bit-exact parity with DEC-008.
2. **Widen SCALE to 2⁴⁰ (or use double-precision atomics).** Shifts the
   int-atomic noise floor ≈10³× lower. This might shrink the iter-43
   near-tie from 5.96e-8 to ~5.96e-11, but the iter-45 flip (2.2e-3) is
   a genuine float32 gain disagreement, not a quantization artifact —
   widening SCALE won't address it. Requires a separate G3 sweep to test.
3. **Hybrid: Path 5 for large N, T1/T2 for small N and bins=128.**
   Config #8 is the only failure, and it sits at a specific boundary.
   Empirical routing by (N, bins) would sidestep the flip regime but
   introduces an accept-conditions table that would need its own
   calibration sweep and ongoing maintenance. Not recommended for the
   speedup-at-stake.
4. **Abandon DEC-026; pursue DEC-027 (alternative accumulation path).**
   E.g. per-feature deterministic radix-sum as used in XGBoost's GPU
   fast-path. Requires new R-tier analysis.

**Recommendation:** option 1 (accept T2, drop Path 5). Smallest risk,
preserves parity. Rebrand R8 as "structural ceiling" rather than
"implementation gap".

---

## 10. Raw data

All artifacts under
`/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/benchmarks/sprint25/g1/`:

- `results/sweep_summary.csv` — 180 rows, `(config, run, kernel, final_loss, elapsed_sec)`
- `results/sweep.log` — per-run human-readable progress log
- `results/traces/c{cid}_r{run}_{kernel}.csv` — 180 trace files, Option A dumps
- `results/analysis/flip_events.csv` — 35 rows, Tail 1 flip events
- `results/analysis/legit_gap_floor.csv` — 18 rows, Tail 2 per-config floor
- `results/analysis/epsilon_threading.json` — ε threading verdict machine-readable

Analyzer source: `analyze_g1.py`
Dump-kernel source: `g1_gain_dump.cpp`
Sweep driver: `run_g1_sweep.sh`

Reproduce the analysis:
```
python3 "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx/benchmarks/sprint25/g1/analyze_g1.py"
```

Environment: macOS 15.x (Darwin 25.3.0), Apple Silicon. Sweep completed
2026-04-21 13:56:23 → 14:01:27, elapsed 5 min 04 s.
