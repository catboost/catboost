# S30-T3-MEASURE — Verdict

## Gate one-liners (copy-paste into sprint close)

```
G3a ST anchor:  FAIL  — aggregate drift 53.30%  (threshold <2.0%)  — T4a BLOCKED
G3b LG-Mid:     FAIL  — ratios 1.274–1.314       (threshold [0.98, 1.02]) — T4b BLOCKED
G3c LG-Stress:  FAIL  — ratios 1.440–1.452       (threshold [0.98, 1.02]) — K2 FIRED
G4 parity:      PASS  — 28/28 (no regression to L2 / DW+Cosine cells)
G5 perf:        FAIL  — K4 overhead +7.1% vs L2 baseline (threshold <5%)  — K3 TRIGGERED
```

**K2 trigger decision**: K2 FIRED — T4b skipped this sprint; S31-LG-DEEP-RESIDUAL to be filed.

**K3 trigger decision**: K3 TRIGGERED — @performance-engineer consultation required before
merge; do NOT remove guards under perf regression. Overhead is borderline (+7.1% vs 5%
threshold) and within measurement uncertainty, but formal K3 invocation is required by DEC-035.

---

## Cell-parameter table (auditor reference)

| Gate | N | depth | max_leaves | iters | seeds | grow_policy | score_function |
|------|---|-------|-----------|-------|-------|-------------|----------------|
| G3a | 50000 | 6 | 64 (2^6) | 50 | {42,43,44,45,46} | SymmetricTree | Cosine |
| G3b | 1000 | 6 | 31 | 50 | {42,43,44,45,46} | Lossguide | Cosine |
| G3c | 2000 | 7 | 64 | 100 | {0,1,2} | Lossguide | Cosine |
| G4 | 10000 | 6 | — | 50 | {1337,42} | ST+DW+LG (L2/Cosine per policy) | L2/Cosine |
| G5 | 50000 | 6 | — | 50 | 42 | SymmetricTree | Cosine vs L2 |

Binary: `csv_train_t3` (built with `-DCOSINE_T3_MEASURE`, contains K4 fp64 fix from T2).
CPU reference: `catboost` pip package, `score_function='Cosine'`, `l2_leaf_reg=3.0`,
`random_strength=0.0`, `bootstrap_type='No'`.

---

## G3a — ST anchor detail

| seed | MLX_RMSE | CPU_RMSE | ratio | drift_pct |
|------|----------|----------|-------|-----------|
| 42 | 0.29659500 | 0.19362645 | 1.531790 | 53.18% |
| 43 | 0.29593600 | 0.19357118 | 1.528823 | 52.88% |
| 44 | 0.29566200 | 0.19320460 | 1.530305 | 53.03% |
| 45 | 0.29550900 | 0.19250458 | 1.535075 | 53.51% |
| 46 | 0.29712900 | 0.19305704 | 1.539074 | 53.91% |

**Aggregate drift: 53.30%** (threshold < 2.0%).

**Interpretation**: The K4 fp64 fix reduced the per-gain-candidate residual 12.5× at iter=1
(T2 verdict: 4.75e-5 → 3.81e-6). However, the 50-iteration RMSE drift remains at ~53%,
approximately the same as the pre-K4 value (~47% documented in S28-OBLIV-DISPATCH REFLECT
section). The fix did NOT collapse the iteration-compound cascade.

Diagnosis: The residual gain error of 3.81e-6 (after K4) is still large enough to flip
individual split-feature selections at some tree depths. These split flips produce wrong
partitions in subsequent iterations (the downstream cascade that DEC-035 T2 targeted). The
12.5× gain-residual reduction is insufficient to eliminate split flips when the signal-to-noise
ratio at the gain boundary is low (many near-equal-gain candidates). The fix reduces
individual-iter drift from 0.77% to ~0.74% (confirmed at iter=1) but iteration-compound
amplification (each mis-split building on the previous) carries the final drift to ~53%.

Spot-check at iter=1 (seed=42):
- Post-K4 drift: 0.74% (MLX=0.58304000, CPU=0.57874830)
- Pre-K4 drift from S28-OBLIV-DISPATCH T7 G6a: 0.77% (seed=42)
- Ratio improvement: 0.74/0.77 = 0.96 — small but consistent with 12.5× gain-residual reduction

Iter-5 drift (seed=42): 3.94% — compounding is rapid, reaching G3a threshold within 5 iterations.

---

## G3b — LG-Mid detail

| seed | MLX_RMSE | CPU_RMSE | ratio | drift_pct |
|------|----------|----------|-------|-----------|
| 42 | 0.24432600 | 0.18599921 | 1.313586 | 31.36% |
| 43 | 0.23802200 | 0.18680430 | 1.274178 | 27.42% |
| 44 | 0.24214500 | 0.18867654 | 1.283387 | 28.34% |
| 45 | 0.24367700 | 0.18713518 | 1.302144 | 30.21% |
| 46 | 0.24637800 | 0.18774096 | 1.312329 | 31.23% |

**All 5 seeds fail**: ratios 1.274–1.314 outside [0.98, 1.02].

**Interpretation**: LG+Cosine at the t5-continuity cell shows 27–31% drift — much larger than
the S29 spike measurement (0.003–0.197% per seed at 50 iter on the shallow cell). The t5-
continuity cell is at depth=6, max_leaves=31, which exercises many more priority-queue
choices than the S29 spike cell (depth=3, max_leaves=8). This is consistent with the S29
verdict limitation warning: "With only 8 leaves the queue makes few contested choices; any
latent ordering sensitivity would be more visible at 64+ leaves." At max_leaves=31 with 50
iterations, the compounding from split selection errors is severe. K4 (targeting cosDen
accumulator) has not resolved this.

---

## G3c — LG-Stress detail (K2 gate)

| seed | MLX_RMSE | CPU_RMSE | ratio | drift_pct |
|------|----------|----------|-------|-----------|
| 0 | 0.13221800 | 0.09163246 | 1.442917 | 44.29% |
| 1 | 0.12870000 | 0.08936491 | 1.440163 | 44.02% |
| 2 | 0.13248700 | 0.09121844 | 1.452415 | 45.24% |

**All 3 seeds fail**: ratios 1.440–1.452 outside [0.98, 1.02].

At max_leaves=64, 100 iterations, depth=7, the drift reaches 44–45% — comparable to ST+Cosine
at 50 iterations. This confirms that LG+Cosine at production-depth configurations exhibits
severe RMSE divergence, not just minor float-precision drift.

---

## G4 — parity suite

28/28 tests pass. No regression to:
- ST+L2 (DEC-028 regression class)
- DW+Cosine (confirmed parity, no degradation)
- LG+L2 (forced-L2 cells, guard-free path)

The K4 change (fp64 accumulators in `FindBestSplit` Cosine path) did not disturb any
non-Cosine code paths. Suite runtime: 55.5 seconds.

---

## G5 — perf regression

| config | mean ms/iter | N samples | stddev |
|--------|-------------|-----------|--------|
| Cosine (K4 fp64) | 12.91 ms | 135 | 1.10 ms |
| L2 (baseline) | 12.05 ms | 135 | 0.57 ms |

**Cosine/L2 ratio: 1.071 (+7.1% overhead)**. Fails the <5% threshold.

Method note: The intended pre-K4 Cosine baseline (`csv_train_ref`, dated Apr 18) could not run
Cosine due to the CLI guard. `csv_train_t3` with `score_function='L2'` is used as the pre-K4
proxy — this is conservative (L2 path is structurally identical to the pre-K4 Cosine path in
tree-search overhead; K4's only change was float32→double in the accumulator). Multi-run
measurement (3 runs × 50 iters = 135 steady-state samples per config, warmup iters 0–4 excluded).

The 7.1% overhead is consistent across runs. However, given the 1.10ms stddev on a 12.91ms
mean (8.5% CV), the measurement is noisy enough that this could be within 5% on a quieter
system. K3 is triggered per DEC-035 rule (threshold 5%; measurement shows 7.1%); @performance-
engineer to confirm or narrow the interval on dedicated hardware.

**K3 TRIGGERED** — do NOT remove guards under this perf regression. Consult @performance-
engineer before merge.

---

## Root-cause analysis and implications for T4

### Why G3a/G3b/G3c all fail

The K4 fix targets accumulator precision: replacing float32 with double in the `cosNum_d` and
`cosDen_d` running sums. This reduced the **gain residual at iter=1** by 12.5× (from 4.75e-5
to 3.81e-6). However, the **split-selection flip rate** was not eliminated. At iter=1 with N=50k
and depth=6 there are 2540 bin candidates per partition; the 3.81e-6 residual is still large
enough to flip the ordering of near-equal-gain candidates.

Each flip in split selection corrupts the partition assignment for that tree, which then corrupts
the gradient signals for all subsequent trees. Fifty iterations of compound misdirection produces
~53% RMSE drift — the same as pre-K4 (the 47% pre-K4 figure in S28 T7 was at the limit of
measurement because T7 only checked 1 iteration; 50-iter compounding had been flagged as "expected
to grow" in T7's REFLECT). K4 marginally improved iter-1 (0.74% vs 0.77%) but the cascade is
largely unchanged.

### Implications for T4a (ST+Cosine guard removal)

T4a is BLOCKED. The G3a criterion (<2% aggregate drift at 50 iter) is not met. The guards are
correct to remain in place. Removing them would expose users to ~53% RMSE drift at standard
50-iteration training runs.

### Implications for T4b (LG+Cosine guard removal)

T4b is BLOCKED and K2 has fired. The guards remain correct.

### What needs to change for T4a/T4b to unlock (S31 scope)

G3a requires aggregate drift < 2%. Currently at 53%. The K4 fix's iter-1 gain residual of
3.81e-6 is still causing split-selection flips. To reach < 2% aggregate drift at 50 iter,
the iter-1 split-flip rate needs to approach zero.

Two directions for S31:

**S31-ST-COSINE-DEEPER**: For ST, the Cosine gain formula evaluates `cosNum_d / sqrt(cosDen_d)`
with double accumulators, but the input histograms `sumLeft`, `weightLeft` etc. are already
float32 from the histogram kernel. The double accumulation eliminates the summation rounding
but not the float32 input quantization. The per-bin gain residual of 3.81e-6 is the float32
quantization of the final gain scalar (≈ gain × fp32_eps); this is irreducible without
changing the histogram representation. The path forward may be: store histograms in float32
(acceptable for gradient magnitudes) but compute the Cosine gain formula entirely in double
using the float32 histogram values as input — the current K4 already does this. If the residual
is truly irreducible at 3.81e-6, then the split-flip rate cannot be eliminated by accumulator
precision alone. This would require a different approach: either (a) verify that the per-bin
gain ranking is actually flipped in practice (not just that the residual exceeds epsilon) by
checking that CPU and MLX agree on the winning split feature at iter=1, or (b) investigate
whether a bias in the gain formula (not just rounding) is causing systematic divergence.

**S31-LG-DEEP-RESIDUAL** (already mandated by K2): For LG, the spike at depth=3/max_leaves=8
showed 0.003–0.197% drift. At depth=6/max_leaves=31 (G3b) this grew to 27–31%. The
priority-queue ordering surface is heavily implicated. A dedicated measurement sprint at
increasing max_leaves values (8, 16, 31, 64) with iter-1 tree-structure comparison (do MLX
and CPU choose the same splits in the same order?) would isolate whether the LG problem is
accumulator precision (fixable by K4-class changes) or priority-queue ordering (requiring a
different algorithmic alignment).

---

## Data artifacts

All raw CSV files are under `docs/sprint30/t3-measure/data/`:

| File | Contents | Rows |
|------|----------|------|
| `g3a_st_anchor.csv` | G3a per-seed RMSE + ratio + drift_pct | 5 seeds |
| `g3b_lg_mid.csv` | G3b per-seed RMSE + ratio + drift_pct | 5 seeds |
| `g3c_lg_stress.csv` | G3c per-seed RMSE + ratio + drift_pct | 3 seeds |
| `g5_perf.csv` | G5 per-iter ms for Cosine(K4) and L2 (3 runs × 50 iter each) | 300 rows |

Runner scripts:
- `docs/sprint30/t3-measure/run_t3_st_anchor.py` — G3a
- `docs/sprint30/t3-measure/run_t3_lg_mid.py` — G3b
- `docs/sprint30/t3-measure/run_t3_lg_stress.py` — G3c

Build command for `csv_train_t3` (bypasses both ST and LG Cosine guards; K4 active):
```
clang++ -std=c++17 -O2 -DCOSINE_T3_MEASURE \
  -I. -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  catboost/mlx/tests/csv_train.cpp -o csv_train_t3
```
