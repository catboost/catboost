# PROBE-C Finding — iter=2 divergence is at depth=2, not depth=0

**Date**: 2026-04-24
**Anchor**: `np.random.default_rng(42)`, N=50000, 20 features,
y = 0.5·X[0] + 0.3·X[1] + 0.1·noise, ST/Cosine/RMSE, depth=6, bins=128, l2=3, lr=0.03

## Summary

L3's "S2 SPLIT divergent at iter=2 depth=0" verdict is **retracted**. PROBE-C
ran a per-feature border-value comparison and a depth-by-depth tree[1] match,
both in-context with the L0–L3 evidence held live. Findings:

1. **MLX's 127-border grid is a strict subset of CPU's 128-border grid**
   (ULP=1 tolerance). Each of the 20 features is missing exactly one CPU
   border at index 6 — the cap-127 truncation (DEC-039) cuts a real border
   out of the grid every time.

2. **iter=2 tree[1] depth-by-depth (physical-value space)**:

   | d | MLX (feat, bin) | MLX phys | CPU feat | CPU phys | verdict |
   |---|---|---|---|---|---|
   | 0 | (0, 64)  |  0.014169 | 0 |  0.014169 | **AGREE** (4.2e-11) |
   | 1 | (1, 58)  | -0.092409 | 1 | -0.092413 | **AGREE** (4.6e-6, grid offset) |
   | 2 | (10, 79) |  0.305683 | 0 | -0.946874 | **DIVERGE** |
   | 3 | (14, 109)|  1.073884 | 0 |  1.042727 | DIVERGE |
   | 4 | (12, 10) | -1.366718 | 1 |  0.815438 | DIVERGE |
   | 5 | (13, 101)|  0.830383 | 1 | -1.081904 | DIVERGE |

3. **L3's depth-0 disagreement was a coordinate-system error**. CPU's
   `tree[1].splits[0].split_index=3` indexes the CBM *stored* border list
   (only 6 stored thresholds for feat=0 at random_seed=42; stored[3]=0.014169).
   MLX's `bin_threshold=64` indexes the upfront 127-bin grid (also 0.014169).
   They are the **same physical split**.

4. **The real divergence is at depth=2**, where CPU re-picks feat=0 (still
   high signal in the d=0+d=1 induced partitions) but MLX picks feat=10 —
   pure noise in this synthetic setup. The mechanism is in the Cosine gain
   argmax at iter=2 depth=2, not iter=2 depth=0.

## What this changes for the successor probe

The original PROBE-C plan was a per-bin (cosNum, cosDen, wL, wR) dump at
**iter=2 depth=0** to find the first divergent (feature, bin). That plan is
obsolete: depth=0 agrees. The successor probe must dump per-feature gain at
iter=2 **depth=2**, with the d=0+d=1 partitions inherited from the agreed
splits — find the first feature where the ranking differs (CPU re-picks
feat=0; MLX prefers feat=10).

## Candidate loci (revised, depth-2)

- (a) **Cosine joint-denominator drift at depth=2** across 4 partitions —
  precision-class mechanism that compounds with depth, not iter.
- (b) **Per-leaf state-vector accumulation** differing across the 4 leaves at
  d=2 — partition-state class.
- (c) **Histogram bin-edge inclusivity** (≤ vs <) at the missing-border seam
  letting docs migrate between bins, biasing leaf totals at d=2.

## Artifacts

- `data/mlx_borders.tsv`, `data/cpu_borders.tsv` — per-feature border values
- `data/border_count.csv`, `data/border_diff.csv` — Stage-1 deltas
- `data/find_missing.txt`, `data/set_diff.txt` — strict-subset proof
- `data/mlx_anchor.json` — full MLX 2-tree model JSON
- `data/tree1_compare.txt` — Stage-2 depth-by-depth comparison
- `scripts/build_dump_borders.sh`, `scripts/build_probe_c2.sh` — build variants
- `scripts/run_probe_c.py`, `scripts/set_diff.py`, `scripts/find_missing.py`,
  `scripts/compare_tree1.py` — analysis pipeline

## Build invariants

- `kernel_sources.h` md5 = `9edaef45b99b9db3e2717da93800e76f` (unchanged).
- Both binaries (`csv_train_probe_c`, `csv_train_probe_c2`) built with
  `-DCOSINE_T3_MEASURE` to bypass the ST+Cosine guard. The DUMP_BORDERS
  variant additionally has `-DCATBOOST_MLX_DUMP_BORDERS` and exits early
  after dumping; the c2 variant runs full training and supports `--output`.
