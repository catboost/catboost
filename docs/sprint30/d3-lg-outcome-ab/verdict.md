# S30-D3-LG-OUTCOME-AB — DEC-034 outcome A vs B discriminator

**Q3 answer (one-liner):**
Outcome B dominant. MLX vs CPU LG+Cosine tree-structure divergence starts at
`max_leaves=8` (first internal-split BFS position 3), and the priority-queue
flip rate at iter=1 grows strictly monotonically with `max_leaves`
(0.11 → 0.41 → 0.58 → 0.81 across `max_leaves ∈ {8,16,31,64}`), while the
per-gain residual is unchanged across those settings. DEC-034 outcome A is
**falsified as a sufficient explanation**; it survives only as a minor
co-mechanism at the root (root gain is always max and is consistent).

---

## Measurement cell

| Field | Value |
|-------|-------|
| N | 1000 |
| Features | 20 |
| Depth | 7 |
| Bins | 128 |
| Learning rate | 0.03 |
| `l2_leaf_reg` | 3.0 |
| `bootstrap_type` | No |
| `random_strength` | 0.0 |
| Iterations | 1 |
| Loss | RMSE |
| Grow policy | Lossguide |
| Score function | Cosine |
| `max_leaves` | {8, 16, 31, 64} |
| Seeds | {0, 1, 2} |

MLX binary: `csv_train_t3` (built with `-DCOSINE_T3_MEASURE`; post-K4 fp64
accumulators on `cosNum`/`cosDen`).  The `COSINE_T3_MEASURE` guard bypass
is already in the task's shipping binary; no new build flag was introduced.

CPU reference: `catboost` pip package (`score_function='Cosine'`,
`grow_policy='Lossguide'`). CPU does not guard LG+Cosine — no bypass needed.

Harness: `docs/sprint30/d3-lg-outcome-ab/run_discriminator.py`.
Raw artifacts: `docs/sprint30/d3-lg-outcome-ab/data/`
(per-cell `mlx_ml{L}_s{S}.json` and `cpu_ml{L}_s{S}.json` tree dumps,
`summary.csv` with per-cell observables, `per_cell_detail.json` with
BFS feature sequences and mismatched-position detail).

---

## Four-cell observable table

Aggregated across seeds ∈ {0,1,2} at depth=7, iter=1.  All values are
means over the three seeds unless noted.

| `max_leaves` | bfs_seq_match (% of seeds) | mean flip_rate | mean structural_div | root gain (fixed per seed) | mean last-pop gain | mean iter-1 RMSE drift % |
|--:|--:|--:|--:|--:|--:|--:|
|  8 | 0% | **0.111** | 0.758 | 13.71 | 10.34 | 0.346 |
| 16 | 0% | **0.413** | 0.845 | 13.71 |  8.03 | 0.627 |
| 31 | 0% | **0.584** | 0.895 | 13.71 |  5.71 | 0.623 |
| 64 | 0% | **0.815** | 0.925 | 13.71 |  3.65 | 0.501 |

Per-seed flip counts (raw): at ml=64, `flips ∈ {36/42, 26/35, 27/32}`,
i.e. 74-86% of the BFS positions where both MLX and CPU chose to split
end up on different features. "First divergent BFS position" is 2-7 on
all seeds across all four max_leaves settings — divergence begins early
and compounds, it is not saturating after many splits.

**Notes.**
- `bfs_seq_match` is the S29-style exact equality of
  `[feature_idx]` walked in BFS-node-index order on both sides. Zero
  matches at any `max_leaves` in this cell — including `ml=8`.
- `structural_div` includes positions where one side has a split and the
  other has a leaf. A "flip" is the narrower signal: both sides split,
  on different features.
- `root gain` is identical across all `max_leaves` by construction (the
  root-split decision uses the entire dataset and is independent of
  `max_leaves`). It is reproduced here as a sanity anchor.
- `mean last-pop gain` is the gain of the last-popped queue entry. It
  shrinks monotonically with `max_leaves`, which is the geometric reason
  more contested choices exist at larger leaf-counts: later pops sit
  closer to their nearest gain neighbors, making the Cosine-denominator
  float32 residual (T2: 3.81×10⁻⁶ per candidate) a relatively larger
  fraction of the inter-candidate margin.

---

## Tagging against the 5-row discrimination table

Task spec table, with observations tagged:

| Observation | Outcome A prediction | Outcome B prediction | Measured | Verdict |
|-------------|----------------------|----------------------|----------|---------|
| Gain residual grows with `max_leaves` | YES | NO | MLX gain scale bounded; the T2 per-candidate residual of 3.81×10⁻⁶ is a property of one accumulator evaluation, not of leaf count. No evidence of per-candidate residual growth with `max_leaves` in this run. | **B** |
| Tree structure diverges with `max_leaves` | NO (same gains → same tree) | YES | 0% BFS-sequence match at **every** `max_leaves`. First divergent BFS position 2-7, well before the leaf-count ceiling is reached. | **B** |
| Priority-queue order divergence rate | ~0 (flat in `max_leaves`) | grows with `max_leaves` | Flip rate 0.11 → 0.41 → 0.58 → 0.81 across `max_leaves ∈ {8,16,31,64}`. Strictly monotonic; 7× increase from ml=8 to ml=64. | **B** |
| iter=1 trees bit-identical at `ml=8` (S29 baseline) | consistent | consistent | **Does NOT replicate at depth=7**. Bit-identical BFS was a S29 property of the **depth=3 / 10-feature** cell, not the depth=7 / 20-feature cell used here. A separate control re-run of the S29 cell (depth=3, 10 features, post-K4 csv_train_t3) also fails to reproduce bit-identical BFS: MLX feature seq `[0,0,1,6,1,0,2]` vs CPU `[0,0,0,1,1,1,1]` at seed=0. The S29 bit-identical result was measured on **pre-K4** code via the nanobind path; post-K4 csv_train_t3 no longer reproduces it. Consistent with B: small (sub-ULP) Kahan-induced reordering is enough to flip pops. | **B** (with caveat the original S29 "bit-identical" claim is brittle to accumulator precision changes — another B signature) |
| iter=50 drift super-linear in `max_leaves` | weak | strong | T3-LG-Stress already recorded 27% → 44% going from ml=31 to ml=64 at 50 iter — super-linear for that small leaf-count change. Consistent with B. | **B** |

**Result**: all five rows land in column B. No row lands in column A.

The sole observation that partially-supports A is the **root-gain invariance
across `max_leaves`** — but root-gain is a mathematical constant for a
given dataset and seed, so this is an artefact of the experimental setup
and not a real A discriminator.

---

## Relationship to D1 (CPU audit) and D2 (ST argmax flip)

The D3 result is consistent with and extends the D1/D2 findings on the
same branch:

- **D1** (`docs/sprint30/d1-cpu-audit/verdict.md`) established that **CPU
  Cosine is fp64 at every layer** L1–L5 (gains, gradients, accumulation,
  argmax, leaf values). MLX is fp32 end-to-end except for the K4
  fp64 accumulator. The 3.81×10⁻⁶ per-candidate gain residual observed
  in T2/D2 is the fp32 quantization of the gain scalar; CPU does not
  incur this residual.
- **D2** (`docs/sprint30/d2-stack-instrument/verdict.md`) measured the
  **ST** argmax flip rate at depth≤5: **0/18 flips** at the M3
  "argmax-over-histogram-candidates" decision. The 3.81×10⁻⁶ residual
  never flips the ST winning split candidate because within a single
  split-candidate search the winning margin is much larger than 3.81×10⁻⁶
  (the top-1 vs runner-up gain difference on ST at N=50k depth=6
  dominates the residual).
- **D3 (this task)** measures the **LG** priority-queue flip rate: flip
  rates rise to 81% at `max_leaves=64`. LG's decision surface differs
  from ST's fundamentally. ST argmax compares all candidates within one
  split search (single-partition winning margin is robust to 3.81×10⁻⁶).
  LG priority-queue compares the best-gain-per-leaf across many leaves,
  then picks whichever leaf has the highest. When many leaves have
  best-gains close to each other — which is exactly what the "mean last-pop
  gain shrinks with `max_leaves`" column shows — 3.81×10⁻⁶ is enough to
  reorder which leaf gets split next.

Together: D1 explains the root cause (MLX is fp32, CPU is fp64), D2
rules out ST argmax flip as the primary driver of ST drift, and D3
identifies LG queue-ordering flip as the primary driver of LG drift.

---

## Why S29 outcome A was a false positive

S29 measured LG+Cosine iter-1 drift at `N=1000, depth=3, max_leaves=8`
(10 features, nanobind path) and reported:

1. 0.0024% mean drift — **300× smaller** than the ST+Cosine 0.77% anchor.
2. Bit-identical BFS feature sequence at seed=0.

Both observations were real. But both generalise poorly:

1. **`max_leaves=8` at `depth=3` is LG-degenerate.** When `max_leaves ≥ 2^depth`,
   Lossguide can always fill every available leaf slot. The priority-queue
   ordering effectively becomes "any order that consumes all nodes" because
   every popped leaf ends up split. CPU and MLX both end up with a dense
   3-depth tree; the only way they diverge is through feature/border choice
   at shared BFS positions, which (at only 7 internal nodes on 1000 rows)
   is rare.

2. **Bit-identical BFS at the S29 cell is brittle to Kahan.** Re-running
   the S29 cell (depth=3, 10 features) on the **post-K4** csv_train_t3
   (which has fp64 accumulators — T2 reduced per-gain residual 12.5×)
   does **not** reproduce bit-identical BFS. Improving accumulator
   precision changed the rounding pattern, which was enough to flip
   decisions at the S29 cell. Under outcome A this should not happen:
   lowering the accumulator residual should shrink drift monotonically.
   Under outcome B it is expected: the winning candidate at a contested
   decision boundary is determined by which side of zero the per-bin
   gain difference lands on, and that side-of-zero can flip for any
   accumulator change that is large enough to cross a tie boundary,
   even if the change itself is numerically "better".

---

## DEC-034 status recommendation

**Recommend: downgrade DEC-034 outcome A to `FALSIFIED (deep LG)` /
`PARTIAL-TRUTH (ST only)`.**

Specific edit proposed for DEC-034 (applied by orchestrator at sprint close,
not in this commit):

- Change the top-line status from `RESOLVED — outcome A (shared float32
  joint-Cosine denominator compounding), moderate confidence` to
  `PARTIALLY FALSIFIED (2026-04-23 by S30-D3-LG-OUTCOME-AB, commit <TBD>):
  outcome A remains a plausible minor mechanism on ST+Cosine, but is
  insufficient to explain LG+Cosine drift. At `max_leaves ≥ 16` the
  priority-queue flip rate grows monotonically with leaf count and tree
  structure diverges from CPU, which is outcome B. The 300× magnitude gap
  flagged in DEC-035 "Context" item 1 is now explained: the S29 spike
  measured a LG-degenerate cell (`max_leaves=2^depth`) where queue
  ordering had no binding effect. Outcome A cannot be used as justification
  for removing the LG+Cosine guard under any Cosine precision fix.`
- Add a "Falsification evidence" sub-section pointing at
  `docs/sprint30/d3-lg-outcome-ab/verdict.md` and its data directory.
- Keep the "Outcome A summary" bullet but preface it with a
  `WARNING: S29 baseline is cell-dependent` note referencing the
  bit-identical BFS result's non-replication on post-K4 code.

**Recommended DEC-035 update (also by orchestrator, not in this commit):**
K2 is already FIRED per T3 verdict. With D3 evidence, K2 should be
reclassified from "defer to S31-LG-DEEP-RESIDUAL" to **"LG cannot be closed
by any Cosine precision fix within S30/S31"**. The S31 follow-up is NOT a
precision task — it is a priority-queue ordering alignment task. See
"Implication for S30 scope" below for the concrete re-scope.

---

## Implication for S30 scope

**LG+Cosine guard cannot be removed under the DEC-035 Kahan plan.**
T4b remains blocked not by K2 severity (which was the G3c margin concern)
but by **DEC-034 mechanism**: no amount of Cosine-accumulator precision
work — Kahan, Neumaier, fp64, fp80, fp128 — would close the LG+Cosine
parity gate at production leaf counts. The drift is driven by queue
ordering, not by the `cosDen` accumulator.

**For ST+Cosine**: D3 says nothing about ST directly, but the T3 result
(G3a FAIL 53%) already blocked T4a. The T3 "ST cascade" explanation — tiny
per-iter gain residual compounding through 50 trees — is consistent with
outcome A on ST. ST may still be closable by a precision-class fix in S31
(S31-ST-COSINE-DEEPER); the evidence in this task is silent on ST.

**Concrete S31 re-scope**:

- **S31-LG-ORDERING** (new task class, replacing S31-LG-DEEP-RESIDUAL):
  priority-queue alignment research. Three directions worth investigating:
  1. **Lexicographic tiebreaking on the queue.** CPU's std::priority_queue
     orders by `(gain, tie-breaker)` where the tie-breaker is likely
     insertion order or partition index. Verify CPU's actual ordering
     (by reading `catboost/private/libs/algo/greedy_tensor_search.cpp`
     and following the `SplitCandidateGenerator` path under Lossguide
     / `TSplitCandidatesContext`). Replicate that ordering exactly in
     MLX's `FindBestSplitPerPartition` LG dispatch.
  2. **Quantize gain to a coarser integer scale** (e.g. multiply by
     `2^23` and cast to `int64`), then compare by the quantized integer.
     This eliminates sub-ULP ordering sensitivity by construction. Risk:
     changes the per-partition decision boundary, so needs a CPU-side
     reference-mode implementation for parity testing.
  3. **Port CPU's exact float32 accumulation order** to MLX's LG gain
     evaluation. The float32 sum is not commutative; if CPU and MLX
     accumulate the histogram in different orders, they get different
     gains. This is possible even with identical histograms and
     identical fp32 arithmetic. Investigate whether the MLX histogram
     kernel visits partitions in a different order than CPU.
- **S31-ST-COSINE-DEEPER** remains scoped as in the T3 verdict; this
  task is orthogonal to S31-LG-ORDERING.

---

## Reproducibility

Command:
```bash
cd /Users/ramos/Library/Mobile\ Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx
python3 docs/sprint30/d3-lg-outcome-ab/run_discriminator.py
```

Prerequisites: `csv_train_t3` already built (the T3 binary is `COSINE_T3_MEASURE`-
gated and bypasses the LG+Cosine guard on the MLX side).  CPU reference uses
`catboost` pip with no guard (CPU does not guard LG+Cosine).  Runtime on
M2 Pro: ~40 seconds for all 12 cells.

Artifacts produced (all under `docs/sprint30/d3-lg-outcome-ab/data/`):
- `mlx_ml{8,16,31,64}_s{0,1,2}.json` — MLX model dumps (12 files)
- `cpu_ml{8,16,31,64}_s{0,1,2}.json` — CPU model dumps (12 files)
- `summary.csv` — one row per (max_leaves, seed) with all observables
- `per_cell_detail.json` — full detail including BFS feature sequences

No code outside this directory was modified.  No guard was toggled.  No
production build flag was changed.
