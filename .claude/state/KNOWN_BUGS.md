# Known Bugs — CatBoost-MLX

> This ledger tracks persistent, confirmed bugs that have not yet been fixed. Each entry has a pointer to the authoritative decision record, a repro command, and a fix-options summary. Maintainers grep this file first.

---

## BUG-T2-001: Features 1-3 atomic-float race in T2-accum (non-deterministic training)

**Decision record**: `DECISIONS.md DEC-023` (OPEN, S24 scope)
**Discovered**: 2026-04-20, Sprint 23 D0 parity sweep
**Status**: OPEN — attack plan in DEC-023; fix targeted for S24 D0

### Summary

T2-accum features 1-3 use `atomic_fetch_add_explicit(memory_order_relaxed)` on `device atomic_float`. FP32 addition is non-associative; non-deterministic Metal GPU thread scheduling produces 1-2 ULP drift in histogram bins, which can flip a near-tie GAIN comparison early in training and cascade to 105+ ULP in final loss at iters=50.

The race is **config-specific**: it fires at N=10000/bins=128 (config #8) but not at the gate config N=50000/bins=128 (config #14), where the dispatch shape happens to resolve additions in a consistent order. Feature-0 is clean (bin-range scan, no atomics).

### Footprint (S23 D0 measured, N=100 per config; 1800 total trials)

| Config | N | Loss | Bins | Result | ULP vs T1 |
|--------|---|------|------|--------|-----------|
| #8 | 10000 | RMSE | 128 | BIMODAL ~50/50 (0.48231599 / 0.48231912) | 0 / 105 |
| #14 (gate) | 50000 | RMSE | 128 | DETERMINISTIC 100/100 (0.47740927) | 0 |
| #1–#7, #9–#13, #15–#18 (all other configs) | various | various | various | DETERMINISTIC 100/100 | 0 |

**Singleton footprint**: exactly 1 of 18 configs fires. Race is narrowly conditioned on (N=10000, RMSE, bins=128). All N=1000, all N=50000, and the other five N=10000 configs (varying loss / bins) are clean.

**Gate-config seed sweep** (config #14, 500 runs × 5 seeds): 100/100 deterministic on every seed. H1 (structural, not seed-coincidental) supported. 1.90× R8 record is seed-robust.

**Cascade onset**: between iters=40 (deterministic) and iters=45 (bimodal). At iters=50 spread = 105 ULP; at iters=100 spread narrows to 39 ULP (non-monotone — branches converge toward common limit).

### Sibling latent race (S-1)

Found during the S23 D0 atomic-float site inventory (d0_bimodality_verification.md §C). `kHistOneByte` writeback in `catboost/mlx/kernels/kernel_sources.h` uses atomic-float and is RACY when `maxBlocksPerPart > 1`. Currently dead code path: NIT-4 CB_ENSURE enforces `maxBlocksPerPart == 1`. Any future optimization relaxing this (e.g., per-partition multi-block dispatch) reactivates the race. Fix alongside BUG-T2-001 if multi-block dispatch is ever needed; otherwise guarded.

### Repro

```bash
# Build from S23 D0 production tip (84529b47ed)
clang++ -std=c++17 -O2 \
  -I"/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx" \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation -Wno-c++20-extensions \
  catboost/mlx/tests/bench_boosting.cpp \
  catboost/mlx/methods/histogram_t2_impl.cpp \
  -o /tmp/bench_s23

# Run multiple times at config #8 — expect bimodal output
for i in $(seq 1 20); do
  /tmp/bench_s23 --rows 10000 --features 50 --classes 1 \
    --depth 6 --iters 50 --bins 128 --lr 0.1 --l2 3.0 --seed 42
done
# Expected: mix of 0.48231599 (ULP=0) and 0.48231912 (ULP=105) across runs
```

### Cascade mechanism

1-2 ULP drift in one histogram bin → near-tie GAIN flip at iteration k → different tree structure selected at k → all subsequent iterations diverge on a different trajectory → ~70× ULP amplification by iters=50.

### Why it hides at gate config

The gate config (N=50000/RMSE/128b) is 100/100 deterministic despite using the same `atomic_fetch_add` path. The Metal GPU scheduler at the gate dispatch shape resolves additions in a consistent order across runs. At N=10000 the partition size is ~156 docs at depth=6, placing threads in a near-tie accumulation race; at N=50000 the larger partition size (~781 docs) produces consistent ordering. This is NOT a proof of correctness — it is config-specific scheduler behavior.

### History

- Bug β was documented in `docs/sprint21/d1r4_synthesis.md §3` (parity risk) but retired in DEC-022 based on 10/10 and 100/100 determinism evidence — which was measured at gate config only.
- S22 D3 parity sweep ran 1-run-per-config, giving 50% miss probability per config for a ~50/50 race.
- S23 D0 5-run-per-config protocol detected the race at config #8.
- Pre-existing in S22 D2/D3 scratch tip `73baadf445`; not introduced by S23 D0 promotion.

### Fix options (from DEC-023)

1. **Threadgroup-local reduce + single-thread commit** (preferred): mirrors feat-0 design; known-clean mechanism; preserves T2 perf envelope. Each TG accumulates into threadgroup memory, one thread commits to global at end.
2. **Int-atomic fixed-point accumulation**: CatBoost CPU uses uint64 fixed-point for exactly this reason; deterministic by construction; requires accuracy calibration vs float.
3. **Kahan/Neumaier compensated summation**: mitigates but does NOT eliminate non-determinism (atomic order remains non-deterministic); probably not sufficient standalone.

**Kill-switch**: if chosen fix degrades gate-config ratio below 0.45×, escalate to structural redesign.

### Fix target

S24 D0, 1-2 days. See `DECISIONS.md DEC-023` and `docs/sprint23/d0_bimodality_verification.md`.

---

## BUG-007: nanobind path doesn't sort group_ids (silent divergence on unsorted ranking input)

**Discovered**: Sprint 12 review (@qa-engineer)
**Status**: OPEN — unscheduled
**Description**: The nanobind in-process path does not sort group_ids before training, causing silent divergence on unsorted ranking inputs compared to the CSV training path.

---

## bench_boosting K=10 anchor mismatch

**Discovered**: Sprint 7 (@qa-engineer)
**Status**: OPEN — unscheduled
**Description**: `bench_boosting` at K=10 produces BENCH_FINAL_LOSS=1.78561831 but expected anchor is 2.22267818. Flagged in Sprint 7, root cause unknown.
