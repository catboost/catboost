# S29-LG-SPIKE-T1 — DEC-034 mechanism spike

**Task:** #84 (Sprint 29, `mlx/sprint-29-dec032-closeout` branch)
**Scope:** data + harness ONLY. Verdict in #85 (S29-LG-SPIKE-T2).

## What this measures

Lossguide + Cosine iter-1 aggregate RMSE drift vs CPU CatBoost at
`N=1000, depth=3, max_leaves=8, seeds=[0,1,2], rs=0.0, bootstrap=no,
bins=128`. Used to discriminate between two hypotheses for DEC-034:

- **Outcome A — shared mechanism:** iter-1 mean drift small (~< 1.5%),
  comparable to the ST+Cosine iter-1 anchor of 0.77% captured in
  `docs/sprint28/fu-obliv-dispatch/t7-gate-report.md`. Priority-queue
  ordering agrees with CPU; drift is pure float32 leaf-value precision.
- **Outcome B — decoupled mechanism:** iter-1 mean drift >= 5%.
  Priority-queue leaf selection amplified float32 gain noise into a
  structurally different tree (splits flipped).
- **Outcome C — ambiguous:** 1.5% <= drift < 5%.

Secondary artifact extends to `iter in {1,2,5,10,25,50}` (drift curve);
tertiary artifact captures the iter-1 seed-0 tree structure on both
sides for a structural (feature-index) comparison.

## How to reproduce

**CRITICAL — requires a LOCAL guard bypass that is NOT committed in
this PR.** Both the Python layer (`python/catboost_mlx/core.py`
`_validate_params`) and the C++/nanobind layer
(`catboost/mlx/train_api.cpp` `TrainConfigToInternal`) reject
`score_function='Cosine'` combined with `grow_policy='Lossguide'`
(see Sprint 28 commits `b9577067ef` and Sprint 29 #82
`73e9460a31`). To run this harness, temporarily short-circuit BOTH
guards in your working copy (e.g. wrap the `if` branch with
`if False and ...`), rebuild `_core.so`, then re-run. `setup.py`
auto-copies the built `.so` into `python/catboost_mlx/` — no nanobind
cache trap to worry about.

```bash
# 1. Bypass both LG+Cosine guards in the working copy.
#    Python: python/catboost_mlx/core.py  (_validate_params LG+Cosine branch)
#    C++:    catboost/mlx/train_api.cpp   (TrainConfigToInternal LG+Cosine branch)

# 2. Rebuild the nanobind extension.
cd python && python setup.py build_ext --inplace

# 3. Run the harness.
cd ..
python docs/sprint29/lg-mechanism-spike/harness.py

# Artifacts refreshed in docs/sprint29/lg-mechanism-spike/data/.
```

Env toggles:
- `S29_SPIKE_SECONDARY=0` — skip drift-vs-iter curve
- `S29_SPIKE_TERTIARY=0`  — skip root-split tree-structure comparison

## Artifacts

- `data/iter1_drift.json` — **PRIMARY.** Per-seed + mean/std iter-1
  drift, config block, ST+Cosine anchor, outcome thresholds. The
  harness intentionally does NOT classify outcome A/B/C — that is
  #85's job.
- `data/iter_curve.csv` — **SECONDARY.** Rows
  `(iterations, seed, cpu_rmse, mlx_rmse, drift_pct)` for
  `iter in {1,2,5,10,25,50}`. Lets the verdict writer compare the
  curve shape against the ST+Cosine compounding curve (iter-1: 0.77%
  -> iter-50: ~47% in t7-gate-report) to corroborate or refute
  Outcome A.
- `data/tree_structure_iter1.json` — **TERTIARY.** At iter=1, seed=0,
  captures both CPU's `float_feature_index` / `border` tuple and MLX's
  `feature_idx` / `bin_threshold` for the root split, plus the BFS-
  ordered feature-index sequence across all splits. If the sequences
  match, priority-queue ordering did NOT diverge -> Outcome A evidence.

## What this harness does NOT include

- **The guard bypass.** It is local-only and must be re-applied by the
  reproducer.
- **The verdict.** The outcome classification is #85's authoritative
  deliverable; this harness writes data only.
- **Parity assertions.** This is a measurement spike, not a gate. No
  pass/fail thresholds are enforced.
- **A debug mode / config flag to disable the guards.** Per task spec
  #84, production code must not ship any such flag — the bypass is
  throwaway local state.
