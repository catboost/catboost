# Operation Verstappen — Sprint 17: Histogram Tree Reduction

## What this is

Sprint 17 is the first **structural kernel rewrite** of Operation Verstappen. Sprint 16 produced a clean diagnosis: the histogram Metal kernel is 97.5–99.2% of per-iteration time across all 18 measured configurations. Sprint 17 rewrites the kernel's threadgroup reduction from a 255-step serial loop to an 8-step log-step tree reduction.

See [`docs/operation-verstappen.md`](../operation-verstappen.md) for the full campaign roadmap.

---

## The lever

The serial reduction at `catboost/mlx/kernels/kernel_sources.h:160–181` has thread 0 seed a 1024-float staging buffer, then threads 1–255 each add their private histogram sequentially — one thread active per step, 254 threads idle, 255 `threadgroup_barrier` calls total. Replacing this with a balanced binary-tree reduction (8 levels for BLOCK_SIZE=256) keeps all 256 threads active at every level and reduces the barrier count from 255 to 8.

Source of truth for the diagnosis: [`docs/sprint16/mst_findings.md`](../sprint16/mst_findings.md) §B.3.

---

## Sprint 17 perf gate (committed in Sprint 16)

**≥30% reduction in `histogram_ms` at N=10k, RMSE, depth=6, 128 bins**, mean of 5 runs, before/after on same build and hardware.  
Full 18-config delta table in [`results.md`](results.md). No config may regress `histogram_ms` >5%.

---

## Documents

| File | Contents | Status |
|------|----------|--------|
| [`design.md`](design.md) | D1 tree-reduction design, variant analysis, storage trade-offs, numerical-stability rationale | Day 0 scaffold |
| [`ablation.md`](ablation.md) | Variant sweep (D1a / D1c) × block sizes × bin counts — filled by @research-scientist (S17-02) | Stub |
| [`results.md`](results.md) | 18-config before/after delta table — filled by @performance-engineer (S17-03) | Stub (before column populated) |
| [`non_goals.md`](non_goals.md) | Explicit Sprint 18+ deferrals | Day 0 scaffold |

---

## Same-PR docs standing order

All Sprint 17 source changes land in a single PR. That PR must include, in the same commit:
- `docs/sprint17/{README,design,ablation,results,non_goals}.md` (this directory)
- `CHANGELOG-DEV.md` Sprint 17 entry
- `catboost/mlx/ARCHITECTURE.md` histogram-reduction section updated
- `DECISIONS.md` DEC-005 (RMSE parity loosening) + DEC-006 (chosen D1 variant)
- `docs/sprint18/plan_prior.md` drafted from ablation findings

No doc, no merge.

---

## Sprint 16 starting point

Sprint 16 baseline (full table): [`docs/sprint16/baseline_results.md`](../sprint16/baseline_results.md).  
Representative anchor: **N=10k, RMSE, depth=6, 128 bins → `histogram_ms` = 308.20 ms (97.9% of iter time)**.
