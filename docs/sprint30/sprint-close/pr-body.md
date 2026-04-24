## Summary

- Sprint 30 executed the full phased T1→T4 Kahan fix plan (DEC-035): K4 fp64 cosNum/cosDen widening and Fix 2 fp64 gain/argmax widening both ship as logically correct precision fixes
- All primary T3 envelope gates failed (G3a 53.30% ST, G3b/G3c LG outside [0.98,1.02]); a 10-task verification battery (D1–D4, V1/V2/V5/V6, Fix 2) was run; V6 at N=500 produced 50.72% drift — flat across 100× N range, falsifying the entire precision-accumulation hypothesis class
- ST+Cosine and LG+Cosine guards remain in place; DEC-036 opens structural divergence investigation for S31 (iter=1 split-selection audit as entry point)

## What lands

- **K4 — fp64 cosNum/cosDen accumulators** (`108c7a59d2`-family): removes the ~4e-3 float32 accumulation floor; will be load-bearing once the structural mechanism is resolved
- **Fix 2 — fp64 gain/argmax** (`90a0cb4475`, `364d4ee962`): 15 sites widened across `FindBestSplit` and `FindBestSplitPerPartition`; prediction failed cleanly at 0.00% improvement; fix is correct and kept
- **13 verdict docs** under `docs/sprint30/`: full chain of evidence for precision-class exhaustion
- **`COSINE_RESIDUAL_INSTRUMENT` instrumentation** in `catboost/mlx/tests/csv_train.cpp`: compile-gated, retained for S31 audit reuse; release builds unaffected
- **DEC-035 PARTIALLY CLOSED**, **DEC-036 OPEN**, **DEC-034 PARTIALLY FALSIFIED** (state-files commit `24a0e829b8`)

## What does NOT land

- Guard removal (T4a/T4b) — mechanism not fixed; guards intact at all three layers (Python `_validate_params`, `train_api.cpp:TrainConfigToInternal`, `csv_train.cpp:ParseArgs`)

## Test plan

- [ ] All 13 verdict docs present under `docs/sprint30/` (t1-instrument, t2-kahan, t3-measure, d1-cpu-audit, d2-stack-instrument, d2-redux, d3-lg-outcome-ab, d4-joint-denom, v1-drift-vs-n, v2-d2-audit, v5-dw-at-scale, v6-n500-confirmer, fix2-fp64-gain)
- [ ] `grep -rn 'TODO-S29-LG-COSINE-RCA'` returns non-zero (LG guard intact at Python + nanobind + CLI + test)
- [ ] `grep -rn 'TODO-S29-ST-COSINE'` returns non-zero (ST guard intact at same four sites)
- [ ] `COSINE_RESIDUAL_INSTRUMENT` compile gate intact in release builds (not defined in `CMakeLists.txt`)
- [ ] Parity suite 28/28 on `tests/test_python_path_parity.py`
- [ ] `bench_boosting` v5 ULP=0 preserved (kernel sources untouched — `784f82a891`)
- [ ] `.claude/state/HANDOFF.md` reflects S30 CLOSING, S31 KICKOFF sections
- [ ] `.claude/state/DECISIONS.md` DEC-035 status reads PARTIALLY CLOSED, DEC-036 status reads OPEN
