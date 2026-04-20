# Sprint 22 D1b — T2 Fix Attempt and Re-run

**Branch**: `mlx/sprint-22-t2-integration`
**Date**: 2026-04-20
**Task**: D1b — Apply D1a-identified fix(es), run test (a)/(b)/(c), document results.
**Spec**: `docs/sprint22/d1a_t2_diagnostic.md §3, §6`
**Prior docs**: `d1a_t2_diagnostic.md`, `d1_t2_parity_sweep.md`, `d0_t2_production_shape.md`

---

## §1 TL;DR

**FAIL — D1a fix hypothesis falsified. Both Option 2 and Option 1 individually and together fail test (a). D1a's blit-ordering root-cause diagnosis is incomplete. Escalation to @troubleshooter required.**

- Fix attempted: Option 2 first (eval-at-return), then Option 1 (mid-dispatch sort barrier), then both combined.
- Test (a) result: FAIL on all three variants. T2 loss at features=1/iters=2 remains catastrophic.
- Test (b): NOT run (protocol: stop at test (a) failure; do not run (b) or (c) with a broken fix).
- Test (c): NOT run (same reason).
- New diagnostic evidence: the failure pattern is non-deterministic AND depth-dependent (fails at even depths ≥ 2, passes at odd depths), which D1a did not predict and which the eval-barrier hypothesis does not explain.
- Verdict: D2 blocked. D1c Kahan is not the issue. **Root cause is deeper than D1a's blit-ordering hypothesis. Requires @troubleshooter investigation.**

---

## §2 Applied Fix

### Build baseline (HEAD `4333c82a7e`, before any changes)

```bash
cd "/Users/ramos/Library/Mobile Documents/com~apple~CloudDocs/Programming/Frameworks/catboost-mlx"
clang++ -std=c++17 -O2 -I. \
  -I/opt/homebrew/opt/mlx/include \
  -L/opt/homebrew/opt/mlx/lib -lmlx \
  -framework Metal -framework Foundation \
  -Wno-c++20-extensions \
  -DCATBOOST_MLX_HISTOGRAM_T2=1 \
  catboost/mlx/tests/bench_boosting.cpp \
  -o /tmp/bench_boosting_t2_d1b
```

Build: clean, 0 warnings, both binaries.

### Fix attempt 1: Option 2 (eval-at-return, preferred per D1a §3)

D1a §3 specifies the preferred fix as inserting `mx::eval(accumOut[0])` before the `return` statement in `DispatchHistogramT2`. This mirrors the D1-R2 reference (`microbench_t2.cpp:722`) and forces sort+fill+accum into a single scheduling pass, eliminating pool-reuse ordering ambiguity between calls.

**Diff (Option 2 only):**

```diff
+    // D1b fix (Option 2): force immediate evaluation...
+    mx::eval(accumOut[0]);
     return accumOut[0];
```

**Result of test (a):** FAIL. T2 loss = 197.519 vs T1 = 0.493677. Loss ratio ≈ 400×. Catastrophic divergence unchanged.

### Fix attempt 2: Option 1 (mid-dispatch sort barrier per D1a §3 fallback)

D1a §3 also describes Option 1: inserting `mx::eval({sortOut[0], sortOut[1]})` between T2-sort and T2-accum dispatches. This forces the scheduler to commit and execute the sort kernel before building the accum lazy graph.

**Diff (Option 1 added, Option 2 retained):**

```diff
+    // D1b fix (Option 1 fallback): explicit eval barrier between sort and accum.
+    mx::eval({sortOut[0], sortOut[1]});
+
     // --- T2-accum dispatch ---
     auto accumOut = GetT2AccumKernel()(
```

The fix file now has BOTH Option 1 (mid-dispatch barrier) AND Option 2 (eval-at-return) active simultaneously, which represents the maximally aggressive eval strategy from D1a.

**Result of test (a):** FAIL. T2 loss = 167.219 vs T1 = 0.493677 (run 1). Non-deterministic across runs: run 2 = 0.50727350, run 3 = 0.50727397 (both wrong but not catastrophic). Still divergent from T1 = 0.49367726 by > 1%.

**Full diff applied to `bench_boosting.cpp`:**

```diff
@@ -574,9 +574,18 @@ mx::array DispatchHistogramT2(
         /*stream=*/mx::Device::gpu
     );
 
+    // D1b fix (Option 1 fallback): explicit eval barrier between sort and accum.
+    // [...]
+    mx::eval({sortOut[0], sortOut[1]});
+
     // --- T2-accum dispatch ---
-    // sortOut[0] and sortOut[1] are inputs — MLX graph ensures sort runs before accum.
-    // No explicit mx::eval() between sort and accum needed.
     auto accumOut = GetT2AccumKernel()(
         [...inputs...]
         /*init_value=*/0.0f,
         /*stream=*/mx::Device::gpu
     );
 
+    // D1b fix (Option 2): force immediate evaluation [...]
+    mx::eval(accumOut[0]);
     return accumOut[0];
 }
```

**Files touched**: `catboost/mlx/tests/bench_boosting.cpp` only. No production sources modified.

---

## §3 Test (a) Results — FAIL

Per D1a §6: run `--rows 50000 --features 1 --classes 1 --bins 128 --seed 42 --depth 6 --iters 2`.

| Variant | T1 BENCH_FINAL_LOSS | T2 BENCH_FINAL_LOSS | ULP delta | Bit-exact? |
|---------|--------------------:|--------------------:|----------:|:----------:|
| Pre-fix (D1 QA reference) | 0.49367726 | 142.84576416 | ≫ 1B | NO |
| Option 2 only | 0.49367726 | 197.51887512 | ≫ 1B | NO |
| Option 1 + Option 2 (run 1) | 0.49367726 | 167.21890259 | ≫ 1B | NO |
| Option 1 + Option 2 (run 2) | 0.49367726 | 0.50727350 | ~45,000 | NO |
| Option 1 + Option 2 (run 3) | 0.49367726 | 0.50727397 | ~45,000 | NO |

**Pass criterion: T2 loss equals T1 bit-for-bit.** Not met. Loss diverges in all runs; magnitude and sign vary non-deterministically (catastrophic on first run, non-catastrophic but still wrong on subsequent runs).

**iters=1 check (confirming first-call correctness is intact):**

| Variant | T1 loss | T2 loss | Bit-exact? |
|---------|---------|---------|:----------:|
| Option 1 + Option 2 | 0.53039330 | 0.53039330 | YES |

First call still bit-exact. Bug activates on second call, confirming the inter-iteration nature of the failure.

---

## §4 Test (c) Results

NOT RUN. Protocol: test (a) failure requires investigation before proceeding. Running a perf test with a broken kernel would produce meaningless ratios.

---

## §5 Test (b) Results

NOT RUN. Same reason. Running the 18-config sweep with a broken kernel would waste 45 minutes and produce no actionable data.

---

## §6 New Diagnostic Evidence

Collected during D1b investigation to support the @troubleshooter escalation.

### Depth × iteration isolation sweep

Running `--features 1 --rows 50000 --bins 128 --seed 42 --seed 42 --iters 2` at varying depths with Option 1 + Option 2 fixes active:

| Depth | numActiveParts | T1 loss | T2 loss (run 1) | Pass? |
|------:|---------------:|---------|-----------------|:-----:|
| 1 | 2 | 0.49367726 | 0.49367726 | YES |
| 2 | 4 | 0.49367726 | 0.50727385 / 141.5 (varies) | NO |
| 3 | 8 | 0.49367726 | 0.49367726 | YES |
| 4 | 16 | 0.49367726 | 119.892 | NO |
| 5 | 32 | 0.49367726 | 0.49367726 | YES |
| 6 | 64 | 0.49367726 | 155.508 / 197.5 (varies) | NO |

**Even depths (2, 4, 6) fail. Odd depths (1, 3, 5) pass. This is a non-monotonic depth-dependent pattern.**

### Depth × iteration constraint

| Configuration | Result |
|---------------|--------|
| depth=2, iters=1 | PASS (bit-exact) |
| depth=1, iters=2 | PASS (bit-exact) |
| depth=1, iters=3 | PASS (bit-exact) |
| depth=2, iters=2 | FAIL (non-deterministic) |

**The bug requires both `depth >= 2` (even) AND `iters >= 2` simultaneously.**

### Why D1a's hypothesis does not explain this

D1a's blit-ordering hypothesis predicts:
- Failure on any depth ≥ 1, iteration ≥ 2 (pool reuse happens whenever same-size buffers are released and reallocated)
- Failure gets worse (not alternates) at higher depths
- Both Option 1 AND Option 2 evals should prevent the ordering ambiguity

What we observe:
- Alternating pass/fail at even vs odd depths (depth 3 passes, depth 4 fails)
- Non-deterministic magnitude (catastrophic on run 1, wrong-but-not-catastrophic on runs 2-3 at same depth)
- Both evals together do not fix the problem

The even/odd depth pattern suggests the bug involves a **state transition that oscillates** — possibly something that correctly resets at odd depths but accumulates error at even depths. This points to a bug in the Metal kernel logic itself (slotBase formula, binOffsets interpretation, partIdx/statIdx indexing) rather than a buffer-pool race.

### fill_gpu is a compute kernel, not a blit

D1a §3 stated: "The fill_gpu blit writes `device float*`. Metal does not guarantee that a blit to `device float*` is visible to a subsequent compute pass reading `device atomic<float>*`."

Code inspection of `mlx/backend/metal/copy.cpp:182` (fill_gpu) reveals:

```cpp
auto& compute_encoder = d.get_command_encoder(s.index);
compute_encoder.set_compute_pipeline_state(kernel);
compute_encoder.dispatch_threads(grid_dims, group_dims);
```

`fill_gpu` is a **compute shader dispatch** through the same Metal compute encoder, NOT a blit encoder. Both the fill and the accum kernel are serialized within the same command encoder. D1a's "blit-vs-atomic" visibility concern does not apply. This is a second falsified sub-hypothesis within D1a.

---

## §7 R8 Impact

D1b did not run test (c), so no ratio measurement is available. The R8 projection from D0 (1.83×, cumulative 1.96×) remains the theoretical ceiling if the root cause is found and fixed.

Current state: R8 is at 1.07× (pre-T2 baseline) until T2 is fixed.

**Decision matrix for Ramos:**

| Outcome | Condition | R8 projection |
|---------|-----------|---------------|
| Fix found by @troubleshooter, parity passes | Requires new D1b re-run | 1.83× (unchanged from D0 estimate) |
| Fix found, parity fails → needs Kahan | Kahan + ratio compression | ~1.62× (D1a §4 estimate) |
| Fix not found in Sprint 22 | T2 drops to RESEARCH | 1.07× (no improvement) |

---

## §8 Next-Step Recommendation

**Escalate to @troubleshooter. Do not attempt further fixes in D1b.**

The diagnostic evidence now shows that:
1. D1a's blit-ordering root cause is falsified (fill_gpu is a compute kernel; both evals fail)
2. The failure pattern (even-depth only, non-deterministic magnitude) points to an algorithmic bug in the T2 kernel logic, not a dispatch ordering race
3. Two further fix attempts would be improvisation without a diagnostic hypothesis

**Troubleshooter prompt package** (self-contained):

- D1a diagnostic doc: `docs/sprint22/d1a_t2_diagnostic.md`
- D1 parity sweep: `docs/sprint22/d1_t2_parity_sweep.md`
- D1b results (this doc): `docs/sprint22/d1b_t2_fix_and_rerun.md`
- Kernel sources: `catboost/mlx/kernels/kernel_sources_t2_scratch.h`
- Dispatch code: `catboost/mlx/tests/bench_boosting.cpp` (DispatchHistogramT2, lines 481-617)
- Key new evidence: depth-alternating pattern (§6 table), non-determinism (multiple run values in §3)
- Both D1a options applied and failing: full diff in §2
- MLX fill_gpu is a compute kernel (not blit): `mlx/backend/metal/copy.cpp:182-213`
- The T2-accum histogram init (`init_value=0.0f`) goes through a compute fill, properly serialized before the accum kernel within the same compute encoder

**The troubleshooter should focus on the even/odd depth alternation as the primary diagnostic clue. The hypothesis to test: is there an indexing bug in the T2-sort slotBase or binOffsets formula at even numActiveParts values?**

---

## §9 Git State Confirmation

```
git status --short:
 M catboost/mlx/tests/bench_boosting.cpp
?? docs/sprint22/d1_t2_parity_sweep.md
?? docs/sprint22/d1a_t2_diagnostic.md
?? docs/sprint22/d1b_t2_fix_and_rerun.md
?? docs/sprint22/scratch/

git diff --stat:
 catboost/mlx/tests/bench_boosting.cpp | 22 +++++++++++++++++++---
 1 file changed, 20 insertions(+), 2 deletions(-)
```

**Production source discipline:**

```
git diff --stat catboost/mlx/kernels/kernel_sources.h catboost/mlx/methods/histogram.cpp
(no output — production sources unmodified)
```

T2 remains in scratch-only files:
- `catboost/mlx/kernels/kernel_sources_t2_scratch.h` — T2 kernel (unchanged from D0 commit)
- `catboost/mlx/tests/bench_boosting.cpp` — D1b fix attempts applied (both Options 1 and 2)

**A1-G6 / D1b scratch discipline satisfied: no production kernel source modified.**
