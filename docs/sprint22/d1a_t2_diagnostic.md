# Sprint 22 D1a — T2 Dispatch Diagnostic

**Branch**: `mlx/sprint-22-t2-integration`  
**Date**: 2026-04-20  
**Task**: D1a — Read-only diagnostic of T2 sort/accum dispatch path. No implementation.  
**Hypothesis under test**: Product Owner two-bug decomposition — bug α (structural dep/init) + bug β (atomic-scatter drift).

---

## §1 TL;DR

**Bug α classification: histogram-init (output buffer not zeroed between iterations).**

Not dep-wiring. `sortOut[0]`/`sortOut[1]` ARE properly wired as MLX array outputs of T2-sort and inputs to T2-accum (`bench_boosting.cpp:580-594`). MLX's lazy graph DOES sequence them correctly because T2-accum lists `sortOut[0]` and `sortOut[1]` as its input arrays, so the graph topologically orders T2-sort before T2-accum. The dep-wiring hypothesis is **rejected**.

The bug is that the `histogram` output buffer allocated for the `accumOut` kernel call is **not zeroed before T2-accum executes** in D0's `DispatchHistogramT2`. T2-accum uses `atomic_fetch_add_explicit` to accumulate into the histogram. With `init_value=0.0f` passed to `GetT2AccumKernel()`, MLX calls `fill_gpu` to zero-initialize the histogram before the kernel runs — BUT this `fill_gpu` is a GPU-side blit that is enqueued into the **same stream** as the kernel. On the first call the buffer is freshly allocated and the blit-then-kernel sequence is correct. On subsequent calls, MLX's allocator (`allocator::malloc`) may return a **buffer from the pool that was released by the prior iteration**. That prior buffer already contains non-zero histogram values from the last call. Whether `fill_gpu` runs before T2-accum reads those values depends on whether the blit and the kernel are in the same Metal command buffer. If they are split across command buffers (which happens when `command_buffer_needs_commit` triggers at the `fill_gpu` step), the kernel can begin reading before the blit completes.

The stronger structural cause: T2-accum is `atomic_outputs=true`, which changes the MLX-generated Metal signature to `device atomic<float>* histogram`. The `fill_gpu` blit writes `device float*`. Metal does not guarantee that a blit to `device float*` is visible to a subsequent compute pass reading `device atomic<float>*` at the same address unless an explicit memory barrier or command-buffer boundary enforces it. In practice, within a single command buffer Metal serializes blits and compute dispatches in submission order — but if `fill_gpu` and the kernel dispatch land in different command buffers, that serialization is absent.

**Bug β assessment: present, secondary.** The 0.13% end-to-end drift at gate config (50k/RMSE/128b, iters=50) after the stale-histogram failure mode is also present via the atomic-scatter path on features 1–3. D1 Experiment 2 shows 5-run non-determinism in the 5th decimal place even at the gate config — this is the per-doc scatter nondeterminism pre-budgeted by `d1r4_synthesis.md §3`. Once bug α is fixed, bug β will be the remaining barrier to DEC-008 RMSE ulp≤4.

---

## §2 Evidence

### Q1 — T2-sort output wiring

| File:Line | Finding |
|-----------|---------|
| `kernel_sources_t2_scratch.h:37` | Header comment: "Output names: sortedDocs, binOffsets" |
| `kernel_sources_t2_scratch.h:103-104, 110-111` | Kernel body writes `sortedDocs[slotBase + pos]` and `binOffsets[offBase + b]` — these are the named Metal output buffers |
| `bench_boosting.cpp:453` | `GetT2SortKernel` declaration: `/*output_names=*/{"sortedDocs", "binOffsets"}` |
| `bench_boosting.cpp:562-575` | `sortOut = GetT2SortKernel()(...)` — call returns `std::vector<mx::array>` with two elements |
| `mlx/backend/metal/custom_kernel.cpp:308-323` | `array::make_arrays(output_shapes, output_dtypes, primitive, inputs)` — each output is a lazy `mx::array` whose primitive IS the CustomKernel; the inputs vector IS the dispatch's input arrays |

T2-sort writes to `sortedDocs` and `binOffsets` as declared MLX output arrays. They are NOT static globals or raw `MTLBuffer`. They are proper `mx::array` objects returned by the kernel call. **Output wiring is structurally correct.**

### Q2 — T2-accum input wiring

| File:Line | Finding |
|-----------|---------|
| `bench_boosting.cpp:580-594` | `accumOut = GetT2AccumKernel()({sortOut[0], sortOut[1], flatCompressed, stats, ...})` — `sortOut[0]` and `sortOut[1]` are the SAME `mx::array` objects returned by `GetT2SortKernel()` |
| `bench_boosting.cpp:465-467` | `GetT2AccumKernel` input_names list: `{"sortedDocs", "binOffsets", ...}` — these match the names declared as sort outputs |
| `mlx/backend/metal/custom_kernel.cpp:308-323` | `array::make_arrays(..., std::move(inputs))` — inputs are stored as the primitive's input edges; when the accum node is evaluated, the graph scheduler evaluates its inputs (sort node) first |

`sortOut[0]` and `sortOut[1]` ARE wired as direct inputs to the accum kernel call. MLX's dependency graph DOES know that accum depends on sort. **Dep-wiring hypothesis rejected.** The graph ordering is correct.

### Q3 — Histogram buffer zero-init and init_value semantics

| File:Line | Finding |
|-----------|---------|
| `bench_boosting.cpp:589-594` | T2-accum call: `init_value=0.0f` — this is passed to `GetT2AccumKernel()` |
| `mlx/backend/metal/custom_kernel.cpp:338-344` | `eval_gpu`: if `init_value_` is set, calls `fill_gpu(copies.back(), out, s)` — GPU-side blit to zero the output buffer; if not set, calls `allocator::malloc(out.nbytes())` (uninitialized) |
| `mlx/backend/metal/custom_kernel.cpp:339-341` | `fill_gpu` is an enqueued GPU operation on stream `s` — not a CPU memset |
| `mlx/backend/metal/eval.cpp:59-67` | `command_buffer_needs_commit` may commit the current command buffer mid-stream; if this fires between `fill_gpu` and the kernel dispatch, the blit and the kernel execute in separate Metal command buffers |
| `bench_boosting.cpp:456-458` | T2-sort call: `init_value=0.0f` — sortedDocs and binOffsets are also zero-initialized; this is **harmless for sort** since the kernel overwrites all written positions, but it does add the same blit-sequencing risk |
| `bench_boosting.cpp:572` | T2-sort: `init_value=0.0f` passed |
| `bench_boosting.cpp:392` | T1 `DispatchHistogram`: `init_value` not shown here but T1 also passes `0.0f` per line 417 — T1 is a single kernel so the blit-then-kernel ordering risk doesn't compound across two kernels |

**Critical finding**: T2-accum uses `atomic_fetch_add_explicit` on the histogram buffer. The histogram output must be zero before the kernel starts. `init_value=0.0f` triggers `fill_gpu`, which is a GPU blit enqueued to stream `s`. This blit-then-kernel sequence is safe IF they share a command buffer. MLX commits command buffers when `command_buffer_needs_commit` returns true — which is implementation-defined and can fire between the sort dispatch, the fill blit, and the accum dispatch. There is no explicit `mx::eval()` between sort and accum in `DispatchHistogramT2` (see `bench_boosting.cpp:578-579` comment: "No explicit mx::eval() between sort and accum needed") — this comment was correct for dep-ordering but WRONG about the fill blit ordering.

**Contrast with D1-R2 reference** (`microbench_t2.cpp:722`): `mx::eval(accumOut[0])` is called inside `runT2()` immediately after building the graph. This forces both sort and accum to execute in a single scheduling pass. The fill blit, sort kernel, and accum kernel are committed in the same pass and share the same command buffer sequence. In D0's `DispatchHistogramT2`, there is no `mx::eval()` — the returned `accumOut[0]` is only evaluated later when `RunIteration` calls `mx::eval(histogram)` at `bench_boosting.cpp:1214`. Between successive iterations, the lazy graph is re-built but the histogram output buffer from the PREVIOUS iteration may be reused by the allocator for the current iteration's `fill_gpu` target — and if a command-buffer commit boundary separates the sort of iteration N from the fill of iteration N, the fill of iteration N writes the buffer that sort of iteration N-1 already released.

**Simpler statement of the bug**: `DispatchHistogramT2` returns a lazy `mx::array` without evaluating it. The caller evaluates it at `mx::eval(histogram)`. At that point, sort, fill, and accum are dispatched together. On the FIRST call this is fine — fresh allocation. On the SECOND call, `allocator::malloc` may return the same physical buffer released by iteration 1's histogram. The fill is supposed to zero it before accum reads it. But the fill is a GPU blit and accum is a compute dispatch in the SAME lazy evaluation pass. If they are placed in the same command buffer, they ARE ordered. If MLX splits them across command buffers (due to `command_buffer_needs_commit` firing), the accum kernel may start before the fill blit completes.

However, there is a more direct second failure mode: the T2-sort kernel writes `sortedDocs` and `binOffsets` with `init_value=0.0f` (fills first, then kernel overwrites). For slots that belong to empty partitions (`partSize == 0`), the kernel returns early and the slot retains the fill value of 0. For the SECOND iteration, if `sortedDocs` is reused from iteration 1 (same size allocation), the fill-then-kernel sequence again depends on command-buffer ordering. Empty-partition slots from iteration 2 should be 0 (fill), but if the fill doesn't complete before accum reads, iteration 2's accum reads iteration 1's garbage `docIdx` values for those slots. This explains the catastrophic divergence at features=1/iters=2: at `numActiveParts=1` (depth=0), `maxPartDocs = numDocs = 50000`, and iteration 2's sortedDocs slot is `numTGs * maxPartDocs` elements — a large buffer that gets pooled and potentially reused.

**iters=1 bit-exact explanation**: iteration 1 uses freshly `allocator::malloc`'d buffers (or new pool entries with no prior content). The fill blit runs, sort runs, accum runs — all correct. No prior-iteration garbage exists.

**bins=2/4 self-heal explanation**: with `foldCount <= 4`, the T2-sort kernel's `tgCounts` loop runs `for b in [0..127]` but only bins 0–4 are populated. The counting sort produces `sortedDocs` with real docs in positions 0..partSize-1 and zeros in positions partSize..(maxPartDocs-1). If iteration 2 reuses the same physical buffer but the fill doesn't complete, the accum kernel reads the iteration-1 sortedDocs. For bins=2/4, each doc falls into only 2 or 4 bins — the range covered by `binOffsets[offBase+b]` to `binOffsets[offBase+b+1]` is the same block of docIdx values regardless of iteration (docs are the same 50k docs; only their partition assignment changes). The histogram sum over those docIdx values is approximately the same even with stale binOffsets — because with 2 bins the partition layout change between iterations barely changes which docs fall in bin 1 vs bin 2. Hence the result "self-heals" to approximately correct at bins=2/4. At bins=128, the partition layout change between iterations is substantial, so stale binOffsets produce wildly wrong range boundaries.

### Q4 — Static/module-scope state

| File:Line | Finding |
|-----------|---------|
| `bench_boosting.cpp:442-479` | `GetT2SortKernel()` and `GetT2AccumKernel()` use `static auto k = mx::fast::metal_kernel(...)` — the kernel OBJECTS are static, initialized once per process |
| `bench_boosting.cpp:445-460` | Static kernel registration is safe and matches the D1-R2 reference pattern; `metal_kernel()` is idempotent |
| `kernel_sources_t2_scratch.h:63-65` | `threadgroup atomic_uint tgCounts[128]` — initialized at TG entry with `atomic_store_explicit(..., 0u)` inside the loop; `threadgroup_barrier` follows — correct per-dispatch initialization |
| `bench_boosting.cpp:499-513` | `foldCountsFlat`, `firstFoldIndicesFlat`, `featureColIndices` are stack-allocated vectors, rebuilt fresh each call — no static state in fold metadata |

No static buffers or global GPU state. The `threadgroup` arrays are properly initialized each kernel invocation. The only cross-call state is the MLX buffer pool — which is by design but whose reuse interacts with the blit-ordering issue above.

### Q5 — D0 dispatch vs D1-R2 reference structural diff

| Dimension | D1-R2 reference (`microbench_t2.cpp`) | D0 `DispatchHistogramT2` (`bench_boosting.cpp`) |
|-----------|---------------------------------------|--------------------------------------------------|
| `mx::eval` after accum | `mx::eval(accumOut[0])` inside `runT2()` (line 722) | None inside `DispatchHistogramT2`; caller evals at line 1214 |
| Number of dispatches per test | 1 dispatch per measurement, single iteration | Up to 6 dispatches per training iteration × 50 iterations = 300 dispatches |
| Buffer reuse opportunity | None (first-time allocation each `runT2()` call because MLX eval happens immediately) | High — `numTGs * maxPartDocs` uint32 buffer is ~5.2 MB; pool can and does reuse after eval |
| `init_value=0.0f` for sort | 0.0f (line 712) | 0.0f (line 572) — identical |
| `init_value=0.0f` for accum | 0.0f (line 721) | 0.0f (line 592) — identical |
| `atomic_outputs` for accum | `true` (line 687) | `true` (line 476) — identical |

**The structural difference is solely the absence of an inner `mx::eval()` in D0's dispatch function.** D1-R2 evaluated immediately after building the two-kernel graph, forcing the blit+sort+blit+accum sequence into a single scheduling pass with no pool-reuse opportunity. D0 defers evaluation to the caller, across a boundary that includes `ComputePartitionLayout`, `ComputeLeafSumsGPU`, and other GPU work — any of which may trigger `command_buffer_needs_commit`, split the command buffer, and allow buffer pool reuse between iterations.

### Q6 — Inter-iteration state

| File:Line | Finding |
|-----------|---------|
| `bench_boosting.cpp:1122-1124` | `partitions = mx::zeros(...)` + `mx::eval(partitions)` — partitions are reset at start of each `RunIteration` call |
| `bench_boosting.cpp:1138` | `statArr` is rebuilt from gradients each iteration — no carry-over |
| `bench_boosting.cpp:1242-1258` | `histogram` is a local `mx::array` in the depth loop — new lazy array each depth level |
| `bench_boosting.cpp:562-597` | `sortOut`, `accumOut` are locals in `DispatchHistogramT2` — new lazy arrays each call |
| `mlx/backend/metal/custom_kernel.cpp:343` | `out.set_data(allocator::malloc(out.nbytes()))` — without `init_value`, buffer is uninitialized pool memory |
| `mlx/backend/metal/custom_kernel.cpp:339-341` | With `init_value`, `fill_gpu` is called — but GPU-side, sequencing depends on command-buffer boundaries |

The ONLY inter-iteration state is the buffer pool. `mx::array` objects are local and die after `mx::eval`. Their backing GPU buffers are returned to the pool. Next iteration, `allocator::malloc` returns them. For `sortedDocs` (uint32, same size each depth level with fixed `maxPartDocs`) and the `histogram` buffer (float32, same size each depth level), pool reuse is guaranteed at depth > 0 of iteration 2. The fill blit for the histogram is the only thing standing between stale iteration-1 values and the accum kernel. If the fill blit is split from the accum dispatch by a command-buffer commit, stale values survive.

---

## §3 Proposed Fix for Bug α (DO NOT implement)

**Classification**: histogram-init / blit-command-buffer-ordering.

**Fix**: Add `mx::eval({sortOut[0], sortOut[1]})` inside `DispatchHistogramT2`, between the sort dispatch and the accum dispatch — i.e., between lines 575 and 580 of `bench_boosting.cpp`.

```
bench_boosting.cpp, after line 575 (end of T2-sort dispatch block):
    insert: mx::eval({sortOut[0], sortOut[1]});
```

This forces the scheduler to commit and execute the sort kernel (and its preceding fill blit) before building the accum lazy graph. The histogram buffer allocated for accum will then be a fresh allocation (since sort's sortedDocs/binOffsets have already been eval'd and released or are still live), and the fill_gpu + accum dispatch will be in a fresh command buffer with no ordering ambiguity.

**Files to touch**: `bench_boosting.cpp` only (scratch-only; D0 discipline maintained).

**Approximate lines**: insert one line after `bench_boosting.cpp:575`.

**Fix type**: barrier insertion (one `mx::eval`) — NOT a graph-rewire, because the dep-wiring is already correct.

**Expected perf cost**: non-zero but small. This forces two Metal command-buffer commits per histogram dispatch instead of one (one for sort, one for accum) plus a CPU-side scheduler round-trip between them. D0 measured T2 histogram_ms = 7.0 ms. The overhead of one additional `mx::eval` synchronization point is comparable to the T2-sort kernel time itself (~30–40% of T2 histogram_ms, estimated ~2–3 ms). This may push the T2/T1 ratio from 0.328× toward 0.40–0.45× — still well below the 0.60 kill-switch.

**Alternative fix (zero-cost, if feasible)**: Restructure `DispatchHistogramT2` to call `mx::eval(accumOut[0])` before returning (mirroring D1-R2 reference line 722). This evaluates sort+accum together in a single scheduling pass, eliminating pool reuse between calls. Cost: zero perf overhead relative to the current deferred-eval path, because the histogram is always evaluated immediately by the caller anyway (`bench_boosting.cpp:1214`). This is the preferred fix.

---

## §4 Proposed Bug β Fix (if applicable)

**Bug β assessment**: present, confirmed secondary. After bug α is fixed and histograms are correct, end-to-end ULP will revert to D1-R2 levels (~64 ULP per-bin). D1 Experiment 2 shows 5-run nondeterminism at the gate config (50k/RMSE/128b) even in the modest ~0.13% regime — this is the atomic-scatter nondeterminism from features 1–3 in T2-accum (`kernel_sources_t2_scratch.h:196-207`).

`d1r4_synthesis.md §3` pre-budgeted Kahan compensation for this case: "If end-to-end fails, Kahan-compensated summation on the per-doc scatter path is the path forward… Budget: +2–3 days if Kahan needed."

**Proposed Kahan fix**: In `kT2AccumSource`, for features 1–3, replace the single `atomic_fetch_add_explicit(dst, s, ...)` with a two-stage Kahan compensated accumulation. Because the output is `device atomic<float>*`, a true Kahan sequence (which requires reading back the current sum to compute the compensation) requires either (a) convert the features 1–3 path to a two-pass approach (first reduce per-bin with Kahan in threadgroup memory, then one atomic add per bin per TG), or (b) accept nondeterminism for features 1–3 and evaluate whether end-to-end ULP compounding stays within DEC-008 after α is fixed.

Option (b) is the correct first test: run D1b after the α fix, measure end-to-end ULP on all 18 configs. If RMSE/Logloss ulp ≤ 4 and MultiClass ulp ≤ 8 are met, Kahan is unnecessary. The 64 ULP per-bin in D1-R2 is measured at a single dispatch; end-to-end compounding over 50 iters × 6 depths × iters of gradient updates may or may not amplify to DEC-008 threshold — this is exactly what D1b measures.

If D1b still fails after α fix: apply Kahan to features 1–3 scatter path in `kernel_sources_t2_scratch.h:194-207`. Add a threadgroup-local Kahan compensator array `threadgroup float tgKahan[128]` (initialized to 0.0f), compute `y = s - tgKahan[b]`, `t = localSum + y`, `tgKahan[b] = (t - localSum) - y`, then issue the `atomic_fetch_add` with `t`. Cite: `d1r4_synthesis.md §3 T2 Risks` (Kahan fallback plan pre-budgeted). Files: `kernel_sources_t2_scratch.h` only. Budget: 2–3 days as pre-budgeted.

---

## §5 Rejected Hypotheses

| PO hypothesis | Status after code read | Evidence |
|---------------|------------------------|----------|
| Bug α = dep-wiring (`sortedDocs`/`binOffsets` not wired as MLX outputs) | **REJECTED** | `bench_boosting.cpp:562-575, 580-594`: sortOut[0]/[1] ARE MLX arrays returned by sort kernel call and passed as inputs to accum call; graph dependency is correct |
| Bug α = two distinct failure modes requiring different root causes | **PARTIALLY REJECTED / REFINED**: one root cause (blit-sequencing on buffer-pool reuse) explains BOTH the catastrophic features=1/iters=2 failure AND the 0.13% drift at gate config. The magnitude difference is explained by different buffer sizes and depth counts, not different bugs. | See Q3 and Q6 analysis |
| bins=2/4 "self-heal" rules out atomic-scatter as the cause | **CONFIRMED with refined explanation** | See Q3: bins=2/4 self-heal is explained by stale `binOffsets` producing approximately correct range boundaries when foldCount is very small (only 2–4 distinct bins over 50k docs, partition layout change is negligible). This is consistent with the histogram-init root cause, not a separate bug |
| `mx::eval()` may be wrong fix even if diagnosis right | **PARTIALLY CONFIRMED**: the preferred fix is `mx::eval(accumOut[0])` before return (mirrors D1-R2 reference), not a mid-dispatch barrier. The current code correctly wires the graph but fails to force evaluation before returning, allowing pool reuse. The fix IS an eval insertion but at the function exit, not mid-dispatch. | `bench_boosting.cpp:578-579` comment "No explicit mx::eval() needed" is correct for dep-ordering but wrong for buffer-lifecycle safety |

---

## §6 Next-Phase Gate (D1b)

After implementing the α fix (`mx::eval(accumOut[0])` before return in `DispatchHistogramT2`):

**(a) features=1 iters=2 bit-exact check**
```bash
/tmp/bench_boosting_t2 --rows 50000 --features 1 --bins 128 --seed 42 --depth 6 --iters 2 --t2
```
Pass criterion: T2 loss equals T1 loss exactly (bit-for-bit). If still catastrophic (>10% divergence), the α fix is incomplete and deeper investigation is required.

**(b) Gate config iters=50 ULP check against DEC-008**
Full 18-config sweep per D1 methodology (`docs/sprint22/d1_t2_parity_sweep.md §3`):
```bash
for each config in DEC-008 matrix:
    /tmp/bench_boosting_t2 --rows N --loss L --bins B --features 50 --depth 6 --iters 50 --lr 0.1 --l2 3.0 --seed 42 --t2
```
Pass criterion: RMSE/Logloss ulp ≤ 4, MultiClass ulp ≤ 8 on all 18 configs. If any RMSE/Logloss config fails with ulp in range 5–200, Kahan (§4) is needed. If ulp > 1000 on any config, the α fix was insufficient.

**(c) Gate config perf ratio check against 0.60 kill-switch**
```bash
/tmp/bench_boosting_t2 --rows 50000 --features 50 --classes 1 --depth 6 --iters 50 --bins 128 --seed 42 --per-kernel-profile --t2
```
Pass criterion: T2/T1 histogram_ms ratio ≤ 0.60 (kill-switch). Expected post-fix ratio: 0.40–0.50× (one extra `mx::eval` adds ~one command-buffer round-trip vs the deferred-eval path). A ratio above 0.60 would mean the eval barrier cost is larger than expected and T2 should be abandoned.

D1b GATE PASS = all three checks pass. D1b GATE FAIL on (c) with ratio 0.45–0.60 = ship with reduced projection (new e2e speedup recalculated from measured ratio). D1b GATE FAIL on (c) with ratio > 0.60 = T2 abandoned per kill-switch.
