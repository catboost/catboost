# Sprint 17 Non-Goals

Items explicitly out of scope for Sprint 17. Each has a designated sprint or condition for pickup. Do not add these to Sprint 17 acceptance criteria or the Sprint 17 PR.

---

## D2 — `privHist[1024]` per-thread private-storage redesign

**What it is:** Replace the per-thread 1024-float stack array (`kernel_sources.h:123`) with a SIMD-group-sliced threadgroup design. This eliminates the ~4 KB/thread spill to device memory that likely contributes significantly to kernel wall time.

**Why deferred:** D1 (tree reduction) is the prerequisite. D2's gain can only be correctly attributed after D1 removes the serial-reduction overhead. D2 also requires its own BUG-001 parity campaign (the original shared-histogram design hit a non-determinism bug; a modern rewrite needs careful validation). Interacts with D1c's SIMD-shuffle variant — if D1c ships, some of this work is partially shared.

**Target:** Sprint 18.

---

## `histogram.cpp:105` `maxBlocksPerPart` retuning

**What it is:** `catboost/mlx/methods/histogram.cpp:105` hardcodes `maxBlocksPerPart = 1`. This is the **library path**, which is dead code for `csv_train` (and therefore the current Python bindings and CLI). The production path at `csv_train.cpp:891–894` already computes `maxBlocksPerPart` dynamically.

**Why deferred:** The library path is dead code and fixing it has no performance impact today. A cleanup issue should either delete the library path or resync it with the production kernel. Not worth Sprint 17 blast radius.

**Target:** Sprint 19, or as a standalone cleanup issue before that.

---

## Multiclass per-dim dispatch fusion

**What it is:** Fuse the `approxDim` serial loop at `csv_train.cpp:3185–3204` (which serialises 3 `DispatchHistogram()` calls for MultiClass) into a single kernel dispatch using the Z grid dimension. Would eliminate ~2× kernel-encoding overhead for MultiClass and allow all three command buffers to schedule concurrently.

**Why deferred:** Wide blast radius — touches stats array layout, leaf accumulator, scorer. Expected gain is bounded to MultiClass only (~2× on that task subset). D1 must land and be measured first; D5 may or may not close the remaining gap.

**Target:** Sprint 18 (after D1 baseline is established).

---

## Per-feature-group kernel fusion (library path)

**What it is:** Fuse the per-feature-group serial dispatch loop at `histogram.cpp:112–155`. This is the library-path analogue of the per-dim multiclass issue above.

**Why deferred:** Library path is dead code for `csv_train`. Only relevant if/when the library path is restored as the production path.

**Target:** Sprint 18/19, contingent on library path being revived.

---

## M1/M2 validation

**What it is:** Confirm tree-reduction correctness and performance on M1 and M2 chips. D1c (SIMD-shuffle) in particular makes assumptions about SIMD width and `simd_shuffle_xor` behaviour that may differ subtly across Apple Silicon generations.

**Why deferred:** Sprint 17 targets M3 exclusively. M1/M2 testing adds hardware dependency and test matrix complexity that is not warranted until the M3 variant is stable.

**Target:** Sprint 18, after the chosen D1 variant ships on M3.

---

## D3 — Pre-permuted stats/compressedData layout

**What it is:** A preprocessing pass that writes stats and packed features in `docIndices`-permuted order before histogram dispatch, removing the gather indirection at `kernel_sources.h:132–133`. Expected gain: 10–20% on `histogram_ms` at depth ≥ 3.

**Why deferred:** Requires a layout change upstream of the histogram kernel (partition layout, data transfer). Secondary after D1 and D2.

**Target:** Sprint 18+.
