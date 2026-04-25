# Architectural Decision Log — CatBoost-MLX

> See also: `.claude/state/DECISIONS.md` for operational decisions (DEC series).

Each decision follows the format: **Context** (why we faced this choice), **Decision** (what we chose), **Rationale** (why), **Risks** (what could go wrong).

---

## ADR-001: Use MLX as GPU Backend Instead of Raw Metal

**Date**: 2026-03-29
**Status**: Accepted

**Context**: We need to run CatBoost on Apple Silicon GPU. Options: (a) write raw Metal shaders from scratch, (b) use MLX's framework which already handles Metal device management, memory allocation, kernel dispatch, and JIT compilation.

**Decision**: Use MLX as the GPU abstraction layer, with custom Metal kernels registered through MLX's `metal_kernel` / custom primitive system.

**Rationale**:
- MLX already solves Metal device management, command encoding, memory allocation, and buffer caching
- MLX's JIT kernel compilation supports dtype specialization out of the box
- Avoids re-inventing Metal boilerplate (residency sets, command buffers, pipeline states)
- Can still write raw `.metal` kernels and register them as custom operations

**Risks**:
- MLX overhead for non-neural-network workloads (GBDT patterns differ from matmul-heavy NN workloads)
- MLX's lazy evaluation model may conflict with CatBoost's imperative training loop
- Dependency on MLX's memory allocator behavior for large histogram buffers

---

## ADR-002: Parallel Backend Architecture (Not Replacement)

**Date**: 2026-03-29
**Status**: Accepted

**Context**: Should we modify CatBoost's existing CUDA backend or create a new parallel backend?

**Decision**: Create a new `catboost/mlx/` directory as a parallel backend, mirroring the structure of `catboost/cuda/`.

**Rationale**:
- Preserves all existing CPU and CUDA functionality
- Allows incremental development — can fall back to CPU for unimplemented ops
- CatBoost already has the `IModelTrainer` interface for pluggable backends
- Clean separation of concerns

**Risks**:
- Code duplication between CUDA and MLX backends
- Must keep MLX backend in sync with CatBoost algorithm changes

---

## ADR-003: PyTorch MPS as Secondary Reference

**Date**: 2026-03-29
**Status**: Accepted

**Context**: PyTorch has a working MPS (Metal Performance Shaders) backend for Apple Silicon. Should we study it?

**Decision**: Use PyTorch's MPS backend (`../pytorch/aten/src/ATen/mps/`) as a secondary reference for Metal integration patterns, but MLX is the primary framework.

**Rationale**:
- PyTorch MPS shows proven patterns for Metal kernel dispatch, memory management, and CPU-GPU synchronization on Apple Silicon
- Different use case (tensor ops vs tree building) but similar Metal API usage
- MLX is more aligned with our needs (simpler, Apple-native, purpose-built for Apple Silicon)

**Risks**:
- PyTorch MPS patterns may not apply well to GBDT workloads
- Time spent studying PyTorch could be better spent on implementation

---

## ADR-004: Histogram-First Implementation Strategy

**Date**: 2026-03-29
**Status**: Accepted

**Context**: CatBoost's GPU pipeline has many components. What should we implement first?

**Decision**: Start with histogram computation kernels, then scoring, then leaf estimation, then the full training loop.

**Rationale**:
- Histogram computation is the hottest loop in GBDT training (>60% of GPU time in CUDA backend)
- It has the clearest parallelization pattern (parallel over features, bins, and documents)
- Can be validated independently against CPU histograms
- Once histograms work, scoring and leaf estimation are simpler kernels
- Training loop orchestration comes last as it ties everything together

**Risks**:
- May discover architectural issues late when integrating the training loop
- Histogram kernel alone doesn't prove the full pipeline works

---

## ADR-005: Float32 as Primary Compute Precision

**Date**: 2026-03-29
**Status**: Accepted

**Context**: Metal supports float16, bfloat16, and float32. CatBoost's CUDA backend uses mixed precision. What precision for MLX?

**Decision**: Use float32 for all accumulation and gradient computation. Use quantized integer types for feature storage (matching CatBoost's existing quantization).

**Rationale**:
- GBDT is sensitive to numerical precision in gradient accumulation (unlike NN inference)
- Apple Silicon's GPU has excellent float32 throughput (no significant penalty vs float16 for compute-bound ops)
- CatBoost's feature quantization (1-byte, half-byte bins) already compresses feature storage
- Can explore float16 for specific operations later after correctness is proven

**Risks**:
- May miss performance gains from float16 in memory-bandwidth-bound kernels
- Higher memory usage for intermediate buffers compared to mixed precision

**Related**: DEC-003 (`.claude/state/DECISIONS.md`) — float32 accumulation constrains `ComputePartitionLayout` to datasets under 16,777,216 rows. Switching to int32 bucket counts (the DEC-003 future enhancement) would remove that ceiling without changing this decision.

---

## ADR-006: All 3 Grow Policies in Library Path (GPU)

**Date**: 2026-04-12
**Status**: Accepted

**Context**: CatBoost's CUDA GPU backend only supports SymmetricTree. Depthwise and Lossguide are CPU-only grow policies upstream. For the MLX library path (`train_lib/train.cpp`), we must decide whether to match CUDA's scope (SymmetricTree only) or support all 3 grow policies on GPU.

**Decision**: Support all 3 grow policies (SymmetricTree, Depthwise, Lossguide) in the MLX library path. This requires implementing `TNonSymmetricTreeModelBuilder` for model export of non-oblivious trees from the GPU backend.

**Rationale**:
- The csv_train standalone path already supports all 3 — this is proven, working code
- All 3 grow policies on GPU is a feature improvement over CUDA, not a conflict with CatBoost's design (the CPU backend already supports all 3)
- Stronger upstream pitch: "MLX does everything CUDA does, plus Depthwise and Lossguide on GPU"
- `TFullModel` and `TNonSymmetricTreeModelBuilder` already exist in CatBoost for CPU-trained non-oblivious trees — no model format changes needed

**Risks**:
- `TNonSymmetricTreeModelBuilder` usage from a GPU backend is unprecedented — may hit undocumented API edge cases
- Larger review surface for upstream PR
- If CatBoost maintainers prefer GPU backends to match CUDA's scope, we may need to gate behind a flag

---

## DEC-028: RandomStrength noise must scale by gradient RMS, not totalHessian/numPartitions

**Date**: 2026-04-22
**Status**: Implemented (S26-D0-6)

**Context**: MLX's `FindBestSplit` (csv_train.cpp) computed the per-split-candidate noise scale as
`randomStrength × totalWeight / (numPartitions × K)`. For RMSE on N=10,000, `totalWeight = N` (the
hessian sum), producing `noiseScale = 10,000`. The true gain at the root split was ~1,602, giving
SNR = 0.16 — noise completely dominated split selection. Empirical evidence (D0-5, branch
`mlx/sprint-26-python-parity`): with `random_strength=1.0`, MLX pred_std_ratio = 0.7218 vs CPU
0.8930; with `random_strength=0`, both are ≈0.919 — the entire gap is attributable to the noise
formula.

**Decision**: Replace the hessian-based noise formula with the gradient-RMS formula matching CPU
CatBoost. Compute `gradRms = sqrt( sum_{k,i} g_k[i]^2 / N )` in `RunTraining` after all gradient
and sample-weight steps, then pass `gradRms` into `FindBestSplit` as a new parameter. The noise
scale becomes `noiseScale = randomStrength × gradRms`.

**Rationale**: CPU CatBoost's `CalcDerivativesStDevFromZeroPlainBoosting`
(greedy_tensor_search.cpp:92–106) returns `sqrt( sum_{k,i} g_k[i]^2 / N )` — the RMS of the
gradient vector, which shrinks as residuals shrink over boosting iterations. This keeps the
noise-to-signal ratio approximately constant across iterations. The old formula (hessian sum = N for
RMSE) is dimensionally wrong: it scales with dataset size rather than gradient magnitude, producing
noise ~16,895× larger than CPU at N=10,000.

**Risks**:
- Lossguide and Depthwise grow policies use `FindBestSplitPerPartition`, which has no noise path.
  RandomStrength has no effect on those policies in the current implementation (this was already
  true before this fix — it is not a regression). If per-partition noise is desired for
  Depthwise/Lossguide in a future sprint, a separate parameter-threading pass is needed.
- `gradRms` is computed via a CPU readback loop over `dimGrads`. For large N this is a minor
  per-iteration cost (profiling target if ever hot).

**Impact**:
- Python-path SymmetricTree RMSE expected to drop from ~0.34 to ~0.20 (parity with CPU ~0.20).
- No impact on bench_boosting ULP=0 record (bench_boosting does not exercise `FindBestSplit`).
- No impact on DEC-008 through DEC-027 code paths.

**S26-FU-2 extension (2026-04-22)**: The same formula and `gradRms` threading were extended to
`FindBestSplitPerPartition` (Depthwise and Lossguide paths) in Sprint 26 Follow-Up 2. The CPU
source audit (T1 triage, `docs/sprint26/fu2/d0-triage.md`) confirmed that CPU uses the identical
global scalar `scoreStDev` — computed once per tree before the depth/partition loop — for all
three grow policies. No new design content beyond mirroring DEC-028 in the non-oblivious path.
Gate artifacts at `docs/sprint26/fu2/`; commits `478e8d5c9d` (C++) and `715b15b613` (tests).

---

## DEC-029: Non-oblivious tree SplitProps never populated → empty model JSON splits

**Date**: 2026-04-22
**Status**: Implemented (S26-D0-8b)

**Context**: Depthwise and Lossguide grow policies showed 560-598% RMSE delta vs CPU
after DEC-028 fixed SymmetricTree. Post-DEC-028 localize.py: Depthwise MLX RMSE 1.2888
vs CPU 0.1950; Lossguide MLX RMSE 1.3754 vs CPU 0.1970. Cause was not the RandomStrength
formula (DEC-028) — Depthwise/Lossguide have no noise path.

**Decision**: Populate `TTreeRecord.SplitProps` and new `TTreeRecord.SplitBfsNodeIds`
in both the Depthwise and Lossguide tree-build paths (previously only SymmetricTree
populated `SplitProps`). Update `WriteModelJSON` to emit `grow_policy` and
`bfs_node_index` per split for non-oblivious trees. Update `_predict_utils.py` to
dispatch `compute_leaf_indices` on `grow_policy` and perform correct BFS traversal.

**Rationale**: The training-time `cursor` was correct: leaf values were indexed by the
bit-packed `partitions` array (bit d = direction at depth d). The bug was entirely in
the model JSON serialization and Python prediction path:
1. `SplitProps` was only populated in the SymmetricTree `else` branch; Depthwise/Lossguide
   `if` branches did not push to `splitProps`.
2. `WriteModelJSON` serialized `splits` from `SplitProps.size()`, which was 0 for
   non-oblivious trees → `"splits": []`.
3. `compute_leaf_indices` iterated over the empty splits list → returned all-zeros
   → all docs assigned to leaf 0 → constant prediction = `leaf_values[0]`.

For Depthwise, the BFS node index for partition `p` at depth `d` is computed
from the bit-pattern of `p`: traverse bits 0..d-1 of `p`, at each level go left
(`2n+1`) or right (`2n+2`). This maps partition → BFS node correctly and is
non-trivial (p=1 at depth 2 → BFS node 5, not node 1). Emitting `bfs_node_index`
explicitly avoids having the Python side re-derive this mapping.

For the Depthwise Python predict, the bit-packed partition value is reconstructed
by `_bfs_traverse_bitpacked`: at BFS node `n` of depth `d = floor(log2(n+1))`,
a right-turn sets bit `d` of the result. This exactly mirrors the C++ partition
update: `bits = updateBits << depth; partitions |= bits`.

**Risks**:
- `ComputeLeafIndicesDepthwise` (C++ validation path) still returns `nodeIdx - numNodes`
  (BFS leaf order), which differs from the bit-packed partition order for depth >= 2.
  Validation RMSE tracking during Depthwise training may be wrong, but this does not
  affect (a) training correctness, (b) final model predictions via Python. Follow-up
  sprint should fix the C++ validation path for completeness.
- The `bfs_node_index` field is new in the model JSON schema. Old models without this
  field (SymmetricTree models or pre-DEC-029 non-oblivious models) still work correctly:
  `compute_leaf_indices` falls back to oblivious for SymmetricTree, and `_build_bfs_node_map`
  returns an empty map for trees without `bfs_node_index` entries (effectively all-leaf-0,
  which matches the pre-fix behavior for old models that were already broken anyway).

**Impact**:
- Depthwise and Lossguide Python predictions expected to go from 560-598% RMSE delta
  to ≤ 5% delta (S26 G2/G3 gate).
- SymmetricTree predictions unchanged (DEC-028 fix preserved).
- No impact on training speed, histogram correctness, or leaf value computation.

---

## DEC-041: Static vs Dynamic Feature Quantization in csv_train.cpp (INVALIDATED)

**Date**: 2026-04-24
**Status**: INVALIDATED 2026-04-24 — premise falsified by post-close probes
**Closes**: (none — does not close DEC-036; DEC-036 reverts to OPEN)

**Original premise (FALSE)**: that CatBoost CPU accumulates feature borders
dynamically while csv_train.cpp builds a static 127-border grid, and that this
difference explains the 52.6% ST+Cosine drift.

**Falsification**:

- **PROBE-A** (`c770ab6630`, `docs/sprint33/probe-a-borders/verdict.md`).
  CatBoost `Pool.quantize` produces **128 borders for every one of the 20
  features** (total 2560), matching csv_train.cpp's static grid 1:1. The
  95/71/0 numbers cited in the L4 verdict are *stored-in-CBM* borders —
  CatBoost's serialization compresses to thresholds the trained trees actually
  reference. Available borders ≠ stored borders. There is no "dynamic border
  accumulation" mechanism in CatBoost.

- **PROBE-B** (`600238f39f`, `docs/sprint33/probe-b-python/verdict.md`).
  The nanobind Python path traces `core.py:1090 _fit_nanobind →
  _nb_core.train → train_api.cpp:14 #include csv_train.cpp →
  train_api.cpp:268 QuantizeFeatures → csv_train.cpp:1177`. The Python path
  calls the **same** QuantizeFeatures as the CLI harness; no `Pool::Quantize`
  is invoked anywhere. Measured Python-path drift at the L4 anchor: **52.64%**
  — matches csv_train to four sig figs. The "production path uses CatBoost
  Pool" rationale (Option 3) is structurally invalid.

**Implications**:
- DEC-036 (ST+Cosine 52.6% drift) reverts to **OPEN** — mechanism unidentified.
- DEC-040 (S33 L0-L4 SCAFFOLD) reverts to **OPEN** — sprint did not close.
- The L0→L3 narrowing to S2 (split selection at iter≥2) **survives** —
  gradients are bit-identical at iter=2 start (max_diff vs CPU 1.5e-8); the
  divergence is downstream in the histogram/scoring path. CPU bin=3 vs MLX
  bin=64 at iter=2 depth=0 with bit-identical iter=1 state is the surviving
  observation.

**Status**: This DEC is preserved as historical record. **Do not act on its
recommendations.** Any successor DEC opened for the genuine DEC-036 mechanism
will be numbered DEC-042 or later.

---

## DEC-042: Degenerate-child skip in FindBestSplit per-partition update — RESOLVED

**Date**: 2026-04-25
**Status**: FULLY CLOSED 2026-04-25
**Closes**: DEC-036 (ST+Cosine structural divergence)
**Commits**: `10c72b4e96` (Cosine fix), `e98c6725cd` (L2 fix), `dd778b0f7d` (validation),
             `e1d72d64e8` (ST guard removed), `d599e5b033` (LG guard removed)
**Branch**: `mlx/sprint-33-iter2-scaffold`
**Gate report**: `docs/sprint33/commit2-gates/REPORT.md`

**Context**:

After five sprints (S30–S33) exhausting the precision-fix class (cosDen accumulator,
gain cast, fp64 widening, histogram N-scaling), PROBE-E confirmed the partition-state
class: the 52.6% ST+Cosine iter=50 drift was structural. The mechanism was located at
`catboost/mlx/tests/csv_train.cpp:1980` in `FindBestSplit`'s per-partition loop.

**Mechanism**:

MLX's per-partition update:
```cpp
if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;
```
skips the entire partition contribution — both `cosNum` and `cosDen` go to zero — when
either child is empty (degenerate split). CPU's reference path
(`catboost/libs/helpers/short_vector_ops.h:155+`, `UpdateScoreBinKernelPlain`) uses a
per-side mask: the empty side contributes zero, but the non-empty side's
`sumX² / (w + λ)` is always added.

For any candidate feature whose tree ancestry creates degenerate children in a subset
of partitions, MLX under-scores the candidate by the omitted partitions' contribution.
At iter=2 depth=2 on the anchor, CPU's signal pick (feat=0, bin=21) had 2 of 4
partitions degenerate (wL=0) after the depth=0 split on feat=0. MLX scored feat=0 at
81.83; CPU scored it at 108.32 (Δ=26.49 gain units). Noise features had no degenerate
partitions; their gains were unchanged at ~101.95. The signal/noise ordering was
inverted at every iter≥2 depth≥2 decision.

Skip rate grows with depth: 0% / 2.5% / 5.0% / 7.6% / 10.6% / 14.6% at depths
0–5. The compounding divergence across 50 iterations matches the 52.6% observation.

**Decision**:

Replace the whole-partition `continue` with a per-side mask at both the Cosine and L2
branches of `FindBestSplit`:
```cpp
const bool wL_pos = (weightLeft  > 1e-15f);
const bool wR_pos = (weightRight > 1e-15f);
if (!wL_pos && !wR_pos) continue;  // truly degenerate on both sides — true no-op
// Compute contributions with each empty side contributing zero
```
Two separate atomic commits per DEC-012: Commit 1 (Cosine path), Commit 1.5 (L2 path).

**Rationale**: Matches CPU's `UpdateScoreBinKernelPlain` reference exactly. The L2 path
receives the same fix for structural correctness even though no L2 regression was
measured, because the shared code pattern has the same latent defect.

**Outcome**:

| Metric | Pre-fix | Post-fix |
|--------|---------|---------|
| ST+Cosine iter=1 drift | ~0% (depth=0 trivial) | 0.0001% |
| ST+Cosine iter=50 drift | 52.6% | 0.027% (1941× improvement) |
| L2 18-config parity | 18/18 [0.9991, 1.0008] | 18/18 [0.9991, 1.0008] |
| LG+Cosine iter=50 drift | unmeasured | 0.382% |
| DW+Cosine 5-seed | 5/5 PASS | 5/5 PASS |
| Kernel md5 | `9edaef45...` | `9edaef45...` (unchanged) |

S28-ST-GUARD and S28-LG-GUARD removed. DEC-032 fully closed: all three grow policies
(SymmetricTree, Depthwise, Lossguide) now support Cosine.

---

## Decision Template

```
## ADR-NNN: Title

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Superseded by ADR-XXX | Deprecated

**Context**: Why are we making this decision?

**Decision**: What did we decide?

**Rationale**: Why this option over alternatives?

**Risks**: What could go wrong?
```
