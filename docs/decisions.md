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
