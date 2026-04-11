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
