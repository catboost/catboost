#pragma once

// Compressed feature index on Metal GPU via MLX arrays.
//
// CatBoost quantizes continuous features into bins (0-255 for one-byte features).
// Multiple features are packed into ui32 words:
//   - One-byte features: 4 features per ui32 (bits 24-31, 16-23, 8-15, 0-7)
//   - Half-byte features: 8 features per ui32 (4 bits each)
//   - Binary features: 32 features per ui32 (1 bit each)
//
// Row-major layout:   CompressedData_[docIdx * numUi32PerDoc + wordIdx]
// Col-major layout:   CompressedDataTransposed_[wordIdx * numDocs + docIdx]
//
// DEC-015 (Sprint 19): The L1a histogram kernel reads one compressedIndex word
// per doc per feature-group. Under row-major layout each 32-doc batch spans up
// to `lineSize` (≈25 at gate) cache lines.  The transposed col-major view
// collapses that to 1 cache line per 32-doc batch, eliminating the 12.78 ms
// gather-latency bottleneck identified in S19-01b.
//
// Feature extraction: (compressedIndex[offset] >> shift) & mask

#include <catboost/mlx/gpu_data/gpu_structures.h>
#include <catboost/mlx/gpu_data/mlx_device.h>

#include <util/generic/vector.h>

namespace NCatboostMlx {

    // Holds quantized feature data on the GPU as mx::array buffers
    class TMLXCompressedIndex {
    public:
        TMLXCompressedIndex() = default;

        // Build from raw CPU data. `compressedData` is the packed ui32 array
        // with shape [numDocs, numUi32PerDoc].
        void Build(const ui32* compressedData,
                   ui32 numDocs,
                   ui32 numUi32PerDoc,
                   const TVector<TCFeature>& features,
                   const TVector<ui32>& externalFeatureIndices = {}) {
            NumDocs_ = numDocs;
            NumUi32PerDoc_ = numUi32PerDoc;
            Features_ = features;
            ExternalFeatureIndices_ = externalFeatureIndices;

            // Transfer packed feature data to GPU — row-major [numDocs, numUi32PerDoc].
            // EvalAtBoundary: materialise so it is ready for downstream Metal kernels
            // and for building the transposed view without lazy graph recompute.
            CompressedData_ = mx::array(
                reinterpret_cast<const int32_t*>(compressedData),
                {static_cast<int>(numDocs), static_cast<int>(numUi32PerDoc)},
                mx::uint32
            );
            TMLXDevice::EvalAtBoundary(CompressedData_);

            // DEC-015: col-major transposed view for the L1a histogram kernel.
            //
            // mx::transpose({1,0}) is a metadata-only op — it returns a col-contiguous
            // strided view, NOT a new row-contiguous buffer.  To avoid triggering a
            // GPU copy on every kernel dispatch (via ensure_row_contiguous=true in
            // DispatchHistogramGroup), we materialise a fresh row-contiguous copy here
            // once at Build() time:
            //
            //   mx::copy(transposed)  → allocates a new row-contiguous [numUi32PerDoc, numDocs] buffer.
            //   mx::reshape(..., {-1}) → zero-copy 1D view of that row-contiguous buffer.
            //   EvalAtBoundary          → drives the Metal copy kernel, leaving a
            //                             single contiguous GPU buffer for all downstream reads.
            //
            // Memory cost: numDocs * numUi32PerDoc * 4 B ≈ 5 MB at gate (50k docs × 25 ui32).
            // That is 2× the row-major buffer — trivial on M3 Ultra unified memory.
            auto transposed = mx::transpose(CompressedData_, {1, 0});  // [numUi32PerDoc, numDocs]
            // mx::copy forces a row-contiguous allocation, making the transposed layout
            // contiguous in memory (col index changes slowest → sequential docIdx reads
            // within one feature column = 1 cache line per 32-doc batch).
            CompressedDataTransposed_ = mx::reshape(mx::copy(transposed), {-1});  // [numUi32PerDoc * numDocs]
            TMLXDevice::EvalAtBoundary(CompressedDataTransposed_);
        }

        const mx::array& GetCompressedData() const {
            return CompressedData_;
        }

        // DEC-015: col-major view — featureColumnIdx * totalNumDocs + docIdx.
        // Shape: [numUi32PerDoc * numDocs], row-contiguous (materialised at Build time).
        // Only the L1a histogram kernel reads from this view; all other consumers
        // (leaves.metal, tree_applier.cpp, etc.) continue to use GetCompressedData().
        const mx::array& GetCompressedDataTransposed() const {
            return CompressedDataTransposed_;
        }

        const TVector<TCFeature>& GetFeatures() const {
            return Features_;
        }

        // Maps GPU local feature index → CatBoost external feature index.
        // Needed for model export (to look up quantization borders).
        const TVector<ui32>& GetExternalFeatureIndices() const {
            return ExternalFeatureIndices_;
        }

        ui32 GetNumDocs() const { return NumDocs_; }
        ui32 GetNumUi32PerDoc() const { return NumUi32PerDoc_; }

    private:
        mx::array CompressedData_;            // [numDocs, numUi32PerDoc] uint32, row-major
        mx::array CompressedDataTransposed_;  // [numUi32PerDoc * numDocs] uint32, col-major (DEC-015)
        TVector<TCFeature> Features_;         // feature metadata (CPU-side, passed to kernels)
        TVector<ui32> ExternalFeatureIndices_; // GPU local idx → external feature idx
        ui32 NumDocs_ = 0;
        ui32 NumUi32PerDoc_ = 0;
    };

}  // namespace NCatboostMlx
