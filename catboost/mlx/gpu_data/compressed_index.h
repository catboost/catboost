#pragma once

// Compressed feature index on Metal GPU via MLX arrays.
//
// CatBoost quantizes continuous features into bins (0-255 for one-byte features).
// Multiple features are packed into ui32 words:
//   - One-byte features: 4 features per ui32 (bits 24-31, 16-23, 8-15, 0-7)
//   - Half-byte features: 8 features per ui32 (4 bits each)
//   - Binary features: 32 features per ui32 (1 bit each)
//
// Layout: compressedIndex[docIdx * numUi32PerDoc + wordIdx]
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

            // Transfer packed feature data to GPU
            CompressedData_ = mx::array(
                reinterpret_cast<const int32_t*>(compressedData),
                {static_cast<int>(numDocs), static_cast<int>(numUi32PerDoc)},
                mx::uint32
            );
            // EvalAtBoundary: materialise compressed feature data on GPU at load time
            // so it is ready for any downstream Metal kernel without lazy overhead.
            TMLXDevice::EvalAtBoundary(CompressedData_);
        }

        const mx::array& GetCompressedData() const {
            return CompressedData_;
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
        mx::array CompressedData_;          // [numDocs, numUi32PerDoc] uint32
        TVector<TCFeature> Features_;       // feature metadata (CPU-side, passed to kernels)
        TVector<ui32> ExternalFeatureIndices_; // GPU local idx → external feature idx
        ui32 NumDocs_ = 0;
        ui32 NumUi32PerDoc_ = 0;
    };

}  // namespace NCatboostMlx
