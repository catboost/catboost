// hist_helpers.metal — Shared utilities for CatBoost-MLX histogram kernels.
// Metal Shading Language 3.0+ (Apple Silicon M1+)

#include <metal_stdlib>
using namespace metal;

// Feature descriptor passed from host.
// Must match TCFeature layout in gpu_structures.h.
struct FeatureInBlock {
    uint64_t compressedIndexOffset;
    int folds;
    int foldOffsetInGroup;
    int groupOffset;
    int groupSize;
};

// Partition descriptor: defines a contiguous range of documents in a leaf.
struct DataPartition {
    uint offset;
    uint size;
};

// Best split result written by scoring kernel.
struct BestSplitProperties {
    uint featureId;
    uint binId;
    float score;
    float gain;
};

// SIMD group (warp) size on Apple Silicon is always 32.
constant constexpr uint SIMD_SIZE = 32;
