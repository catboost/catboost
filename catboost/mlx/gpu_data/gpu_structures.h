#pragma once

// GPU-side POD structures for the MLX Metal backend.
// These mirror catboost/cuda/gpu_data/gpu_structures.h so Metal kernels
// operate on identical data layouts, but without CUDA dependencies.

#include <util/system/types.h>
#include <util/digest/multi.h>

#include <limits>

/// Descriptor for one quantized feature packed into the compressed index.
/// Each TCFeature describes how to extract one logical feature's bin value
/// from a ui32 word: apply a right-shift of Shift bits, then mask with Mask.
/// FirstFoldIndex is the starting offset in the global histogram buffer for
/// this feature's bins; Folds is the number of quantization bins.
struct TCFeature {
    ui64 Offset = static_cast<ui64>(-1);    // byte offset into compressed index buffer
    ui32 Mask = 0;                           // bitmask to extract this feature's bits
    ui32 Shift = 0;                          // right-shift count after masking
    ui32 FirstFoldIndex = 0;                 // first histogram bin index for this feature
    ui32 Folds = 0;                          // number of bins (folds) for this feature
    bool OneHotFeature = false;
    bool SkipFirstBinInScoreCount = false;

    TCFeature() = default;

    TCFeature(ui64 offset, ui32 mask, ui32 shift, ui32 firstFoldIndex, ui32 folds,
              bool oneHotFeature, bool skipFirstBinInScoreCount)
        : Offset(offset)
        , Mask(mask)
        , Shift(shift)
        , FirstFoldIndex(firstFoldIndex)
        , Folds(folds)
        , OneHotFeature(oneHotFeature)
        , SkipFirstBinInScoreCount(skipFirstBinInScoreCount)
    {
    }

    bool IsEmpty() const {
        return Offset == static_cast<ui64>(-1);
    }
};

/// Identifies a specific (feature, bin) pair as a split candidate.
struct TCBinFeature {
    ui32 FeatureId = static_cast<ui32>(-1);
    ui32 BinId = static_cast<ui32>(-1);
    bool SkipInScoreCount = false;

    bool operator<(const TCBinFeature& other) const {
        return FeatureId < other.FeatureId || (FeatureId == other.FeatureId && BinId < other.BinId);
    }
};

/// Result of the best-split search: the winning (feature, bin) pair and its gain.
/// Score < 0 indicates a valid split; Score == infinity means no split was found.
struct TBestSplitProperties {
    ui32 FeatureId = static_cast<ui32>(-1);
    ui32 BinId = 0;
    float Score = std::numeric_limits<float>::infinity();
    float Gain = std::numeric_limits<float>::infinity();

    TBestSplitProperties() = default;

    TBestSplitProperties(ui32 featureId, ui32 binId, float score, float gain)
        : FeatureId(featureId)
        , BinId(binId)
        , Score(score)
        , Gain(gain)
    {
    }

    bool operator<(const TBestSplitProperties& other) const {
        if (Gain < other.Gain) {
            return true;
        } else if (Gain == other.Gain) {
            if (FeatureId < other.FeatureId) {
                return true;
            } else if (FeatureId == other.FeatureId) {
                return BinId < other.BinId;
            }
            return false;
        }
        return false;
    }

    bool Defined() const {
        return FeatureId != static_cast<ui32>(-1);
    }

    void Reset() {
        (*this) = TBestSplitProperties();
    }
};

/// TBestSplitProperties augmented with a block index for GPU block-level argmax reductions.
struct TBestSplitPropertiesWithIndex : public TBestSplitProperties {
    ui32 Index = 0;

    bool operator<(const TBestSplitPropertiesWithIndex& other) const {
        return TBestSplitProperties::operator<(other);
    }
};

/// Accumulated gradient/weight statistics for one leaf partition.
/// Weight is the sum of hessians (or sample weights); Sum is the gradient sum; Count is the document count.
struct TPartitionStatistics {
    double Weight;
    double Sum;
    double Count;

    TPartitionStatistics(double weight = 0, double sum = 0, double count = 0)
        : Weight(weight)
        , Sum(sum)
        , Count(count)
    {
    }

    TPartitionStatistics& operator+=(const TPartitionStatistics& other) {
        Weight += other.Weight;
        Sum += other.Sum;
        Count += other.Count;
        return *this;
    }

    bool operator==(const TPartitionStatistics& other) const {
        return Weight == other.Weight && Sum == other.Sum && Count == other.Count;
    }
};

/// Maps one feature group into the histogram output buffer.
/// Passed to the histogram Metal kernel to locate where each group's bins are written.
struct TFeatureInBlock {
    ui64 CompressedIndexOffset = 0;
    int Folds = 0;
    int FoldOffsetInGroup = 0;
    int GroupOffset = 0;    // offset into global histogram buffer
    int GroupSize = 0;      // total bin-features in this group across devices
};

/// Split decision at one tree node, stored in CatBoost's native packed format.
/// For oblivious trees, one TTreeNode per depth level; for depthwise trees, one per internal node.
struct TTreeNode {
    ui16 FeatureId = 0;
    ui16 Bin = 0;
    ui16 LeftSubtree = 0;
    ui16 RightSubtree = 0;

    ui64 GetHash() const {
        return MultiHash(FeatureId, Bin, LeftSubtree, RightSubtree);
    }

    bool operator==(const TTreeNode& rhs) const {
        return std::tie(FeatureId, Bin, LeftSubtree, RightSubtree) ==
               std::tie(rhs.FeatureId, rhs.Bin, rhs.LeftSubtree, rhs.RightSubtree);
    }

    bool operator!=(const TTreeNode& rhs) const {
        return !(*this == rhs);
    }
};
