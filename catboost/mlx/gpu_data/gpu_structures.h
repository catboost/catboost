#pragma once

// GPU-side POD structures for the MLX Metal backend.
// These mirror catboost/cuda/gpu_data/gpu_structures.h so Metal kernels
// operate on identical data layouts, but without CUDA dependencies.

#include <util/system/types.h>
#include <util/digest/multi.h>

#include <limits>

// How to extract a quantized feature value from the compressed index.
// Corresponds to one logical feature packed inside a ui32 word.
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

// Identifies a specific (feature, bin) pair for split evaluation.
struct TCBinFeature {
    ui32 FeatureId = static_cast<ui32>(-1);
    ui32 BinId = static_cast<ui32>(-1);
    bool SkipInScoreCount = false;

    bool operator<(const TCBinFeature& other) const {
        return FeatureId < other.FeatureId || (FeatureId == other.FeatureId && BinId < other.BinId);
    }
};

// Result of finding the best split across all features and bins.
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

struct TBestSplitPropertiesWithIndex : public TBestSplitProperties {
    ui32 Index = 0;

    bool operator<(const TBestSplitPropertiesWithIndex& other) const {
        return TBestSplitProperties::operator<(other);
    }
};

// Per-partition (leaf) accumulated statistics for gradient/weight.
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

// Describes how a feature group maps into the histogram output buffer.
// Used by kernels to know where to write histogram bins.
struct TFeatureInBlock {
    ui64 CompressedIndexOffset = 0;
    int Folds = 0;
    int FoldOffsetInGroup = 0;
    int GroupOffset = 0;    // offset into global histogram buffer
    int GroupSize = 0;      // total bin-features in this group across devices
};

// Split decision at a tree node (for oblivious trees, one per depth level).
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
