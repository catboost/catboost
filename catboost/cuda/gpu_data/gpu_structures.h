#pragma once

#include <util/system/types.h>

#ifndef __NVCC__

#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <util/ysaveload.h>
#endif
//struct to make bin-feature from ui32 feature
// (compressedIndex[Offset] & Mask  should be true
struct TCBinFeature {
    ui32 FeatureId;
    ui32 BinId;
};

struct TCFeature {
    //how to get features
    //ui64 cindex offset
    ui64 Offset = static_cast<ui64>(-1);
    //offset and mask in ui32
    ui32 Mask = 0;
    ui32 Shift = 0;
    //where and how to write histograms
    //local fold idx (index of first fold for grid on device)
    ui32 FirstFoldIndex = 0;
    //fold count
    ui32 Folds = 0;
    //    global index (not feature-id, index in grid only)
    //    ui32 Index;
    bool OneHotFeature = false;

    TCFeature() = default;

    TCFeature(ui64 offset, ui32 mask, ui32 shift, ui32 firstFoldIndex, ui32 folds, bool oneHotFeature)
        : Offset(offset)
        , Mask(mask)
        , Shift(shift)
        , FirstFoldIndex(firstFoldIndex)
        , Folds(folds)
        , OneHotFeature(oneHotFeature)
    {}
};

struct TBestSplitProperties {
    ui32 FeatureId = 0;
    ui32 BinId = 0;
    float Score = 0;

    TBestSplitProperties() = default;

    TBestSplitProperties(ui32 featureId, ui32 binId, float score)
        : FeatureId(featureId)
        , BinId(binId)
        , Score(score)
    {}

    bool operator<(const TBestSplitProperties& other) {
        if (Score < other.Score) {
            return true;
        } else if (Score == other.Score) {
            if (FeatureId < other.FeatureId) {
                return true;
            } else if (FeatureId == other.FeatureId) {
                return BinId < other.BinId;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
};

struct TPartitionStatistics {
    double Weight;
    double Sum;
    double Count;

    TPartitionStatistics(double weight = 0,
                         double sum = 0,
                         double count = 0)
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

    bool operator==(const TPartitionStatistics& other) {
        return Weight == other.Weight &&
               Sum == other.Sum &&
               Count == other.Count;
    }
};

#ifndef __NVCC__
Y_DECLARE_PODTYPE(TCFeature);

namespace NCudaLib {
    namespace NHelpers {
        template <>
        class TEmptyObjectsHelper<TCFeature> {
        public:
            static inline bool IsEmpty(const TCFeature& val) {
                return val.Offset == static_cast<ui64>(-1);
            }
        };
    }
}
#endif
