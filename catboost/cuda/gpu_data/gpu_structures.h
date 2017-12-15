#pragma once

#include <util/system/types.h>
//struct to make bin-feature from ui32 feature
// (compressedIndex[Offset] & Mask  should be true
struct TCBinFeature {
    ui32 FeatureId;
    ui32 BinId;
};

struct TCFeature {
    //ui32 line
    ui32 Offset;
    //offset and mask in ui32
    ui32 Mask;
    ui32 Shift;
    //local fold idx
    ui32 FirstFoldIndex;
    //fold count
    ui32 Folds;
    //index on device
    ui32 LocalIndex;
    //global index (not feature-id, index in grid only)
    ui32 Index;
    bool OneHotFeature;
};

struct TBestSplitProperties {
    ui32 FeatureId;
    ui32 BinId;
    float Score;
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
