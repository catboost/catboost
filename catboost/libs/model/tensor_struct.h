#pragma once

#include "split.h"

#include <util/digest/multi.h>

struct TTensorStructure3 {
    yvector<TModelSplit> SelectedSplits;

    Y_SAVELOAD_DEFINE(SelectedSplits)

    void Add(const TModelSplit& split) {
        SelectedSplits.push_back(split);
    }

    size_t GetHash() const {
        size_t hashValue = 1234501;
        for (const auto& split : SelectedSplits) {
            hashValue = MultiHash(hashValue, split.GetHash());
        }
        return hashValue;
    }

    int GetDepth() const {
        return SelectedSplits.ysize();
    }

    bool operator==(const TTensorStructure3& other) const {
        return SelectedSplits == other.SelectedSplits;
    }
};

yvector<TBinFeature> GetBinFeatures(const TTensorStructure3& tree);
yvector<TOneHotFeature> GetOneHotFeatures(const TTensorStructure3& tree);
yvector<TModelCtrSplit> GetCtrSplits(const TTensorStructure3& tree);
