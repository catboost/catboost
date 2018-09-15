#pragma once

#include <library/binsaver/bin_saver.h>

struct TPair {
    int WinnerId;
    int LoserId;
    float Weight;

    TPair() = default;
    TPair(int winnerId, int loserId, float weight)
        : WinnerId(winnerId)
        , LoserId(loserId)
        , Weight(weight)
    {
    }

    bool operator==(const TPair& other) const {
        return (std::tie(WinnerId, LoserId, Weight) == std::tie(other.WinnerId, other.LoserId, other.Weight));
    }

    SAVELOAD(WinnerId, LoserId, Weight);
};

struct TCompetitor {
    int Id; // index that is relative to group start
    float Weight;
    float SampleWeight;

    TCompetitor() = default;
    TCompetitor(int id, float weight)
        : Id(id)
        , Weight(weight)
        , SampleWeight(weight)
    {
    }

    SAVELOAD(Id, Weight, SampleWeight);
};

using TFlatPairsInfo = TVector<TPair>;
