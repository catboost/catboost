#pragma once

#include <library/binsaver/bin_saver.h>

struct TPair {
    ui32 WinnerId;
    ui32 LoserId;
    float Weight;

    TPair() = default;
    TPair(ui32 winnerId, ui32 loserId, float weight)
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
    ui32 Id; // index that is relative to group start
    float Weight;
    float SampleWeight;

    TCompetitor() = default;
    TCompetitor(ui32 id, float weight)
        : Id(id)
        , Weight(weight)
        , SampleWeight(weight)
    {
    }

    SAVELOAD(Id, Weight, SampleWeight);
};

using TFlatPairsInfo = TVector<TPair>;
