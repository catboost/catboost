#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/digest/multi.h>
#include <util/stream/output.h>
#include <util/str_stl.h>
#include <util/ysaveload.h>


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
    Y_SAVELOAD_DEFINE(WinnerId, LoserId, Weight);
};


template <>
struct THash<TPair> {
    inline size_t operator()(const TPair& pair) const {
        return MultiHash(pair.WinnerId, pair.LoserId, pair.Weight);
    }
};


void OutputHumanReadable(const TPair& pair, IOutputStream* out);

TString HumanReadableDescription(const TPair& pair);


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

    bool operator==(const TCompetitor& rhs) const {
        return (Id == rhs.Id) && (Weight == rhs.Weight) && (SampleWeight == rhs.SampleWeight);
    }

    SAVELOAD(Id, Weight, SampleWeight);
};

using TFlatPairsInfo = TVector<TPair>;
