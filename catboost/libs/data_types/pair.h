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
    SAVELOAD(WinnerId, LoserId, Weight);
};

struct TCompetitor {
    int Id;
    float Weight;

    TCompetitor() = default;
    TCompetitor(int id, float weight)
        : Id(id)
        , Weight(weight) { }
    SAVELOAD(Id, Weight);
};
