#pragma once

struct TPair {
    int WinnerId;
    int LoserId;
    float Weight;

    TPair(int winnerId, int loserId, float weight)
        : WinnerId(winnerId)
        , LoserId(loserId)
        , Weight(weight)
    {
    }
};

struct TCompetitor {
    int Id;
    float Weight;

    TCompetitor(int id, float weight)
        : Id(id)
        , Weight(weight) { }
};
