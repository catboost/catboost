#pragma once

struct TPair {
    int WinnerId;
    int LoserId;

    TPair(int winnerId, int loserId)
        : WinnerId(winnerId)
        , LoserId(loserId) {
    }
};

struct TCompetitor {
    int Id;
    float Weight;

    TCompetitor(int id, float weight)
        : Id(id)
        , Weight(weight) { }
};
