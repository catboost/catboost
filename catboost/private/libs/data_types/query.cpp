#include "query.h"

#include <catboost/libs/helpers/exception.h>

TVector<TGroupBounds> GroupSamples(TConstArrayRef<TGroupId> qid) {
    TVector<TGroupBounds> qdata;
    TVector<ui64> seenQids;

    ui32 i = 0;
    while (i < qid.size()) {
        const auto current = qid[i];
        TGroupBounds group;
        group.Begin = i++;
        while (i < qid.size() && qid[i] == current) {
            ++i;
        }
        group.End = i;
        qdata.push_back(group);
        seenQids.push_back(current);
    }
    Sort(seenQids);
    CB_ENSURE(
        seenQids.end() == std::adjacent_find(seenQids.begin(), seenQids.end()),
        "Error: queryIds should be grouped"
    );
    return qdata;
}
