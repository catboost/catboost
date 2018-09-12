#include "data_utils.h"
#include <catboost/cuda/cuda_lib/helpers.h>

#include <catboost/libs/helpers/exception.h>


void NCatboostCuda::GroupSamples(const TVector<TGroupId>& qid, TVector<TVector<ui32>>* qdata) {
    TSet<TGroupId> knownQids;
    for (ui32 i = 0; i < qid.size(); ++i) {
        auto current = qid[i];
        CB_ENSURE(knownQids.count(current) == 0, "Error: queryIds should be groupped");
        qdata->resize(qdata->size() + 1);
        TVector<ui32>& query = qdata->back();
        while (i < qid.size() && qid[i] == current) {
            query.push_back(i);
            ++i;
        }
        knownQids.insert(current);
        --i;
    }
}

TVector<ui32> NCatboostCuda::ComputeGroupOffsets(const TVector<TVector<ui32>>& queries) {
    TVector<ui32> offsets;
    ui32 cursor = 0;
    for (const auto& query : queries) {
        offsets.push_back(cursor);
        cursor += query.size();
    }
    return offsets;
}

TVector<ui32> NCatboostCuda::ComputeGroupSizes(const TVector<TVector<ui32>>& gdata) {
    TVector<ui32> sizes;
    sizes.resize(gdata.size());
    for (ui32 i = 0; i < gdata.size(); ++i) {
        sizes[i] = gdata[i].size();
    }
    return sizes;
}
