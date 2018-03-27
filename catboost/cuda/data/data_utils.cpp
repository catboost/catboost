#include "data_utils.h"
#include <catboost/cuda/cuda_lib/helpers.h>
#include <util/random/shuffle.h>

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
