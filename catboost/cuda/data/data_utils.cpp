#include "data_utils.h"

void NCatboostCuda::GroupQueries(const TVector<ui32>& qid, TVector<TVector<ui32>>* qdata) {
    for (ui32 i = 0; i < qid.size(); ++i) {
        auto current = qid[i];
        qdata->resize(qdata->size() + 1);
        TVector<ui32>& query = qdata->back();
        while (i < qid.size() && qid[i] == current) {
            query.push_back(i);
            ++i;
        }
        --i;
    }
}
