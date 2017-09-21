#include "data_utils.h"

void GroupQueries(const yvector<ui32>& qid, yvector<yvector<ui32>>* qdata) {
    for (ui32 i = 0; i < qid.size(); ++i) {
        ui32 current = qid[i];
        qdata->resize(qdata->size() + 1);
        yvector<ui32>& query = qdata->back();
        while (i < qid.size() && qid[i] == current) {
            query.push_back(i);
            ++i;
        }
        --i;
    }
}
