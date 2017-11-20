#include <catboost/libs/helpers/exception.h>
#include "ctr_data.h"


void TCtrData::Save(IOutputStream* s) const {
    TCtrDataStreamWriter ctrStreamSerializer(s, LearnCtrs.size());
    for (const auto& iter : LearnCtrs) {
        CB_ENSURE(iter.first == iter.second.ModelCtrBase);
        ctrStreamSerializer.SaveOneCtr(iter.second);
    }
}

void TCtrData::Load(IInputStream* s) {
    const size_t cnt = ::LoadSize(s);
    LearnCtrs.reserve(cnt);

    for (size_t i = 0; i != cnt; ++i) {
        TCtrValueTable table;
        table.Load(s);
        TModelCtrBase ctrBase = table.ModelCtrBase;
        LearnCtrs[ctrBase] = std::move(table);
    }
}
