#include "ctr_data.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/set.h>
#include <util/stream/mem.h>


void TCtrData::Save(IOutputStream* s) const {
    TCtrDataStreamWriter ctrStreamSerializer(s, LearnCtrs.size());
    TSet<TModelCtrBase> sortedCtrBases;
    for (const auto& iter : LearnCtrs) {
        sortedCtrBases.insert(iter.first);
    }
    Y_ASSERT(sortedCtrBases.size() == LearnCtrs.size());
    for (const auto& ctrBase: sortedCtrBases) {
        const auto& tableRef = LearnCtrs.at(ctrBase);
        CB_ENSURE(ctrBase == tableRef.ModelCtrBase);
        ctrStreamSerializer.SaveOneCtr(tableRef);
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

void TCtrData::LoadNonOwning(TMemoryInput *in) {
    const size_t cnt = ::LoadSize(in);
    LearnCtrs.reserve(cnt);

    for (size_t i = 0; i != cnt; ++i) {
        TCtrValueTable table;
        table.LoadThin(in);
        LearnCtrs[table.ModelCtrBase] = std::move(table);
    }
}
