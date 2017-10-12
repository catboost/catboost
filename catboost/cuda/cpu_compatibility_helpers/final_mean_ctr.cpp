#include "final_mean_ctr.h"
#include <catboost/libs/algo/index_hash_calcer.h>
#include <util/generic/vector.h>

void CalcTargetMeanFinalCtrs(const TModelCtrBase& ctr,
                             const TAllFeatures& data,
                             const ui64 sampleCount,
                             const yvector<float>& target,
                             const yvector<int>& learnPermutation,
                             TCtrValueTable* result) {
    CB_ENSURE(ctr.CtrType == ECtrType::FloatTargetMeanValue);
    yvector<ui64> hashArr;
    CalcHashes(ctr.Projection, data, sampleCount, learnPermutation, &hashArr);

    const auto topSize = Max<ui64>();

    auto leafCount = ReindexHash(
                         sampleCount,
                         topSize,
                         &hashArr,
                         &result->Hash)
                         .first;
    auto ctrMean = result->AllocateBlobAndGetArrayRef<TCtrMeanHistory>(leafCount);
    Y_ASSERT(hashArr.size() == sampleCount);

    for (ui32 z = 0; z < sampleCount; ++z) {
        const ui64 elemId = hashArr[z];
        TCtrMeanHistory& elem = ctrMean[elemId];
        elem.Add(target[z]);
    }
};
