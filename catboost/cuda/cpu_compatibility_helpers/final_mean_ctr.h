#pragma once

#include <catboost/libs/model/online_ctr.h>
#include <catboost/libs/algo/full_features.h>
#include <catboost/libs/model/model.h>

namespace NCatboostCuda
{
    void CalcTargetMeanFinalCtrs(const TModelCtrBase& ctr,
                                 const TAllFeatures& data,
                                 const ui64 sampleCount,
                                 const yvector<float>& target,
                                 const yvector<int>& learnPermutation,
                                 TCtrValueTable* result);

}
