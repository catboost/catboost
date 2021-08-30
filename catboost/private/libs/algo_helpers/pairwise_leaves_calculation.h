#pragma once

#include <catboost/private/libs/data_types/query.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/containers/2d_array/2d_array.h>

#include <util/generic/fwd.h>

namespace NPar {
    class ILocalExecutor;
}


TVector<double> CalculatePairwiseLeafValues(
    const TArray2D<double>& pairwiseWeightSums,
    const TVector<double>& derSums,
    float l2DiagReg,
    float pairwiseBucketWeightPriorReg
);

TArray2D<double> ComputePairwiseWeightSums(
    const TVector<TQueryInfo>& queriesInfo,
    int leafCount,
    int querycount,
    const TVector<TIndexType>& indices,
    NPar::ILocalExecutor* localExecutor
);

