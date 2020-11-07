#pragma once

#include <catboost/private/libs/options/enums.h>
#include <catboost/libs/data/weights.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NCB {
    TVector<float> CalculateClassWeights(
        TConstArrayRef<float> targetClasses,
        const TWeights<float>& itemWeights,
        ui32 classCount,
        EAutoClassWeightsType autoClassWeightsType,
        NPar::ILocalExecutor* localExecutor);
}
