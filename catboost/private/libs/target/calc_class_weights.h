#pragma once

#include <catboost/private/libs/options/enums.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

TVector<float> CalculateClassWeights(
    TConstArrayRef<float> targetClasses,
    ui32 classCount,
    EAutoClassWeightsType autoClassWeightsType,
    NPar::TLocalExecutor* localExecutor);
