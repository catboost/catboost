#include "calc_class_weights.h"

#include <catboost/libs/logging/logging.h>

#include <util/generic/algorithm.h>
#include <util/system/yassert.h>

static std::function<float(ui64, ui64)> GetWeightFunction(EAutoClassWeightsType autoClassWeightsType) {
    switch (autoClassWeightsType) {
        case EAutoClassWeightsType::Balanced:
            return [](ui64 maxClassSize, ui64 classSize) -> float {
                return classSize > 0 ? static_cast<float>(maxClassSize) / classSize : 1.f;
            };
        case EAutoClassWeightsType::SqrtBalanced:
            return [](ui64 maxClassSize, ui64 classSize) -> float {
                return classSize > 0 ? sqrt(static_cast<float>(maxClassSize) / classSize) : 1.f;
            };
        case EAutoClassWeightsType::None:
            Y_VERIFY(false);
    }
}

static TVector<ui64> CalculateNumberOfClassItems(
    TConstArrayRef<float> targetClasses,
    ui32 classCount,
    NPar::TLocalExecutor* localExecutor
) {
    const int numThreads = std::max(localExecutor->GetThreadCount(), 1);
    TVector<TVector<ui64>> numClassItemsPerBatch(numThreads, TVector<ui64>(classCount, 0ll));

    NPar::TLocalExecutor::TExecRangeParams params{0, targetClasses.ysize()};
    params.SetBlockCount(numThreads);

    localExecutor->ExecRange(
        [&](int i) {
            int blockId = i / params.GetBlockSize();
            ++numClassItemsPerBatch[blockId][static_cast<size_t>(targetClasses[i])];
        }, params, NPar::TLocalExecutor::WAIT_COMPLETE);

    TVector<ui64> numClassItems(classCount, 0ll);

    for (const auto& batchNumClassItems : numClassItemsPerBatch) {
        for (ui32 i = 0; i < classCount; ++i) {
            numClassItems[i] += batchNumClassItems[i];
        }
    }

    return numClassItems;
}

TVector<float> CalculateClassWeights(
    TConstArrayRef<float> targetClasses,
    ui32 classCount,
    EAutoClassWeightsType autoClassWeightsType,
    NPar::TLocalExecutor* localExecutor
) {
    Y_VERIFY(classCount > 0);
    TVector<ui64> numClassItems = CalculateNumberOfClassItems(targetClasses, classCount, localExecutor);
    Y_VERIFY(numClassItems.size() == classCount);

    CATBOOST_INFO_LOG << "Class weights type: " << autoClassWeightsType << Endl;
    TVector<float> classWeights;
    classWeights.reserve(classCount);

    const ui64 maxClassSize = *MaxElement(numClassItems.begin(), numClassItems.end());
    const auto weightFunction = GetWeightFunction(autoClassWeightsType);
    for (const auto& classSize : numClassItems) {
        classWeights.emplace_back(weightFunction(maxClassSize, classSize));
        CATBOOST_INFO_LOG << "Weight of class " << classWeights.size() - 1 << ": " << classWeights.back() << Endl;
    }

    return classWeights;
}
