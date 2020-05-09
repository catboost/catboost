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

TVector<float> CalculateClassWeights(TConstArrayRef<float> targetClasses, ui32 classCount, EAutoClassWeightsType autoClassWeightsType) {
    Y_VERIFY(classCount > 0);
    TVector<ui64> numClassItems(classCount, 0ll);

    for (const auto& target : targetClasses) {
        ++numClassItems[static_cast<size_t>(target)];
    }

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
