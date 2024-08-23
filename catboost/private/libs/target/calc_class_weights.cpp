#include "calc_class_weights.h"

#include <catboost/libs/logging/logging.h>

#include <util/generic/algorithm.h>
#include <util/system/compiler.h>
#include <util/system/yassert.h>

const float MINIMAL_CLASS_WEIGHT = 1e-8;

static std::function<float(ui64, ui64)> GetWeightFunction(EAutoClassWeightsType autoClassWeightsType) {
    switch (autoClassWeightsType) {
        case EAutoClassWeightsType::Balanced:
            return [](float maxSummaryClassWeight, float summaryClassWeight) -> float {
                return summaryClassWeight > MINIMAL_CLASS_WEIGHT ?
                       maxSummaryClassWeight / summaryClassWeight : 1.f;
            };
        case EAutoClassWeightsType::SqrtBalanced:
            return [](float maxSummaryClassWeight, float summaryClassWeight) -> float {
                return summaryClassWeight > MINIMAL_CLASS_WEIGHT ?
                    sqrt(maxSummaryClassWeight / summaryClassWeight) : 1.f;
            };
        case EAutoClassWeightsType::None:
            CB_ENSURE(false, "Unexcepted auto class weights type");
    }
    Y_UNREACHABLE();
}

static TVector<float> CalculateSummaryClassWeight(
    TConstArrayRef<float> targetClasses,
    const NCB::TWeights<float>& itemWeights,
    ui32 classCount,
    NPar::ILocalExecutor* localExecutor
) {
    const int numThreads = localExecutor->GetThreadCount() + 1;
    TVector<TVector<double>> summaryClassWeightsPerBlock(numThreads, TVector<double>(classCount, 0.));

    NPar::ILocalExecutor::TExecRangeParams params{0, targetClasses.ysize()};
    params.SetBlockCount(numThreads);
    localExecutor->ExecRange(
        [&](int i) {
            int blockId = i / params.GetBlockSize();
            summaryClassWeightsPerBlock[blockId][static_cast<size_t>(targetClasses[i])] += itemWeights[i];
        }, params, NPar::TLocalExecutor::WAIT_COMPLETE);

    TVector<float> summaryClassWeights(classCount, 0.);

    for (const auto& blockSummaryClassWeights : summaryClassWeightsPerBlock) {
        for (ui32 i = 0; i < classCount; ++i) {
            summaryClassWeights[i] += blockSummaryClassWeights[i];
        }
    }

    return summaryClassWeights;
}

namespace NCB {
    TVector<float> CalculateClassWeights(
        TConstArrayRef<float> targetClasses,
        const TWeights<float>& itemWeights,
        ui32 classCount,
        EAutoClassWeightsType autoClassWeightsType,
        NPar::ILocalExecutor* localExecutor
    ) {
        CB_ENSURE(classCount > 0, "Class count should be > 0");
        TVector<float> summaryClassWeights = CalculateSummaryClassWeight(
            targetClasses,
            itemWeights,
            classCount,
            localExecutor
        );

        CB_ENSURE(summaryClassWeights.size() == classCount, "Number of classes and class weights mismatch");

        CATBOOST_INFO_LOG << "Class weights type: " << autoClassWeightsType << Endl;
        TVector<float> classWeights;
        classWeights.reserve(classCount);

        const float maxSummaryClassWeight = *MaxElement(summaryClassWeights.begin(),
                                                        summaryClassWeights.end());
        const auto weightFunction = GetWeightFunction(autoClassWeightsType);
        for (const auto& summaryClassWeight : summaryClassWeights) {
            classWeights.emplace_back(weightFunction(maxSummaryClassWeight, summaryClassWeight));
            CATBOOST_INFO_LOG << "Weight of class " << classWeights.size() - 1 << ": "
                              << classWeights.back() << Endl;
        }

        return classWeights;
    }
}
