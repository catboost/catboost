#pragma once

#include <catboost/libs/model/enums.h>
#include <catboost/libs/model/fwd.h>

namespace NCB {
    class TObjectsDataProvider;
    class IFeaturesBlockIterator;

    TIntrusivePtr<NCB::NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const IFeaturesBlockIterator& featuresBlockIterator,
        size_t start,
        size_t end,
        EFormulaEvaluatorType formulaEvaluatorType = EFormulaEvaluatorType::CPU);

    TIntrusivePtr<NCB::NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        size_t start,
        size_t end,
        EFormulaEvaluatorType formulaEvaluatorType = EFormulaEvaluatorType::CPU);

    TIntrusivePtr<NCB::NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData);
}
