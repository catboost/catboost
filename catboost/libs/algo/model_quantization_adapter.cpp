#include "model_quantization_adapter.h"

#include "features_data_helpers.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/cpu/quantization.h>

using namespace NCB;
using namespace NCB::NModelEvaluation;

static void BinarizeRawFeatures(
    const TFullModel& model,
    const TRawObjectsDataProvider& rawObjectsData,
    size_t start,
    size_t end,
    TCPUEvaluatorQuantizedData* result) {

    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, rawObjectsData, &columnReorderMap);
    const auto docCount = end - start;
    const auto blockSize = Min(docCount, FORMULA_EVALUATION_BLOCK_SIZE);
    TVector<ui32> transposedHash(blockSize * model.GetUsedCatFeaturesCount());
    TVector<float> ctrs(model.ObliviousTrees->GetUsedModelCtrs().size() * blockSize);
    TRawFeatureAccessor featureAccessor(model, rawObjectsData, columnReorderMap, start, end);

    BinarizeFeatures(
        *model.ObliviousTrees,
        model.CtrProvider,
        featureAccessor.GetFloatAccessor(),
        featureAccessor.GetCatAccessor(),
        0,
        docCount,
        result,
        transposedHash,
        ctrs
    );
}

static void AssignFeatureBins(
    const TFullModel& model,
    const TQuantizedForCPUObjectsDataProvider& quantizedObjectsData,
    size_t start,
    size_t end,
    TCPUEvaluatorQuantizedData* cpuEvaluatorQuantizedData)
{
    THashMap<ui32, ui32> columnReorderMap;
    CheckModelAndDatasetCompatibility(model, quantizedObjectsData, &columnReorderMap);
    TQuantizedFeatureAccessor quantizedFeatureAccessor(
        model,
        quantizedObjectsData,
        columnReorderMap,
        start,
        end
    );

    AssignFeatureBins(
        *model.ObliviousTrees,
        quantizedFeatureAccessor.GetFloatAccessor(),
        quantizedFeatureAccessor.GetCatAccessor(),
        0,
        end - start,
        cpuEvaluatorQuantizedData
    );
}

namespace NCB {
    TIntrusivePtr<NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        size_t start,
        size_t end) {

        TIntrusivePtr<TCPUEvaluatorQuantizedData> result = MakeIntrusive<TCPUEvaluatorQuantizedData>();
        result->QuantizedData = TMaybeOwningArrayHolder<ui8>::CreateOwning(TVector<ui8>(
            model.ObliviousTrees->GetEffectiveBinaryFeaturesBucketsCount() * (end - start)
        ));
        if (const auto* const rawObjectsData = dynamic_cast<const TRawObjectsDataProvider*>(&objectsData)) {
            BinarizeRawFeatures(model, *rawObjectsData, start, end, result.Get());
        } else if (
            const auto* const quantizedObjectsData
                = dynamic_cast<const TQuantizedForCPUObjectsDataProvider*>(&objectsData)) {
            AssignFeatureBins(model, *quantizedObjectsData, start, end, result.Get());
        } else {
            ythrow TCatBoostException() << "Unsupported objects data - neither raw nor quantized for CPU";
        }
        return result;
    }

    TIntrusivePtr<NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData) {
        return MakeQuantizedFeaturesForEvaluator(
            model,
            objectsData,
            /*start*/0,
            objectsData.GetObjectCount()
        );
    }
}
