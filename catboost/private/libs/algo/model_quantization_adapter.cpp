#include "model_quantization_adapter.h"

#include "features_data_helpers.h"

#include <catboost/libs/model/cpu/quantization.h>

using namespace NCB;
using namespace NCB::NModelEvaluation;


namespace {

    class TMakeQuantizedFeaturesVisitor final : public IFeaturesBlockIteratorVisitor {
    public:
        TMakeQuantizedFeaturesVisitor(const TFullModel& model, size_t objectsStart, size_t objectsEnd)
            : Model(model)
            , ObjectsStart(objectsStart)
            , ObjectsEnd(objectsEnd)
            , Result(MakeIntrusive<TCPUEvaluatorQuantizedData>())
        {
            Result->QuantizedData = TMaybeOwningArrayHolder<ui8>::CreateOwning(
                TVector<ui8>(
                    Model.ModelTrees->GetEffectiveBinaryFeaturesBucketsCount()
                        * (ObjectsEnd - ObjectsStart)));
        }

        void Visit(const TRawFeaturesBlockIterator& rawFeaturesBlockIterator) override {
            TRawFeatureAccessor rawFeatureAccessor = rawFeaturesBlockIterator.GetAccessor();

            const auto docCount = ObjectsEnd - ObjectsStart;
            const auto blockSize = Min(docCount, FORMULA_EVALUATION_BLOCK_SIZE);
            TVector<ui32> transposedHash(blockSize * Model.GetUsedCatFeaturesCount());
            TVector<float> ctrs(Model.ModelTrees->GetUsedModelCtrs().size() * blockSize);
            TVector<float> estimatedFeatures(Model.ModelTrees->GetEstimatedFeatures().size() * blockSize);

            BinarizeFeatures(
                *Model.ModelTrees,
                Model.CtrProvider,
                Model.TextProcessingCollection,
                rawFeatureAccessor.GetFloatAccessor(),
                rawFeatureAccessor.GetCatAccessor(),
                rawFeatureAccessor.GetTextAccessor(),
                0,
                docCount,
                Result.Get(),
                transposedHash,
                ctrs,
                estimatedFeatures
            );
        }

        void Visit(const TQuantizedFeaturesBlockIterator& quantizedFeaturesBlockIterator) override {
            TQuantizedFeatureAccessor quantizedFeatureAccessor = quantizedFeaturesBlockIterator.GetAccessor();
            AssignFeatureBins(
               *Model.ModelTrees,
               quantizedFeatureAccessor.GetFloatAccessor(),
               quantizedFeatureAccessor.GetCatAccessor(),
               0,
               ObjectsEnd - ObjectsStart,
               Result.Get());
        }

        TIntrusivePtr<TCPUEvaluatorQuantizedData> GetResult() {
            return std::move(Result);
        }

    private:
        const TFullModel& Model;
        size_t ObjectsStart;
        size_t ObjectsEnd;
        TIntrusivePtr<TCPUEvaluatorQuantizedData> Result;
    };

}

namespace NCB {
    TIntrusivePtr<NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const IFeaturesBlockIterator& featuresBlockIterator,
        size_t start,
        size_t end) {

        TMakeQuantizedFeaturesVisitor visitor(model, start, end);
        featuresBlockIterator.Accept(&visitor);
        return visitor.GetResult();
    }

    TIntrusivePtr<NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        size_t start,
        size_t end) {

        THolder<IFeaturesBlockIterator> featuresBlockIterator
            = CreateFeaturesBlockIterator(model, objectsData, start, end);
        featuresBlockIterator->NextBlock(end - start);

        return MakeQuantizedFeaturesForEvaluator(model, *featuresBlockIterator, start, end);
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
