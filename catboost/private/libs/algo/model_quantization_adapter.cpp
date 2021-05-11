#include "model_quantization_adapter.h"

#include "features_data_helpers.h"

#include <catboost/libs/model/cpu/quantization.h>
#include <catboost/libs/model/cuda/evaluator.cuh>
#include <catboost/libs/model/enums.h>

using namespace NCB;
using namespace NCB::NModelEvaluation;


namespace {

    class TMakeQuantizedFeaturesVisitor final : public IFeaturesBlockIteratorVisitor {
    public:
        TMakeQuantizedFeaturesVisitor(const TFullModel& model, size_t objectsStart, size_t objectsEnd,
                                      EFormulaEvaluatorType formulaEvaluatorType = EFormulaEvaluatorType::CPU)
            : Model(model)
            , ObjectsStart(objectsStart)
            , ObjectsEnd(objectsEnd)
            , FormulaEvaluatorType(formulaEvaluatorType)
        {
            if (FormulaEvaluatorType == EFormulaEvaluatorType::CPU) {
                ResultCpu = MakeIntrusive<TCPUEvaluatorQuantizedData>();
                ResultCpu->QuantizedData = TMaybeOwningArrayHolder<ui8>::CreateOwning(
                        TVector<ui8>(
                                Model.ModelTrees->GetEffectiveBinaryFeaturesBucketsCount()
                                * (ObjectsEnd - ObjectsStart)));
            } else {
                ResultGpu = MakeIntrusive<TCudaQuantizedData>();
                ResultGpu->SetDimensions(Model.ModelTrees->GetFloatFeatures().size(), ObjectsEnd - ObjectsStart);
            }
        }

        void Visit(const TRawFeaturesBlockIterator& rawFeaturesBlockIterator) override {
            TRawFeatureAccessor rawFeatureAccessor = rawFeaturesBlockIterator.GetAccessor();

            const auto docCount = ObjectsEnd - ObjectsStart;
            const auto blockSize = Min(docCount, FORMULA_EVALUATION_BLOCK_SIZE);
            TVector<ui32> transposedHash(blockSize * Model.GetUsedCatFeaturesCount());
            auto applyData = Model.ModelTrees->GetApplyData();
            TVector<float> ctrs(applyData->UsedModelCtrs.size() * blockSize);
            TVector<float> estimatedFeatures(Model.ModelTrees->GetEstimatedFeatures().size() * blockSize);

            if (FormulaEvaluatorType == EFormulaEvaluatorType::CPU) {
                BinarizeFeatures(
                        *Model.ModelTrees,
                        *applyData,
                        Model.CtrProvider,
                        Model.TextProcessingCollection,
                        Model.EmbeddingProcessingCollection,
                        rawFeatureAccessor.GetFloatAccessor(),
                        rawFeatureAccessor.GetCatAccessor(),
                        rawFeatureAccessor.GetTextAccessor(),
                        rawFeatureAccessor.GetEmbeddingAccessor(),
                        0,
                        docCount,
                        ResultCpu.Get(),
                        transposedHash,
                        ctrs,
                        estimatedFeatures
                );
            } else {
                size_t featureCount = Model.ModelTrees->GetFloatFeatures().size();
                TVector<TVector<float>> featuresVec(docCount, TVector<float>(featureCount));

                TVector<TConstArrayRef<float>> featuresVecSecond(docCount);
                for (size_t i = 0; i < docCount; ++i) {
                    for (size_t j = 0; j < featureCount; ++j) {
                        auto featurePosition = Model.ModelTrees->GetFloatFeatures()[j].Position;
                        featuresVec[i][featurePosition.Index] = rawFeatureAccessor.GetFloatAccessor()(featurePosition, i);
                    }
                    featuresVecSecond[i] = MakeArrayRef(featuresVec[i].begin(), featuresVec[i].end());
                }

                IQuantizedData* iQuantizedData = reinterpret_cast<IQuantizedData*>(ResultGpu.Get());
                Model.GetCurrentEvaluator()->Quantize(featuresVecSecond, iQuantizedData);
            };
        }

        void Visit(const TQuantizedFeaturesBlockIterator& quantizedFeaturesBlockIterator) override {
            TQuantizedFeatureAccessor quantizedFeatureAccessor = quantizedFeaturesBlockIterator.GetAccessor();

            if (FormulaEvaluatorType == EFormulaEvaluatorType::CPU) {
                const auto docCount = ObjectsEnd - ObjectsStart;
                const auto blockSize = Min(docCount, FORMULA_EVALUATION_BLOCK_SIZE);
                TVector <ui32> transposedHash(blockSize * Model.GetUsedCatFeaturesCount());
                auto applyData = Model.ModelTrees->GetApplyData();
                TVector<float> ctrs(applyData->UsedModelCtrs.size() * blockSize);

                ComputeEvaluatorFeaturesFromPreQuantizedData(
                        *Model.ModelTrees,
                        *applyData,
                        Model.CtrProvider,
                        quantizedFeatureAccessor.GetFloatAccessor(),
                        quantizedFeatureAccessor.GetCatAccessor(),
                        0,
                        docCount,
                        ResultCpu.Get(),
                        transposedHash,
                        ctrs
                );
            } else {
                CB_ENSURE(false, "Cannot apply visitor to GPU quantized data, please contact catboost developers via GitHub issue or in support chat");
            }
        }

        TIntrusivePtr<IQuantizedData> GetResult() {
            if (FormulaEvaluatorType == EFormulaEvaluatorType::CPU) {
                return std::move(ResultCpu);
            }
            return std::move(ResultGpu);
        }

    private:
        const TFullModel& Model;
        size_t ObjectsStart;
        size_t ObjectsEnd;
        TIntrusivePtr<TCPUEvaluatorQuantizedData> ResultCpu;
        TIntrusivePtr<TCudaQuantizedData> ResultGpu;
        EFormulaEvaluatorType FormulaEvaluatorType;
    };

}

namespace NCB {
    TIntrusivePtr<NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const IFeaturesBlockIterator& featuresBlockIterator,
        size_t start,
        size_t end,
        EFormulaEvaluatorType formulaEvaluatorType) {

        TMakeQuantizedFeaturesVisitor visitor(model, start, end, formulaEvaluatorType);
        featuresBlockIterator.Accept(&visitor);
        return visitor.GetResult();
    }

    TIntrusivePtr<NModelEvaluation::IQuantizedData> MakeQuantizedFeaturesForEvaluator(
        const TFullModel& model,
        const TObjectsDataProvider& objectsData,
        size_t start,
        size_t end,
        EFormulaEvaluatorType formulaEvaluatorType) {

        THolder<IFeaturesBlockIterator> featuresBlockIterator
            = CreateFeaturesBlockIterator(model, objectsData, start, end);
        featuresBlockIterator->NextBlock(end - start);

        return MakeQuantizedFeaturesForEvaluator(model, *featuresBlockIterator, start, end, formulaEvaluatorType);
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
