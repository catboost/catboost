#include "estimated_features_apply.h"

#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/model/cpu/evaluator.h>
#include <catboost/libs/model/cpu/quantization.h>
#include <catboost/private/libs/algo/apply.h>
#include <catboost/private/libs/algo/features_data_helpers.h>

namespace NCB {
namespace {

struct TEstimatedApplyCalcer {
    TTextFeatureCalcerPtr Text;
    TEmbeddingFeatureCalcerPtr Embedding;
    IDynamicBlockIteratorPtr<TText> TextRows;
    IDynamicBlockIteratorPtr<TConstEmbedding> EmbeddingRows;
    TVector<float> Values;

    ui32 FeatureCount() const {
        return Text ? Text->FeatureCount() : Embedding->FeatureCount();
    }

    void ComputeBlock(size_t rows) {
        const ui32 featureCount = FeatureCount();
        Values.assign(rows * featureCount, 0.0f);
        if (Text) {
            const auto texts = TextRows->Next(rows);
            CB_ENSURE(texts.size() == rows, "Metal estimated apply text row count differs");
            for (size_t row = 0; row < rows; ++row) {
                Text->Compute(texts[row], TOutputFloatIterator(
                    Values.data() + row, rows, Values.size() - row));
            }
        } else {
            const auto embeddings = EmbeddingRows->Next(rows);
            CB_ENSURE(embeddings.size() == rows, "Metal estimated apply embedding row count differs");
            for (size_t row = 0; row < rows; ++row) {
                Embedding->Compute(embeddings[row], TOutputFloatIterator(
                    Values.data() + row, rows, Values.size() - row));
            }
        }
    }
};

} // namespace

TVector<TVector<double>> ApplyMetalModelWithEstimatedFeatures(
    const TFullModel& model,
    const TTrainingDataProvider& data,
    const TFeatureEstimatorsPtr& estimators,
    NPar::ILocalExecutor* executor)
{
    const auto& trees = *model.ModelTrees;
    const ui32 rows = data.GetObjectCount();
    const ui32 dimensions = model.GetDimensionsCount();
    TVector<TVector<double>> result(dimensions, TVector<double>(rows));
    if (!rows) {
        return result;
    }
    if (!model.GetTreeCount()) {
        const auto& bias = model.GetScaleAndBias().GetBiasRef();
        CB_ENSURE(bias.size() == dimensions, "Metal estimated apply model bias dimension differs");
        for (ui32 dim = 0; dim < dimensions; ++dim) {
            Fill(result[dim].begin(), result[dim].end(), bias[dim]);
        }
        return result;
    }
    if (trees.GetEstimatedFeatures().empty()) {
        return ApplyModelMulti(model, *data.ObjectsData, EPredictionType::RawFormulaVal,
            0, model.GetTreeCount(), executor);
    }
    CB_ENSURE(estimators, "Metal estimated apply requires training feature estimators");
    const auto& objects = *data.ObjectsData;

    THashMap<TGuid, ui32> calcerIndexes;
    TVector<TEstimatedApplyCalcer> calcers;
    TVector<ui32> featureCalcers;
    featureCalcers.reserve(trees.GetEstimatedFeatures().size());
    for (const auto& feature : trees.GetEstimatedFeatures()) {
        const auto& descriptor = feature.ModelEstimatedFeature;
        auto found = calcerIndexes.find(descriptor.CalcerId);
        if (found == calcerIndexes.end()) {
            const auto source = estimators->GetEstimatorSourceFeatureIdx(descriptor.CalcerId);
            TEstimatedApplyCalcer calcer;
            if (descriptor.SourceFeatureType == EEstimatedSourceFeatureType::Text) {
                CB_ENSURE(model.TextProcessingCollection,
                    "Metal estimated apply requires finalized text calcers");
                CB_ENSURE(source.TokenizedFeatureId < objects.GetFeaturesLayout()->GetTextFeatureCount(),
                    "Metal estimated apply tokenized feature index is out of range");
                const auto holder = objects.GetTextFeature(source.TokenizedFeatureId);
                CB_ENSURE(holder, "Metal estimated apply tokenized source is unavailable");
                calcer.Text = model.TextProcessingCollection->GetCalcer(descriptor.CalcerId);
                calcer.TextRows = (*holder)->GetBlockIterator();
            } else {
                CB_ENSURE(descriptor.SourceFeatureType == EEstimatedSourceFeatureType::Embedding,
                    "Metal estimated apply has an unknown estimated source type");
                CB_ENSURE(model.EmbeddingProcessingCollection,
                    "Metal estimated apply requires finalized embedding calcers");
                CB_ENSURE(source.TextFeatureId < objects.GetFeaturesLayout()->GetEmbeddingFeatureCount(),
                    "Metal estimated apply embedding feature index is out of range");
                const auto holder = objects.GetEmbeddingFeature(source.TextFeatureId);
                CB_ENSURE(holder, "Metal estimated apply embedding source is unavailable");
                calcer.Embedding = model.EmbeddingProcessingCollection->GetCalcer(descriptor.CalcerId);
                calcer.EmbeddingRows = (*holder)->GetBlockIterator();
            }
            CB_ENSURE(calcer.FeatureCount() > 0, "Metal estimated apply calcer has no output features");
            const ui32 index = calcers.size();
            calcers.push_back(std::move(calcer));
            found = calcerIndexes.emplace(descriptor.CalcerId, index).first;
        }
        const auto& calcer = calcers[found->second];
        CB_ENSURE(descriptor.LocalId >= 0 && static_cast<ui32>(descriptor.LocalId) < calcer.FeatureCount(),
            "Metal estimated apply requires finalized calcer-local feature indices");
        featureCalcers.push_back(found->second);
    }

    // Original text columns are replaced by tokenized columns during training.
    // Check only ordinary columns against the quantized layout. A metadata-only
    // model preserves shared numeric/category name remapping without requiring
    // the unavailable original text columns to match the tokenized layout.
    TFullModel ordinaryModel;
    ordinaryModel.ModelTrees.GetMutable()->SetFloatFeatures(
        TVector<TFloatFeature>(trees.GetFloatFeatures().begin(), trees.GetFloatFeatures().end()));
    ordinaryModel.ModelTrees.GetMutable()->SetCatFeatures(
        TVector<TCatFeature>(trees.GetCatFeatures().begin(), trees.GetCatFeatures().end()));
    ordinaryModel.UpdateDynamicData();
    THashMap<ui32, ui32> columnRemap;
    CheckModelAndDatasetCompatibility(ordinaryModel, objects, &columnRemap);
    TQuantizedFeaturesBlockIterator ordinaryFeatures(ordinaryModel, objects, columnRemap, 0);
    auto evaluator = NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::CPU, model);
    evaluator->SetPredictionType(NModelEvaluation::EPredictionType::RawFormulaVal);
    const auto applyData = trees.GetApplyData();
    const size_t buckets = trees.GetEffectiveBinaryFeaturesBucketsCount();
    for (ui32 start = 0; start < rows; start += NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE) {
        const size_t count = Min<size_t>(NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE, rows - start);
        ordinaryFeatures.NextBlock(count);
        const auto accessor = ordinaryFeatures.GetAccessor();
        const auto floatAccessor = accessor.GetFloatAccessor();
        for (auto& calcer : calcers) {
            calcer.ComputeBlock(count);
        }
        NModelEvaluation::TCPUEvaluatorQuantizedData quantized;
        quantized.QuantizedData = TMaybeOwningArrayHolder<ui8>::CreateOwning(TVector<ui8>(buckets * count));
        quantized.BlockStride = buckets * NModelEvaluation::FORMULA_EVALUATION_BLOCK_SIZE;
        quantized.BlocksCount = 1;
        quantized.ObjectsCount = count;
        ui8* const blockStart = quantized.QuantizedData.data();
        ui8* write = blockStart;
        for (const auto& feature : trees.GetFloatFeatures()) {
            if (!feature.UsedInModel()) {
                continue;
            }
            CB_ENSURE(feature.Borders.size() <= 255,
                "Metal estimated apply requires at most 255 borders per numeric feature");
            NModelEvaluation::BinarizeQuantizedFloatFeature(feature, floatAccessor, 0, count, write);
        }
        for (size_t index = 0; index < trees.GetEstimatedFeatures().size(); ++index) {
            const auto& feature = trees.GetEstimatedFeatures()[index];
            const auto& values = calcers[featureCalcers[index]].Values;
            const float* featureValues = values.data() + feature.ModelEstimatedFeature.LocalId * count;
            NModelEvaluation::BinarizeFloats<false>(TFeaturePosition(), count,
                [featureValues](TFeaturePosition, size_t row) { return featureValues[row]; },
                MakeConstArrayRef(feature.Borders), 0, write);
        }
        TVector<ui32> transposedHash(count * applyData->UsedCatFeaturesCount);
        TVector<float> ctrs(count * applyData->UsedModelCtrs.size());
        NModelEvaluation::ComputeOneHotAndCtrFeaturesForBlock(trees, *applyData, model.CtrProvider,
            accessor.GetCatAccessor(), 0, count, blockStart, transposedHash, ctrs, &write);
        CB_ENSURE(write == blockStart + buckets * count,
            "Metal estimated apply model bucket layout differs");
        TVector<double> flatResult(count * dimensions);
        evaluator->Calc(&quantized, 0, model.GetTreeCount(), flatResult);
        for (size_t row = 0; row < count; ++row) {
            for (ui32 dim = 0; dim < dimensions; ++dim) {
                result[dim][start + row] = flatResult[row * dimensions + dim];
            }
        }
    }
    return result;
}

} // namespace NCB
