#include "estimated_features.h"
#include "tree_ctr_permutations.h"

#include <catboost/libs/helpers/checksum.h>
#include <catboost/libs/model/model_build_helper.h>
#include <catboost/private/libs/algo/full_model_saver.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/quantization/grid_creator.h>
#include <catboost/private/libs/quantization/utils.h>

#include <algorithm>
#include <cmath>

namespace NCB {
namespace {
    constexpr ui64 MaxInputBytes = ui64(1) << 30;

    TVector<TVector<ui32>> MakeHistoryOrders(const TTrainingDataProvider& data,
            const NCatboostOptions::TCatBoostOptions& options, bool online) {
        if (options.BoostingOptions->DataPartitionType == EDataPartitionType::FeatureParallel) {
            return MakeMetalFeatureParallelHistoryOrders(data, options);
        }
        const ui32 count = !online || options.DataProcessingOptions->HasTimeFlag
            ? 1 : options.BoostingOptions->PermutationCount.Get();
        CB_ENSURE(count >= 1 && count <= 64, "Metal estimated-feature permutation count must be in [1,64]");
        TVector<TVector<ui32>> orders;
        for (ui32 p = 0; p < count; ++p) {
            // DocParallel uses the same CUDA block-size-one group shuffle.
            orders.push_back(MakeMetalFeatureParallelHistoryOrder(data, p, 1));
        }
        return orders;
    }
}

TModelSplit TMetalEstimatedFeatures::GetSplit(ui32 feature, ui32 bin) const {
    CB_ENSURE(feature < Features.size() && bin < Borders[feature].size(),
        "Metal returned an invalid estimated-feature split");
    return TModelSplit(TEstimatedFeatureSplit(Features[feature], Borders[feature][bin]));
}

void TMetalEstimatedFeatures::FinalizeModel(TFullModel* model, NPar::ILocalExecutor* executor) const {
    const auto& selected = model->ModelTrees->GetEstimatedFeatures();
    if (selected.empty()) return;
    CB_ENSURE(Estimators && LearnObjects, "Metal estimated-feature finalization requires its training estimators");
    CB_ENSURE(!model->TextProcessingCollection && !model->EmbeddingProcessingCollection,
        "Metal estimated-feature model has already been finalized");
    TTextProcessingCollection text;
    TEmbeddingProcessingCollection embedding;
    TVector<TEstimatedFeature> remapped;
    CreateProcessingCollections(*Estimators,
        LearnObjects->GetQuantizedFeaturesInfo()->GetTextDigitizers(),
        TVector<TEstimatedFeature>(selected.begin(), selected.end()),
        &text, &embedding, &remapped, executor);
    if (!text.Empty()) model->TextProcessingCollection = MakeIntrusive<TTextProcessingCollection>(std::move(text));
    if (!embedding.Empty()) model->EmbeddingProcessingCollection = MakeIntrusive<TEmbeddingProcessingCollection>(std::move(embedding));
    model->UpdateEstimatedFeaturesIndices(std::move(remapped));
}

TMetalEstimatedFeatures PrepareMetalEstimatedFeatures(const TTrainingDataProviders& trainingData,
        const NCatboostOptions::TCatBoostOptions& options, NPar::ILocalExecutor* executor) {
    TMetalEstimatedFeatures result;
    const auto& learn = *trainingData.Learn;
    const auto& objects = *learn.ObjectsData;
    const auto& layout = *objects.GetQuantizedFeaturesInfo()->GetFeaturesLayout();
    result.AllTextFeatures = CreateTextFeatures(layout);
    result.AllEmbeddingFeatures = CreateEmbeddingFeatures(layout);
    result.Estimators = trainingData.FeatureEstimators;
    result.LearnObjects = learn.ObjectsData;
    if (!result.Estimators || result.Estimators->Empty()) return result;
    const ui32 rows = learn.GetObjectCount();
    CB_ENSURE(rows > 0, "Metal estimated features require training objects");
    TMap<TEstimatedFeatureId, ui32> columns;
    result.Estimators->ForEach([&](TEstimatorId id, const TFeatureEstimatorPtr& estimator) {
        const auto meta = estimator->FeaturesMeta();
        if (meta.FeaturesCount) result.HasOnlineFeatures |= id.IsOnline;
        for (ui32 local = 0; local < meta.FeaturesCount; ++local) {
            columns[{id, local}] = result.Features.size();
            result.IsOnline.push_back(id.IsOnline);
            const ui32 maxBins = objects.GetQuantizedFeaturesInfo()->GetFloatFeatureBinarization(Max<ui32>()).BorderCount + 1;
            result.BinCountUpperBounds.push_back(meta.UniqueValuesUpperBoundHint ? (*meta.UniqueValuesUpperBoundHint)[local] : maxBins);
            const auto guid = result.Estimators->GetEstimatorGuid(id);
            result.Features.emplace_back(result.Estimators->GetEstimatorSourceFeatureIdx(id).TextFeatureId,
                guid, local, FeatureTypeToEstimatedSourceFeatureType(estimator->GetSourceType()));
        }
    });
    if (result.Features.empty()) return result;
    const auto orders = MakeHistoryOrders(learn, options, result.HasOnlineFeatures);
    CB_ENSURE(ui64(rows) * result.Features.size() * orders.size() <= MaxInputBytes,
        "Metal estimated-feature permutation inputs exceed the experimental 1 GiB limit");
    result.Borders.resize(result.Features.size());
    result.BinsByPermutation.resize(orders.size(), TVector<ui8>(ui64(rows) * result.Features.size()));
    const auto& binarization = objects.GetQuantizedFeaturesInfo()->GetFloatFeatureBinarization(Max<ui32>());
    CB_ENSURE(binarization.BorderCount <= 255, "Metal estimated features require at most 255 borders");
    TOnCpuGridBuilderFactory gridFactory;
    for (ui32 permutation = 0; permutation < orders.size(); ++permutation) {
        result.Estimators->ForEach([&](TEstimatorId id, const TFeatureEstimatorPtr& estimator) {
            const auto meta = estimator->FeaturesMeta();
            if (!meta.FeaturesCount) return;
            const bool packed = !meta.Type.empty() && meta.Type.front() == EFeatureCalcerType::BoW;
            if (!id.IsOnline && permutation) {
                for (ui32 local = 0; local < meta.FeaturesCount; ++local) {
                    const ui64 offset = ui64(columns.at({id, local})) * rows;
                    std::copy_n(result.BinsByPermutation[0].begin() + offset, rows,
                        result.BinsByPermutation[permutation].begin() + offset);
                }
                return;
            }
            TVector<ui8> seen(meta.FeaturesCount);
            auto save = [&](ui32 local, TConstArrayRef<float> values, TMaybe<ui32> test) {
                CB_ENSURE(local < meta.FeaturesCount, "Shared estimated feature index is outside its metadata");
                const ui32 column = columns.at({id, local});
                auto& borders = result.Borders[column];
                const ui32 expected = test ? trainingData.Test[*test]->GetObjectCount() : rows;
                CB_ENSURE(values.size() == expected, "Shared estimated feature row count does not match its Pool");
                for (float value : values) CB_ENSURE(std::isfinite(value), "Metal estimated features must be finite");
                if (!permutation && !test) {
                    if (packed) borders = {0.5f};
                    else {
                        TVector<float> sorted(values.begin(), values.end());
                        Sort(sorted);
                        borders = gridFactory.Create(binarization.BorderSelectionType)->BuildBorders(sorted, binarization.BorderCount);
                        // CUDA preserves constant estimated columns in its grid.
                        if (borders.empty()) borders.push_back(0.5f);
                    }
                    CB_ENSURE(borders.size() <= 255, "Metal estimated features require at most 255 borders");
                    result.BinsPerFeature = Max<ui32>(result.BinsPerFeature, borders.size() + 1);
                    result.Checksum = UpdateCheckSum(result.Checksum, id.Id, id.IsOnline, local, borders);
                }
                CB_ENSURE(!borders.empty(), "Metal estimated-feature borders must be computed from learn first");
                const auto bins = BinarizeLine<ui8>(values, ENanMode::Forbidden, borders);
                // GUIDs are intentionally excluded: shared estimators generate
                // fresh GUIDs on a snapshot retry. Their stable ids, borders,
                // values and configured processing definitions identify data.
                if (!test || !permutation) result.Checksum = UpdateCheckSum(result.Checksum,
                    permutation, test ? *test + 1 : 0u, column, values, bins);
                if (!test) {
                    CB_ENSURE(!seen[local], "Shared estimated feature was emitted twice");
                    seen[local] = 1;
                    std::copy(bins.begin(), bins.end(), result.BinsByPermutation[permutation].begin() + ui64(column) * rows);
                }
            };
            auto visitor = [&](TMaybe<ui32> test) {
                if (packed) return TCalculatedFeatureVisitor(TCalculatedFeatureVisitor::TPackedFeatureWriter(
                    [&, test](TConstArrayRef<ui32> ids, TConstArrayRef<ui32> values) {
                        CB_ENSURE(ids.size() <= 32, "Shared estimated-feature pack exceeds 32 bits");
                        TVector<float> unpacked(values.size());
                        for (ui32 bit = 0; bit < ids.size(); ++bit) {
                            for (ui32 row = 0; row < values.size(); ++row) unpacked[row] = (values[row] >> bit) & 1u;
                            save(ids[bit], unpacked, test);
                        }
                    }));
                return TCalculatedFeatureVisitor(TCalculatedFeatureVisitor::TSingleFeatureWriter(
                    [&, test](ui32 local, TConstArrayRef<float> values) { save(local, values, test); }));
            };
            TVector<TCalculatedFeatureVisitor> testVisitors;
            for (ui32 test = 0; test < trainingData.Test.size(); ++test) testVisitors.push_back(visitor(test));
            if (id.IsOnline) result.Estimators->GetOnlineFeatureEstimator(id.Id)->ComputeOnlineFeatures(
                orders[permutation], visitor(Nothing()), testVisitors, executor);
            else estimator->ComputeFeatures(visitor(Nothing()), testVisitors, executor);
            for (ui8 emitted : seen) CB_ENSURE(emitted, "Shared estimated feature was not emitted");
        });
    }
    return result;
}
}
