#pragma once

#include "categorical.h"
#include "estimated_features.h"
#include "feature_manager_ids.h"
#include "full_matrix_rsm.h"

#include <catboost/private/libs/ctr_description/ctr_config.h>
#include <catboost/private/libs/options/enum_helpers.h>

#include <util/generic/map.h>
#include <util/generic/ylimits.h>

#include <tuple>

namespace NCB {
    struct TMetalStaticFeatureMetadata {
        TVector<ui32> DenseToManagerIds;
        ui32 ManagerCount = 0;
        TVector<TMetalRsmFeature> RsmFeatures;
    };

    // TMetalData is local to the native adapter. Keep this metadata builder
    // templated so every score consumer can share its source feature IDs.
    template <class TMetalData>
    TMetalStaticFeatureMetadata MakeMetalStaticFeatureMetadata(const TMetalData& data,
            const TTrainingDataProvider& learn, const NCatboostOptions::TCatBoostOptions& options) {
        const auto& objects = *learn.ObjectsData;
        const auto original = MakeMetalOriginalFeatureManagerIds(*objects.GetFeaturesLayout(),
            *objects.GetQuantizedFeaturesInfo());
        const auto& categories = data.Categorical;
        const ui32 numericCount = data.FeatureIndices.size();
        const ui32 categoryCount = categories.SplitCandidates.size();
        const ui32 estimatedCount = data.Estimated.GetFeatureCount();
        CB_ENSURE(categories.ColumnModelCtrs.size() == categoryCount &&
            categories.ColumnCtrBinarizations.size() == categoryCount &&
            categories.ColumnCatFeatureIndices.size() == categoryCount &&
            categories.ColumnUniqueValuesOnAll.size() == categoryCount,
            "Metal categorical source metadata differs from its runtime columns");
        CB_ENSURE(ui64(original.NextId) + estimatedCount < Max<ui32>(),
            "Metal estimated feature manager IDs overflow");
        TMetalStaticFeatureMetadata result;
        result.DenseToManagerIds.resize(numericCount + categoryCount + estimatedCount, Max<ui32>());
        result.ManagerCount = original.NextId + estimatedCount;
        // CUDA deliberately omits exclusive bundles for full-matrix ranking
        // and the two multiclass losses. Other losses retain bundle metadata
        // in CPU-prequantized Pools, despite fresh GPU quantization disabling
        // bundling. Every bundle registers before simple CTRs.
        TConstArrayRef<TExclusiveFeaturesBundle> bundles;
        if (!IsGpuPlainDocParallelOnlyMode(options.LossFunctionDescription->GetLossFunction()))
            bundles = objects.GetExclusiveFeatureBundlesMetaData();
        CB_ENSURE(ui64(result.ManagerCount) + bundles.size() < Max<ui32>(),
            "Metal exclusive bundle source IDs overflow");
        const ui32 firstBundle = result.ManagerCount;
        result.ManagerCount += bundles.size();
        TVector<ui32> originalToBundle(original.NextId, Max<ui32>());
        for (ui32 bundle = 0; bundle < bundles.size(); ++bundle) {
            for (const auto& part : bundles[bundle].Parts) {
                const ui32 flat = objects.GetFeaturesLayout()->GetExternalFeatureIdx(part.FeatureIdx, part.FeatureType);
                const auto& ids = original.ByFlatIndex[flat];
                CB_ENSURE(ids.size() == 1, "CUDA exclusive bundles require unsliced original features");
                CB_ENSURE(originalToBundle[ids.front()] == Max<ui32>(),
                    "An original feature belongs to multiple exclusive bundles");
                originalToBundle[ids.front()] = bundle;
            }
        }
        TMap<ui32, TMetalRsmFeature> byManager;
        auto add = [&](ui32 dense, ui32 manager, ui32 folds, ui32 bins, bool ctr, bool dependent) {
            result.DenseToManagerIds[dense] = manager;
            auto [it, inserted] = byManager.emplace(manager,
                TMetalRsmFeature{manager, folds, bins, ctr, dependent, {}});
            CB_ENSURE(inserted || (it->second.FoldCount == folds && it->second.BinCount == bins &&
                it->second.IsCtr == ctr && it->second.PermutationDependent == dependent),
                "Equivalent Metal columns have inconsistent source feature grids");
            it->second.RuntimeFeatures.push_back(dense);
        };
        auto addOriginal = [&](ui32 dense, ui32 manager, ui32 folds, ui32 bins) {
            if (originalToBundle[manager] != Max<ui32>()) {
                const ui32 bundle = originalToBundle[manager];
                const ui32 bundleBins = bundles[bundle].GetBinCount();
                // Search weights address the bundle, before ToSplit translates
                // the winning predicate back to its original model feature.
                add(dense, firstBundle + bundle, bundleBins - 1, bundleBins, false, false);
            } else {
                add(dense, manager, folds, bins, false, false);
            }
        };
        for (ui32 dense = 0; dense < numericCount; ++dense) {
            const auto& feature = data.AllFloatFeatures[data.FeatureIndices[dense]];
            const auto& ids = original.ByFlatIndex[feature.Position.FlatIndex];
            CB_ENSURE(ids.size() == 1, "Metal numeric source metadata requires at most 255 borders per column");
            addOriginal(dense, ids.front(), feature.Borders.size(), feature.Borders.size() + 1);
        }

        // GetKnownSimpleCtrTypes creates the global binarization registry by
        // scanning simple descriptions, then per-feature descriptions in map
        // order. TCtrConfig then sorts by prior before target parameter.
        TVector<NCatboostOptions::TBinarizationOptions> binarizations;
        auto binarizationId = [&](const NCatboostOptions::TBinarizationOptions& value) {
            for (ui32 i = 0; i < binarizations.size(); ++i) if (binarizations[i] == value) return i;
            binarizations.push_back(value);
            return static_cast<ui32>(binarizations.size() - 1);
        };
        for (const auto& description : options.CatFeatureParams->SimpleCtrs.Get())
            binarizationId(description.GetCtrBinarization());
        for (const auto& perFeature : options.CatFeatureParams->PerFeatureCtrs.Get())
            for (const auto& description : perFeature.second) binarizationId(description.GetCtrBinarization());

        using TCtrKey = std::pair<ui32, TCtrConfig>;
        TMap<TCtrKey, TVector<ui32>> ctrColumns;
        for (ui32 column = 0; column < categoryCount; ++column) {
            const ui32 catIndex = categories.ColumnCatFeatureIndices[column];
            CB_ENSURE(catIndex < categories.AllCatFeatures.size(), "Metal source categorical index is out of range");
            const auto& cat = categories.AllCatFeatures[catIndex];
            const auto& ids = original.ByFlatIndex[cat.Position.FlatIndex];
            CB_ENSURE(ids.size() == 1, "A categorical feature must have one source manager ID");
            if (categories.ColumnModelCtrs[column]) {
                CB_ENSURE(categories.ColumnCtrBinarizations[column], "Metal CTR binarization metadata is missing");
                const auto& ctr = *categories.ColumnModelCtrs[column];
                TCtrConfig config;
                config.Type = ctr.Base.CtrType;
                config.Prior = {ctr.PriorNum, ctr.PriorDenom};
                config.ParamId = ctr.TargetBorderIdx;
                config.CtrBinarizationConfigId = binarizationId(*categories.ColumnCtrBinarizations[column]);
                ctrColumns[{ids.front(), config}].push_back(column);
            } else {
                const ui32 bins = categories.ColumnUniqueValuesOnAll[column];
                const bool oneHot = bins > 1 && bins <= options.CatFeatureParams->OneHotMaxSize;
                const ui32 folds = !oneHot ? 0 : bins > 2 ? bins : bins - 1;
                addOriginal(numericCount + column, ids.front(), folds, bins);
            }
        }
        // CUDA registers all estimated descriptors before any simple CTR,
        // even if their runtime columns were appended after categories.
        CB_ENSURE(data.Estimated.Borders.size() == estimatedCount && data.Estimated.IsOnline.size() == estimatedCount &&
            data.Estimated.BinCountUpperBounds.size() == estimatedCount,
            "Metal estimated source metadata differs from its runtime columns");
        for (ui32 column = 0; column < estimatedCount; ++column) {
            const ui32 bins = data.Estimated.BinCountUpperBounds[column];
            add(numericCount + categoryCount + column, original.NextId + column,
                bins ? bins - 1 : 0, bins, false, data.Estimated.IsOnline[column] != 0);
        }
        // GetLearnFeatureIds registers category IDs in order, then its CTR
        // types/configurations in sorted order; empty grids still own an ID.
        for (const auto& item : ctrColumns) {
            CB_ENSURE(result.ManagerCount < Max<ui32>(), "Metal CTR source manager IDs overflow");
            const ui32 manager = result.ManagerCount++;
            for (ui32 column : item.second) {
                const auto& description = *categories.ColumnCtrBinarizations[column];
                const ui32 bins = description.BorderCount + 1;
                add(numericCount + column, manager, bins - 1, bins, true,
                    item.first.second.Type != ECtrType::FeatureFreq);
            }
        }
        for (auto& item : byManager) result.RsmFeatures.push_back(std::move(item.second));
        return result;
    }
}
