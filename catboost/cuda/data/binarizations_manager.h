#pragma once

#include "feature.h"

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/private/libs/ctr_description/ctr_config.h>
#include <catboost/libs/data/cat_feature_perfect_hash.h>
#include <catboost/libs/data/exclusive_feature_bundling.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/feature_estimators.h>
#include <catboost/libs/data/quantized_features_info.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/binarization_options.h>
#include <catboost/private/libs/options/cat_feature_options.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/map.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/system/types.h>

namespace NCatboostCuda {

    //stores expression for binarized features calculations and mapping from this expression to unique ids
    //WARNING: not thread-safe
    class TBinarizedFeaturesManager {
    public:
        /*
         * Separate featuresLayout parameter is necessary because FeaturesLayout in quantizedFeaturesInfo
         *   contains ignored features information at the moment of quantization but it is possible that
         *   some additional features are set to be ignored at later processing stages
         *   (For example, for feature evaluation)
         */
        TBinarizedFeaturesManager(const NCatboostOptions::TCatFeatureParams& catFeatureOptions,
                                  NCB::TFeatureEstimatorsPtr estimators,
                                  const NCB::TFeaturesLayout& featuresLayout,
                                  const TVector<NCB::TExclusiveFeaturesBundle>& learnExclusiveFeatureBundles,
                                  NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
                                  ui32 maxObjectsCount,
                                  bool enableShuffling = true);

        TBinarizedFeaturesManager(const TBinarizedFeaturesManager& featureManager, const TVector<ui32>& ignoredFeatureIds);

        ENanMode GetNanMode(const ui32 featureId) const;

        bool HasBorders(ui32 featureId) const;

        void SetBorders(ui32 featureId, TVector<float> borders) {
            CB_ENSURE(!HasBorders(featureId));
            Borders[featureId] = std::move(borders);
        }

        bool IsFloat(ui32 featureId) const;

        bool IsCat(ui32 featureId) const;

        bool IsEstimated(ui32 featureId) const;

        bool UseForOneHotEncoding(ui32 featureId) const;

        bool UseForCtr(ui32 featureId) const;

        bool UseForTreeCtr(ui32 featureId) const;

        bool IsTreeCtrsEnabled() const {
            return !DataProviderCatFeatureIdToFeatureManagerId.empty() &&
                   (CatFeatureOptions.MaxTensorComplexity > 1);
        }

        bool UseAsBaseTensorForTreeCtr(const TFeatureTensor& tensor) const {
            return (tensor.GetComplexity() < CatFeatureOptions.MaxTensorComplexity);
        }

        bool UseForTreeCtr(const TFeatureTensor& tensor) const;

        bool IsCtr(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor, "featureId " << featureId << ", Cursor " << Cursor);
            return InverseCtrs.contains(featureId);
        }

        ui32 GetCtrsCount() const {
            return InverseCtrs.size();
        }

        bool IsTreeCtr(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor);
            return IsCtr(featureId) && !GetCtr(featureId).IsSimple();
        }

        bool IsFeatureBundle(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor, "Unexpected feature id " << featureId << ", should be less than " << Cursor);
            return FeatureManagerIdToExclusiveBundleId.contains(featureId);
        }

        const NCB::TExclusiveFeaturesBundle& GetFeatureBundleForFeatureId(ui32 featureId) const {
            const auto exclusiveBundleId = FeatureManagerIdToExclusiveBundleId.at(featureId);
            return LearnExclusiveFeatureBundles.at(exclusiveBundleId);
        }

        TBinarySplit TranslateFeatureBundleSplitToBinarySplit(ui32 featureId, ui32 split) const {
            const auto exclusiveBundleId = FeatureManagerIdToExclusiveBundleId.at(featureId);
            const auto& bundleParts = LearnExclusiveFeatureBundles.at(exclusiveBundleId).Parts;
            ui32 partId = 0;
            if (split > 0) {
                for (auto i : xrange(bundleParts.size())) {
                    if (split < bundleParts[i].Bounds.End) {
                        partId = i;
                        break;
                    }
                }
            } else {
                for (auto i : xrange(bundleParts.size())) {
                    if (bundleParts[i].FeatureType == EFeatureType::Categorical) {
                        partId = i;
                        break;
                    }
                }
            }
            const auto& part = bundleParts[partId];
            auto featuresLayout = QuantizedFeaturesInfo->GetFeaturesLayout();
            auto dataProviderIdxForPart = featuresLayout->GetExternalFeatureIdx(part.FeatureIdx, part.FeatureType);
            if (part.FeatureType == EFeatureType::Float) {
                const auto floatIdx = DataProviderFloatFeatureIdToFeatureManagerId.at(dataProviderIdxForPart).at(0);
                return TBinarySplit{floatIdx, split - part.Bounds.Begin, EBinSplitType::TakeGreater};
            } else {
                auto catIdx = DataProviderCatFeatureIdToFeatureManagerId.at(dataProviderIdxForPart);
                return TBinarySplit{catIdx, split ? split - part.Bounds.Begin + 1 : 0, EBinSplitType::TakeBin};
            }
        }

        bool IsEstimatedFeature(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor);
            return FeatureManagerIdToEstimatedFeatureId.contains(featureId);
        }

        bool IsPermutationDependent(const TCtr& ctr) const;

        bool HasPermutationDependentSplit(const TVector<TBinarySplit>& splits) const;

        const TCtr& GetCtr(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor);
            return InverseCtrs.at(featureId);
        }

        ui32 GetFeatureCount() const {
            return Cursor;
        }

        ui32 GetTreeCtrCount() const {
            for (ui32 idx = 0; idx < Cursor; ++idx) {
                if (IsTreeCtr(idx)) {
                    return Cursor - idx;
                }
            }
            return 0;
        }

        ui32 GetFeatureManagerIdForCatFeature(ui32 dataProviderId) const;

        const TVector<ui32>& GetFeatureManagerIdForFloatFeature(ui32 dataProviderId) const;

        ui32 GetDataProviderId(ui32 featureId) const {
            return FeatureManagerIdToDataProviderId.at(featureId);
        }

        NCB::TEstimatedFeatureId GetEstimatedFeature(ui32 featureId) const {
            return FeatureManagerIdToEstimatedFeatureId.at(featureId);
        }

        NCB::TFeatureEstimatorsPtr GetFeatureEstimators() const {
            return FeatureEstimators;
        }

        ui32 GetEstimatedFeatureCount() const {
            return EstimatedFeatureToFeatureManagerId.size();
        }

        bool IsKnown(const TCtr& ctr) const {
            return KnownCtrs.contains(ctr);
        }

        ui32 AddCtr(const TCtr& ctr);

        ui32 AddCtr(const TCtr& ctr,
                    TVector<float>&& borders) {
            ui32 id = AddCtr(ctr);
            Borders[id] = std::move(borders);
            return id;
        }

        bool IsUsedCtr(ui32 featureId) const {
            return UsedCtrs.contains(featureId);
        }

        void AddUsedCtr(ui32 featureId) const {
            UsedCtrs.insert(featureId);
        }

        TVector<ui32> GetEstimatedFeatureIds() const;

        TVector<ui32> GetExclusiveFeatureBundleIds() const {
            TVector<ui32> result;
            result.reserve(FeatureManagerIdToExclusiveBundleId.size());
            for (auto it : FeatureManagerIdToExclusiveBundleId) {
                result.push_back(it.first);
            }
            return result;
        }

        ui32 GetExclusiveFeatureBundleIdxForFeatureManagerIdx(ui32 localIdx) const {
            return FeatureManagerIdToExclusiveBundleId.at(localIdx);
        }

        bool PresentInExclusiveFeatureBundle(ui32 featureId) const {
            return FeatureManagerFeaturesToBundleId.contains(featureId);
        }

        const TVector<float>& GetBorders(ui32 featureId) const;

        ui32 CtrsPerTreeCtrFeatureTensor() const;

        ui32 GetBinCount(ui32 localId) const;

        const NCatboostOptions::TBinarizationOptions& GetTargetBinarizationDescription() const {
            return CatFeatureOptions.TargetBinarization;
        }

        const NCatboostOptions::TBinarizationOptions& GetBinarizationDescription(const TCtr& ctr) const {
            return GetCtrBinarizationForConfig(ctr.Configuration);
        }

        ui32 GetId(const TCtr& ctr) const {
            CB_ENSURE(KnownCtrs.contains(ctr));
            return KnownCtrs[ctr];
        }

        ui32 GetId(const NCB::TEstimatedFeatureId& feature) const {
            CB_ENSURE(EstimatedFeatureToFeatureManagerId.contains(feature), "Unknown estimated features, this is probably a bug");
            return EstimatedFeatureToFeatureManagerId[feature];
        }

        const NCatboostOptions::TBinarizationOptions& GetBinarizationDescription(const NCB::TEstimatedFeatureId&) const {
            return QuantizedFeaturesInfo->GetFloatFeatureBinarization(Max<ui32>());
        }

        TSet<ECtrType> GetKnownSimpleCtrTypes() const;

        TVector<ui32> CreateSimpleCtrsForType(ui32 featureId,
                                              ECtrType type);

        TVector<ui32> GetAllSimpleCtrs() const;

        TVector<ui32> CreateCombinationCtrForType(ECtrType type);

        void CreateSimpleCtrs(const ui32 featureId, const TSet<NCB::TCtrConfig>& configs, TSet<ui32>* resultIds);

        TVector<NCB::TCtrConfig> CreateTreeCtrConfigs(ETaskType taskType) const;

        TMap<ECtrType, TSet<NCB::TCtrConfig>> CreateGrouppedTreeCtrConfigs() const;

        ui32 MaxTreeCtrBinFeaturesCount() const;

        TVector<ui32> GetCatFeatureIds() const;

        TVector<ui32> GetFloatFeatureIds() const;

        ui32 FeatureCount() const {
            return Cursor;
        }

        bool HasTargetBinarization() const {
            return static_cast<bool>(!GetTargetBorders().empty());
        }

        TBinarizedFeaturesManager& SetTargetBorders(TVector<float>&& borders) {
            TargetBorders = borders;
            return *this;
        }

        const TVector<float>& GetTargetBorders() const {
            return TargetBorders;
        }

        TVector<ui32> GetDataProviderFeatureIds() const;

        const NCatboostOptions::TBinarizationOptions& GetCtrBinarization(const TCtr& ctr) const {
            return GetCtrBinarizationForConfig(ctr.Configuration);
        }

        bool UseFullSetForCatFeatureStatCtrs() {
            return CatFeatureOptions.CounterCalcMethod.Get() == ECounterCalc::Full;
        }

        TMap<ECtrType, TSet<NCB::TCtrConfig>> CreateGrouppedSimpleCtrConfigs() const;

        TMap<ui32, TMap<ECtrType, TSet<NCB::TCtrConfig>>> CreateGrouppedPerFeatureCtrs() const;

        bool HasPerFeatureCtr(ui32 featureId) const {
            ui32 featureIdInPool = GetDataProviderId(featureId);
            return CatFeatureOptions.PerFeatureCtrs->contains(featureIdInPool);
        }

        TMap<ECtrType, TSet<NCB::TCtrConfig>> CreateGrouppedPerFeatureCtr(ui32 featureId) const;

        const NCatboostOptions::TCatFeatureParams& GetCatFeatureOptions() const {
            return CatFeatureOptions;
        }

        void AddCustomCtr(const TFeatureTensor& tensor, const NCatboostOptions::TCtrDescription& description) {
            UserCombinations.push_back(TUserDefinedCombination(tensor, description));
        }

        ui32 GetMaxCtrUniqueValues(const TCtr& ctr) const {
            ui32 maxCtrUniqueValues = 1 << ctr.FeatureTensor.GetSplits().size();
            for (ui32 idx: ctr.FeatureTensor.GetCatFeatures()) {
                CB_ENSURE_INTERNAL(IsCat(idx), "Unknown cat feature");
                maxCtrUniqueValues *= GetUniqueValuesCounts(idx).OnAll;
            }
            return Min(maxCtrUniqueValues, MaxObjectsCount);
        }

        ui32 GetMaxCtrUniqueValues(ui32 idx) const {
            CB_ENSURE_INTERNAL(InverseCtrs.contains(idx), "Unknown ctr idx");
            return GetMaxCtrUniqueValues(InverseCtrs[idx]);
        }

        bool UseShuffle() const {
            return EnableShuffling;
        }

    private:
        void RegisterDataProviderCatFeature(ui32 featureId) {
            CB_ENSURE(!DataProviderCatFeatureIdToFeatureManagerId.contains(featureId));
            const ui32 id = RequestNewId();
            DataProviderCatFeatureIdToFeatureManagerId[featureId] = id;
            FeatureManagerIdToDataProviderId[id] = featureId;
        }

        void RegisterDataProviderFloatFeature(ui32 featureId) {
            CB_ENSURE(!DataProviderFloatFeatureIdToFeatureManagerId.contains(featureId));
            auto internalFeatureId = QuantizedFeaturesInfo->GetFeaturesLayout()->GetInternalFeatureIdx<EFeatureType::Float>(
                featureId
            );
            if (!QuantizedFeaturesInfo->HasBorders(internalFeatureId) || QuantizedFeaturesInfo->GetBorders(internalFeatureId).empty()) {
                const ui32 id = RequestNewId();
                DataProviderFloatFeatureIdToFeatureManagerId[featureId].push_back(id);
                FeatureManagerIdToDataProviderId[id] = featureId;
                return;
            }
            auto& borders = QuantizedFeaturesInfo->GetBorders(internalFeatureId);
            for (ui32 bordersSliceStart = 0; bordersSliceStart < borders.size(); bordersSliceStart += 255) {
                const ui32 id = RequestNewId();
                DataProviderFloatFeatureIdToFeatureManagerId[featureId].push_back(id);
                FeatureManagerIdToDataProviderId[id] = featureId;
                Borders[id].assign(borders.begin() + bordersSliceStart, borders.begin() + Min<ui32>(bordersSliceStart + 255, borders.size()));
            }
        }


        void RegisterFeatureEstimators(NCB::TFeatureEstimatorsPtr estimators) {
            if (!estimators) {
                return;
            }

            estimators->ForEach(
                [&](
                    NCB::TEstimatorId estimatorId,
                    NCB::TFeatureEstimatorPtr estimator
                ) {
                    RegisterFeatureEstimator(estimatorId, estimator->FeaturesMeta());
                }
            );
        }

        void RegisterFeatureBundles() {
            if (!LearnExclusiveFeatureBundles) {
                return;
            }
            auto featuresLayout = QuantizedFeaturesInfo->GetFeaturesLayout();
            for (auto i : xrange(LearnExclusiveFeatureBundles.size())) {
                auto managerId = RequestNewId();
                FeatureManagerIdToExclusiveBundleId[managerId] = i;
                for (auto& part : LearnExclusiveFeatureBundles[i].Parts) {
                    auto dataProviderIdxForPart = featuresLayout->GetExternalFeatureIdx(part.FeatureIdx, part.FeatureType);
                    if (part.FeatureType == EFeatureType::Float) {
                        auto& floatIdxs = DataProviderFloatFeatureIdToFeatureManagerId.at(dataProviderIdxForPart);
                        CB_ENSURE_INTERNAL(floatIdxs.size() == 1, "We don't expect wide float features in bundles on GPU");
                        FeatureManagerFeaturesToBundleId[floatIdxs[0]] = i;
                    } else {
                        Y_ASSERT(part.FeatureType == EFeatureType::Categorical);
                        FeatureManagerFeaturesToBundleId[DataProviderCatFeatureIdToFeatureManagerId.at(dataProviderIdxForPart)] = i;
                    }
                }
            }
        }

        void RegisterFeatureEstimator(const NCB::TEstimatorId& estimatorId, const NCB::TEstimatedFeaturesMeta& meta) {
            for (ui32 f = 0; f < meta.FeaturesCount; ++f) {
                NCB::TEstimatedFeatureId feature{estimatorId, f};
                const ui32 maxBins = QuantizedFeaturesInfo->GetFloatFeatureBinarization(Max<ui32>()).BorderCount + 1;
                ui32 maxBinsUpperBoundHint = meta.UniqueValuesUpperBoundHint ? (*meta.UniqueValuesUpperBoundHint)[f] : maxBins;
                AddEstimatedFeature(feature, maxBinsUpperBoundHint);
            }
        }

        ui32 AddEstimatedFeature(const NCB::TEstimatedFeatureId& feature, ui32 maxBins) {
            CB_ENSURE(!EstimatedFeatureToFeatureManagerId.contains(feature));
            const ui32 id = RequestNewId();
            EstimatedFeatureToFeatureManagerId[feature] = id;
            EstimatedFeatureUpperBoundHints[id] = maxBins;
            FeatureManagerIdToEstimatedFeatureId[id] = feature;
            return id;
        }

        void CreateCtrConfigsFromDescription(const NCatboostOptions::TCtrDescription& ctrDescription,
                                             TMap<ECtrType, TSet<NCB::TCtrConfig>>* grouppedConfigs) const;

        const NCatboostOptions::TBinarizationOptions& GetCtrBinarizationForConfig(const NCB::TCtrConfig& config) const {
            CB_ENSURE(config.CtrBinarizationConfigId < CtrBinarizationOptions.size(), "error: unknown ctr binarization id " << config.CtrBinarizationConfigId);
            return CtrBinarizationOptions[config.CtrBinarizationConfigId];
        }

        //stupid line-search here is not issue :)
        inline ui32 GetOrCreateCtrBinarizationId(const NCatboostOptions::TBinarizationOptions& binarization) const;

        NCB::TCatFeatureUniqueValuesCounts GetUniqueValuesCounts(ui32 featureId) const;

        ui32 RequestNewId() {
            return Cursor++;
        }

    private:
        mutable TMap<TCtr, ui32> KnownCtrs;
        mutable TMap<ui32, TCtr> InverseCtrs;
        mutable THashSet<ui32> UsedCtrs;

        mutable TMap<ui32, TVector<ui32>> DataProviderFloatFeatureIdToFeatureManagerId;
        mutable TMap<ui32, ui32> DataProviderCatFeatureIdToFeatureManagerId;
        mutable TMap<ui32, ui32> FeatureManagerIdToDataProviderId;

        mutable TMap<NCB::TEstimatedFeatureId, ui32> EstimatedFeatureToFeatureManagerId;
        mutable TMap<ui32, ui32> EstimatedFeatureUpperBoundHints;
        mutable TMap<ui32, NCB::TEstimatedFeatureId> FeatureManagerIdToEstimatedFeatureId;

        mutable ui32 Cursor = 0;
        const ui32 MaxObjectsCount;

        mutable TVector<NCatboostOptions::TBinarizationOptions> CtrBinarizationOptions;

        TVector<float> TargetBorders;
        const NCatboostOptions::TCatFeatureParams& CatFeatureOptions;

        bool EnableShuffling = true;

        // for ctr features and float features
        THashMap<ui32, TVector<float>> Borders;

        NCB::TQuantizedFeaturesInfoPtr QuantizedFeaturesInfo;
        NCB::TFeatureEstimatorsPtr FeatureEstimators;

        struct TUserDefinedCombination {
            TFeatureTensor Tensor;
            NCatboostOptions::TCtrDescription Description;

            TUserDefinedCombination(const TFeatureTensor& tensor,
                                    const NCatboostOptions::TCtrDescription& description)
                : Tensor(tensor)
                , Description(description)
            {
            }
        };

        TVector<TUserDefinedCombination> UserCombinations;
        TSet<ui32> IgnoredFeatures;

        TVector<NCB::TExclusiveFeaturesBundle> LearnExclusiveFeatureBundles;
        TMap<ui32, ui32> FeatureManagerIdToExclusiveBundleId;
        THashMap<ui32, ui32> FeatureManagerFeaturesToBundleId;
    };
}
