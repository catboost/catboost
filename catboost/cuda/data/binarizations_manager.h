#pragma once

#include "feature.h"

#include <catboost/private/libs/ctr_description/ctr_config.h>
#include <catboost/libs/data/cat_feature_perfect_hash.h>
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
                                  NCB::TQuantizedFeaturesInfoPtr quantizedFeaturesInfo);

        TBinarizedFeaturesManager(const TBinarizedFeaturesManager& featureManager, const TVector<ui32>& ignoredFeatureIds);

        ENanMode GetNanMode(const ui32 featureId) const;

        bool HasBorders(ui32 featureId) const;

        void SetBorders(ui32 featureId, TVector<float> borders) {
            CB_ENSURE(!HasBorders(featureId));
            Borders[featureId] = std::move(borders);
        }

        bool IsFloat(ui32 featureId) const;

        bool IsCat(ui32 featureId) const;

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
            CB_ENSURE(featureId < Cursor);
            return InverseCtrs.contains(featureId);
        }

        bool IsTreeCtr(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor);
            return IsCtr(featureId) && !GetCtr(featureId).IsSimple();
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

        ui32 GetFeatureManagerIdForCatFeature(ui32 dataProviderId) const;

        ui32 GetFeatureManagerIdForFloatFeature(ui32 dataProviderId) const;

        ui32 GetDataProviderId(ui32 featureId) const {
            return FeatureManagerIdToDataProviderId.at(featureId);
        }

        TEstimatedFeature GetEstimatedFeature(ui32 featureId) const {
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

        TVector<ui32> GetEstimatedFeatureIds() const;

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

        ui32 GetId(const TEstimatedFeature& feature) const {
            CB_ENSURE(EstimatedFeatureToFeatureManagerId.contains(feature), "Unknown estimated features, this is probably a bug");
            return EstimatedFeatureToFeatureManagerId[feature];
        }

        const NCatboostOptions::TBinarizationOptions& GetBinarizationDescription(const TEstimatedFeature&) const {
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


    private:
        ui32 RegisterDataProviderCatFeature(ui32 featureId) {
            CB_ENSURE(!DataProviderCatFeatureIdToFeatureManagerId.contains(featureId));
            const ui32 id = RequestNewId();
            DataProviderCatFeatureIdToFeatureManagerId[featureId] = id;
            FeatureManagerIdToDataProviderId[id] = featureId;
            return id;
        }

        ui32 RegisterDataProviderFloatFeature(ui32 featureId) {
            CB_ENSURE(!DataProviderFloatFeatureIdToFeatureManagerId.contains(featureId));
            const ui32 id = RequestNewId();
            DataProviderFloatFeatureIdToFeatureManagerId[featureId] = id;
            FeatureManagerIdToDataProviderId[id] = featureId;
            return id;
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

        void RegisterFeatureEstimator(const NCB::TEstimatorId& estimatorId, const NCB::TEstimatedFeaturesMeta& meta) {
            for (ui32 f = 0; f < meta.FeaturesCount; ++f) {
                TEstimatedFeature feature{estimatorId, f};
                const ui32 maxBins = QuantizedFeaturesInfo->GetFloatFeatureBinarization(Max<ui32>()).BorderCount + 1;
                ui32 maxBinsUpperBoundHint = meta.UniqueValuesUpperBoundHint ? (*meta.UniqueValuesUpperBoundHint)[f] : maxBins;
                AddEstimatedFeature(feature, maxBinsUpperBoundHint);
            }
        }

        ui32 AddEstimatedFeature(const TEstimatedFeature& feature, ui32 maxBins) {
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

        mutable TMap<ui32, ui32> DataProviderFloatFeatureIdToFeatureManagerId;
        mutable TMap<ui32, ui32> DataProviderCatFeatureIdToFeatureManagerId;
        mutable TMap<ui32, ui32> FeatureManagerIdToDataProviderId;

        mutable TMap<TEstimatedFeature, ui32> EstimatedFeatureToFeatureManagerId;
        mutable TMap<ui32, ui32> EstimatedFeatureUpperBoundHints;
        mutable TMap<ui32, TEstimatedFeature> FeatureManagerIdToEstimatedFeatureId;

        mutable ui32 Cursor = 0;

        mutable TVector<NCatboostOptions::TBinarizationOptions> CtrBinarizationOptions;

        TVector<float> TargetBorders;
        const NCatboostOptions::TCatFeatureParams& CatFeatureOptions;


        // for ctr features, for float - get from QuantizedFeaturesInfo
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

    };
}
