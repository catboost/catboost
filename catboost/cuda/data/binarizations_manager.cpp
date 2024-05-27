#include "binarizations_manager.h"

#include <catboost/private/libs/ctr_description/ctr_type.h>
#include <catboost/libs/data/feature_estimators.h>
#include <catboost/private/libs/options/restrictions.h>

#include <util/generic/algorithm.h>
#include <util/generic/xrange.h>
#include <util/system/compiler.h>

using namespace NCB;

namespace NCatboostCuda {
    TBinarizedFeaturesManager::TBinarizedFeaturesManager(
        const NCatboostOptions::TCatFeatureParams& catFeatureOptions,
        TFeatureEstimatorsPtr estimators,
        const TFeaturesLayout& featuresLayout,
        const TVector<NCB::TExclusiveFeaturesBundle>& learnExclusiveFeatureBundles,
        TQuantizedFeaturesInfoPtr quantizedFeaturesInfo,
        ui32 maxObjectsCount,
        bool enableShuffling)
        : MaxObjectsCount(maxObjectsCount)
        , CatFeatureOptions(catFeatureOptions)
        , EnableShuffling(enableShuffling)
        , QuantizedFeaturesInfo(quantizedFeaturesInfo)
        , FeatureEstimators(estimators)
        , LearnExclusiveFeatureBundles(learnExclusiveFeatureBundles)
    {
        const auto& featuresMetaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

        for (auto featureIdx : xrange(featuresMetaInfo.size())) {
            const auto& featureMetaInfo = featuresMetaInfo[featureIdx];
            if (featureMetaInfo.Type == EFeatureType::Float) {
                RegisterDataProviderFloatFeature(featureIdx);
            } else if (featureMetaInfo.Type == EFeatureType::Categorical) {
                RegisterDataProviderCatFeature(featureIdx);
            }
            if (featureMetaInfo.IsIgnored) {
                IgnoredFeatures.insert(featureIdx);
            }
        }

        RegisterFeatureEstimators(FeatureEstimators);
        RegisterFeatureBundles();
    }

    TBinarizedFeaturesManager::TBinarizedFeaturesManager(
        const TBinarizedFeaturesManager& featureManager,
        const TVector<ui32>& ignoredFeatureIds)
        : KnownCtrs(featureManager.KnownCtrs)
        , InverseCtrs(featureManager.InverseCtrs)
        , UsedCtrs(featureManager.UsedCtrs)
        , DataProviderFloatFeatureIdToFeatureManagerId(featureManager.DataProviderFloatFeatureIdToFeatureManagerId)
        , DataProviderCatFeatureIdToFeatureManagerId(featureManager.DataProviderCatFeatureIdToFeatureManagerId)
        , FeatureManagerIdToDataProviderId(featureManager.FeatureManagerIdToDataProviderId)
        , EstimatedFeatureToFeatureManagerId(featureManager.EstimatedFeatureToFeatureManagerId)
        , EstimatedFeatureUpperBoundHints(featureManager.EstimatedFeatureUpperBoundHints)
        , FeatureManagerIdToEstimatedFeatureId(featureManager.FeatureManagerIdToEstimatedFeatureId)
        , Cursor(featureManager.Cursor)
        , MaxObjectsCount(featureManager.MaxObjectsCount)
        , CtrBinarizationOptions(featureManager.CtrBinarizationOptions)
        , TargetBorders(featureManager.TargetBorders)
        , CatFeatureOptions(featureManager.CatFeatureOptions)
        , Borders(featureManager.Borders)
        , QuantizedFeaturesInfo(featureManager.QuantizedFeaturesInfo)
        , FeatureEstimators(featureManager.FeatureEstimators)
        , UserCombinations(featureManager.UserCombinations)
        , IgnoredFeatures(ignoredFeatureIds.begin(), ignoredFeatureIds.end()) {
    }

    ENanMode TBinarizedFeaturesManager::GetNanMode(const ui32 featureId) const {
        ENanMode nanMode = ENanMode::Forbidden;
        if (IsFloat(featureId)) {
            return QuantizedFeaturesInfo->GetNanMode(
                QuantizedFeaturesInfo->GetFeaturesLayout()->GetInternalFeatureIdx<EFeatureType::Float>(
                    FeatureManagerIdToDataProviderId[featureId]));
        }
        return nanMode;
    }

    bool TBinarizedFeaturesManager::HasBorders(ui32 featureId) const {
        return Borders.contains(featureId);
    }

    bool TBinarizedFeaturesManager::IsEstimated(ui32 featureId) const {
        return FeatureManagerIdToEstimatedFeatureId.contains(featureId);
    }


    bool TBinarizedFeaturesManager::IsFloat(ui32 featureId) const {
        if (FeatureManagerIdToDataProviderId.contains(featureId)) {
            return DataProviderFloatFeatureIdToFeatureManagerId.contains(FeatureManagerIdToDataProviderId.at(featureId));
        } else {
            return false;
        }
    }

    bool TBinarizedFeaturesManager::IsCat(ui32 featureId) const {
        if (FeatureManagerIdToDataProviderId.contains(featureId)) {
            return DataProviderCatFeatureIdToFeatureManagerId.contains(FeatureManagerIdToDataProviderId.at(featureId));
        } else {
            return false;
        }
    }

    bool TBinarizedFeaturesManager::UseForOneHotEncoding(ui32 featureId) const {
        auto uniqValuesOnLearn = GetUniqueValuesCounts(featureId).OnAll; // match for OnAll in TBinarizationInfoProvider::GetFoldsCount
        return (uniqValuesOnLearn > 1) && (uniqValuesOnLearn <= CatFeatureOptions.OneHotMaxSize);
    }

    bool TBinarizedFeaturesManager::UseForCtr(ui32 featureId) const {
        if (IsEstimatedFeature(featureId)) {
            return false;
        }
        return !UseForOneHotEncoding(featureId);
    }

    bool TBinarizedFeaturesManager::UseForTreeCtr(ui32 featureId) const {
        if (IsEstimatedFeature(featureId)) {
            return false;
        }
        return UseForCtr(featureId) && (CatFeatureOptions.MaxTensorComplexity > 1);
    }

    bool TBinarizedFeaturesManager::UseForTreeCtr(const TFeatureTensor& tensor) const  {
        for (auto split : tensor.GetSplits()) {
            if (IsEstimatedFeature(split.FeatureId)) {
                return false;
            }
        }
        return (tensor.GetComplexity() <= CatFeatureOptions.MaxTensorComplexity);
    }

    bool TBinarizedFeaturesManager::IsPermutationDependent(const TCtr& ctr) const {
        if (IsPermutationDependentCtrType(ctr.Configuration.Type)) {
            return true;
        }
        return HasPermutationDependentSplit(ctr.FeatureTensor.GetSplits());
    }

    bool TBinarizedFeaturesManager::HasPermutationDependentSplit(const TVector<TBinarySplit>& splits) const {
        for (auto& split : splits) {
            if (IsCtr(split.FeatureId) && IsPermutationDependent(GetCtr(split.FeatureId))) {
                return true;
            }
        }
        return false;
    }

    ui32 TBinarizedFeaturesManager::GetFeatureManagerIdForCatFeature(ui32 dataProviderId) const {
        CB_ENSURE(DataProviderCatFeatureIdToFeatureManagerId.contains(dataProviderId),
                  "Error: feature #" << dataProviderId << " is not categorical");
        return DataProviderCatFeatureIdToFeatureManagerId.at(dataProviderId);
    }

    const TVector<ui32>& TBinarizedFeaturesManager::GetFeatureManagerIdForFloatFeature(ui32 dataProviderId) const {
        CB_ENSURE(DataProviderFloatFeatureIdToFeatureManagerId.contains(dataProviderId),
                  "Error: feature #" << dataProviderId << " is not float");
        return DataProviderFloatFeatureIdToFeatureManagerId.at(dataProviderId);
    }

    const TVector<float>& TBinarizedFeaturesManager::GetBorders(ui32 featureId) const {
        CB_ENSURE(Borders.contains(featureId), "Can't find borders for feature #" << featureId);
        return Borders.at(featureId);
    }

    ui32 TBinarizedFeaturesManager::CtrsPerTreeCtrFeatureTensor() const {
        ui32 totalCount = 0;
        auto treeCtrConfigs = CreateGrouppedTreeCtrConfigs();
        for (const auto& treeCtr : treeCtrConfigs) {
            totalCount += treeCtr.second.size();
        }
        return totalCount;
    }

    ui32 TBinarizedFeaturesManager::GetBinCount(ui32 localId) const {
        if (HasBorders(localId)) {
            return GetBorders(localId).size() + 1;
        } else if (IsCat(localId)) {
            return GetUniqueValuesCounts(localId).OnAll;
        } else if (InverseCtrs.contains(localId)) {
            return GetBinarizationDescription(InverseCtrs[localId]).BorderCount + 1;
        } else if (IsFloat(localId)) {
            return 0;
        } else if (IsEstimatedFeature(localId)) {
            return EstimatedFeatureUpperBoundHints.at(localId);
        } else if (IsFeatureBundle(localId)) {
            return LearnExclusiveFeatureBundles.at(FeatureManagerIdToExclusiveBundleId.at(localId)).GetBinCount();
        } else {
            ythrow TCatBoostException() << "Error: unknown feature id #" << localId;
        }
    }

    ui32 TBinarizedFeaturesManager::AddCtr(const TCtr& ctr) {
        CB_ENSURE(!KnownCtrs.contains(ctr));
        const ui32 id = RequestNewId();
        KnownCtrs[ctr] = id;
        InverseCtrs[id] = ctr;
        return id;
    }

    TSet<ECtrType> TBinarizedFeaturesManager::GetKnownSimpleCtrTypes() const {
        TSet<ECtrType> result;
        auto simpleCtrs = CreateGrouppedSimpleCtrConfigs();
        for (const auto& simpleCtr : simpleCtrs) {
            result.insert(simpleCtr.first);
        }
        auto perFeatureCtrs = CreateGrouppedPerFeatureCtrs();
        for (const auto& perFeatureCtr : perFeatureCtrs) {
            for (const auto& grouppedCtr : perFeatureCtr.second) {
                result.insert(grouppedCtr.first);
            }
        }
        return result;
    }

    TVector<ui32> TBinarizedFeaturesManager::CreateSimpleCtrsForType(ui32 featureId, ECtrType type) {
        CB_ENSURE(UseForCtr(featureId));
        TSet<ui32> resultIds;

        if (HasPerFeatureCtr(featureId)) {
            auto perFeatureCtrs = CreateGrouppedPerFeatureCtr(featureId);
            if (perFeatureCtrs.contains(type)) {
                CreateSimpleCtrs(featureId, perFeatureCtrs.at(type), &resultIds);
            }
        } else {
            auto simpleCtrConfigs = CreateGrouppedSimpleCtrConfigs();
            CB_ENSURE(simpleCtrConfigs.at(type), "Simple ctr type is not enabled " << type);
            CreateSimpleCtrs(featureId, simpleCtrConfigs.at(type), &resultIds);
        }

        return TVector<ui32>(resultIds.begin(), resultIds.end());
    }

    TVector<ui32> TBinarizedFeaturesManager::GetAllSimpleCtrs() const {
        TVector<ui32> ids;
        for (auto& knownCtr : KnownCtrs) {
            const TCtr& ctr = knownCtr.first;
            if (ctr.IsSimple()) {
                ids.push_back(knownCtr.second);
            }
        }
        return ids;
    }

    TVector<ui32> TBinarizedFeaturesManager::CreateCombinationCtrForType(ECtrType type) {
        TSet<ui32> resultIds;

        for (auto& combination : UserCombinations) {
            if (combination.Description.Type != type) {
                continue;
            }
            TMap<ECtrType, TSet<TCtrConfig>> configs;
            CreateCtrConfigsFromDescription(combination.Description, &configs);
            if (configs.contains(type)) {
                for (auto& ctrConfig : configs[type]) {
                    TCtr ctr;
                    ctr.FeatureTensor = combination.Tensor;
                    ctr.Configuration = ctrConfig;

                    if (!KnownCtrs.contains(ctr)) {
                        AddCtr(ctr);
                    }
                    resultIds.insert(GetId(ctr));
                }
            }
        }
        return TVector<ui32>(resultIds.begin(), resultIds.end());
    }

    void TBinarizedFeaturesManager::CreateSimpleCtrs(const ui32 featureId, const TSet<TCtrConfig>& configs,
                                                     TSet<ui32>* resultIds) {
        for (const auto& ctrConfig : configs) {
            TCtr ctr;
            ctr.FeatureTensor.AddCatFeature(featureId);
            ctr.Configuration = ctrConfig;

            if (!KnownCtrs.contains(ctr)) {
                AddCtr(ctr);
            }
            resultIds->insert(GetId(ctr));
        }
    }

    TVector<TCtrConfig> TBinarizedFeaturesManager::CreateTreeCtrConfigs(ETaskType taskType) const {
        TVector<TCtrConfig> result;
        auto treeCtrConfigs = CreateGrouppedTreeCtrConfigs();

        for (const auto& ctrConfigs : treeCtrConfigs) {
            auto ctrType = ctrConfigs.first;
            CB_ENSURE(IsSupportedCtrType(taskType, ctrType));
            for (const auto& ctrConfig : ctrConfigs.second) {
                result.push_back(ctrConfig);
            }
        }
        return result;
    }

    TMap<ECtrType, TSet<TCtrConfig>> TBinarizedFeaturesManager::CreateGrouppedTreeCtrConfigs() const {
        TMap<ECtrType, TSet<TCtrConfig>> treeCtrConfigs;
        for (const auto& treeCtr : CatFeatureOptions.CombinationCtrs.Get()) {
            CreateCtrConfigsFromDescription(treeCtr, &treeCtrConfigs);
        }
        return treeCtrConfigs;
    }

    ui32 TBinarizedFeaturesManager::MaxTreeCtrBinFeaturesCount() const {
        ui32 total = 0;
        auto treeCtrConfigs = CreateGrouppedTreeCtrConfigs();
        for (const auto& treeCtrConfig : treeCtrConfigs) {
            for (auto& config : treeCtrConfig.second) {
                ui32 binarization = GetCtrBinarizationForConfig(config).BorderCount + 1;
                total += binarization;
            }
        }
        return total;
    }

    TVector<ui32> TBinarizedFeaturesManager::GetCatFeatureIds() const {
        const auto& featuresLayout = *QuantizedFeaturesInfo->GetFeaturesLayout();
        const auto& metaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

        TVector<ui32> featureIds;

        for (const auto& feature : DataProviderCatFeatureIdToFeatureManagerId) {
            if (metaInfo[feature.first].IsAvailable && !IgnoredFeatures.contains(feature.first)) {
                featureIds.push_back(feature.second);
            }
        }
        Sort(featureIds.begin(), featureIds.end());
        return featureIds;
    }

    TVector<ui32> TBinarizedFeaturesManager::GetFloatFeatureIds() const {
        const auto& featuresLayout = *QuantizedFeaturesInfo->GetFeaturesLayout();
        const auto& metaInfo = featuresLayout.GetExternalFeaturesMetaInfo();

        TVector<ui32> featureIds;

        for (const auto& feature : DataProviderFloatFeatureIdToFeatureManagerId) {
            if (metaInfo[feature.first].IsAvailable && !IgnoredFeatures.contains(feature.first)) {
                featureIds.insert(featureIds.end(), feature.second.begin(), feature.second.end());
            }
        }
        return featureIds;
    }

    TVector<ui32> TBinarizedFeaturesManager::GetDataProviderFeatureIds() const {
        TVector<ui32> features;
        for (auto id : GetFloatFeatureIds()) {
            Y_ASSERT(GetBinCount(id));
            features.push_back(id);
        }
        for (auto id : GetCatFeatureIds()) {
            Y_ASSERT(GetBinCount(id));
            features.push_back(id);
        }
        return features;
    }

    TMap<ECtrType, TSet<TCtrConfig>> TBinarizedFeaturesManager::CreateGrouppedSimpleCtrConfigs() const {
        TMap<ECtrType, TSet<TCtrConfig>> simpleCtrs;

        for (const auto& simpleCtr : CatFeatureOptions.SimpleCtrs.Get()) {
            CreateCtrConfigsFromDescription(simpleCtr, &simpleCtrs);
        }
        return simpleCtrs;
    }

    TMap<ui32, TMap<ECtrType, TSet<TCtrConfig>>> TBinarizedFeaturesManager::CreateGrouppedPerFeatureCtrs() const {
        TMap<ui32, TMap<ECtrType, TSet<TCtrConfig>>> perFeatureCtrs;

        for (const auto& perFeatureCtr : CatFeatureOptions.PerFeatureCtrs.Get()) {
            CB_ENSURE(DataProviderCatFeatureIdToFeatureManagerId.contains(perFeatureCtr.first),
                      "Error: Feature with id #" << perFeatureCtr.first << " is not categorical. Can't create ctr");
            const ui32 featureId = DataProviderCatFeatureIdToFeatureManagerId[perFeatureCtr.first];
            for (const auto& ctrDescription : perFeatureCtr.second) {
                CreateCtrConfigsFromDescription(ctrDescription, &perFeatureCtrs[featureId]);
            }
        }
        return perFeatureCtrs;
    }

    TMap<ECtrType, TSet<TCtrConfig>> TBinarizedFeaturesManager::CreateGrouppedPerFeatureCtr(ui32 featureId) const {
        CB_ENSURE(IsCat(featureId), "Feature #" << featureId << " is not categorical. Can't create per feature CTRs");
        ui32 featureIdInPool = GetDataProviderId(featureId);
        CB_ENSURE(CatFeatureOptions.PerFeatureCtrs->contains(featureIdInPool), "No perFeatureCtr for feature #" << featureIdInPool << " was found");
        TMap<ECtrType, TSet<TCtrConfig>> perFeatureCtr;
        for (const auto& ctrDescription : CatFeatureOptions.PerFeatureCtrs->at(featureIdInPool)) {
            CreateCtrConfigsFromDescription(ctrDescription, &perFeatureCtr);
        }
        return perFeatureCtr;
    }

    void TBinarizedFeaturesManager::CreateCtrConfigsFromDescription(const NCatboostOptions::TCtrDescription& ctrDescription,
                                                                    TMap<ECtrType, TSet<TCtrConfig>>* grouppedConfigs) const {
        for (const auto& prior : ctrDescription.GetPriors()) {
            ECtrType type = ctrDescription.Type;
            if ((type != ECtrType::Counter) && !HasTargetBinarization()) {
                continue;
            }

            CB_ENSURE(ctrDescription.GetPriors().size(), "Set priors first");

            TCtrConfig defaultConfig;

            defaultConfig.Prior = prior;
            if (defaultConfig.Prior.size() == 1) {
                defaultConfig.Prior.push_back(1);
            }
            defaultConfig.Type = type;
            defaultConfig.CtrBinarizationConfigId = GetOrCreateCtrBinarizationId(ctrDescription.CtrBinarization);
            CB_ENSURE(defaultConfig.Prior.size() == 2, "Error: currently priors are num and denum biases. Need 2 params in option, got " << prior.size());

            if (type == ECtrType::Buckets || type == ECtrType::Borders) {
                ui32 numBins = (type == ECtrType::Buckets)
                                   ? TargetBorders.size() + 1
                                   : TargetBorders.size();
                const ui32 numCtrs = numBins;

                for (ui32 i = 0; i < numCtrs; ++i) {
                    //don't calc 0-class ctr for binary classification, it's unneeded
                    if (i == 0 && numBins == 2 && type == ECtrType::Buckets) {
                        continue;
                    }
                    TCtrConfig config = defaultConfig;
                    config.ParamId = i;
                    (*grouppedConfigs)[type].insert(config);
                }
            } else {
                (*grouppedConfigs)[type].insert(defaultConfig);
            }
        }
    }

    ui32 TBinarizedFeaturesManager::GetOrCreateCtrBinarizationId(
        const NCatboostOptions::TBinarizationOptions& binarization) const {
        for (ui32 i = 0; i < CtrBinarizationOptions.size(); ++i) {
            if (CtrBinarizationOptions[i] == binarization) {
                return i;
            }
        }
        ui32 id = CtrBinarizationOptions.size();
        CtrBinarizationOptions.push_back(binarization);
        return id;
    }

    TCatFeatureUniqueValuesCounts TBinarizedFeaturesManager::GetUniqueValuesCounts(ui32 featureId) const {
        CB_ENSURE(IsCat(featureId));
        return QuantizedFeaturesInfo->GetUniqueValuesCounts(
            QuantizedFeaturesInfo->GetFeaturesLayout()->GetInternalFeatureIdx<EFeatureType::Categorical>(
                FeatureManagerIdToDataProviderId[featureId]));
    }
    TVector<ui32> TBinarizedFeaturesManager::GetEstimatedFeatureIds() const {
        TVector<ui32> result;
        for (const auto& [estimatedFeature, id] : EstimatedFeatureToFeatureManagerId) {
            Y_UNUSED(estimatedFeature);
            result.push_back(id);
        }
        return result;
    }
}
