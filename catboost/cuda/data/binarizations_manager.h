#pragma once

#include "feature.h"
#include "cat_feature_perfect_hash.h"
#include <catboost/libs/options/cat_feature_options.h>
#include <catboost/libs/logging/logging.h>

#include <util/generic/guid.h>

namespace NCatboostCuda {
    //stores expression for binarized features calculations and mapping from this expression to unique ids
    //WARNING: not thread-safe
    class TBinarizedFeaturesManager {
    public:
        explicit TBinarizedFeaturesManager(const NCatboostOptions::TCatFeatureParams& catFeatureOptions,
                                           const NCatboostOptions::TBinarizationOptions& floatFeaturesBinarization)
            : CatFeatureOptions(catFeatureOptions)
            , FloatFeaturesBinarization(floatFeaturesBinarization)
            , CatFeaturesPerfectHash(TStringBuilder() << "cat_feature_index." << CreateGuidAsString() << ".tmp")
        {
        }

        bool IsKnown(const ui32 featuresProviderId) const {
            return DataProviderFloatFeatureIdToFeatureManagerId.has(featuresProviderId) ||
                   DataProviderCatFeatureIdToFeatureManagerId.has(featuresProviderId);
        }

        bool IsKnown(const IFeatureValuesHolder& feature) const {
            return IsKnown(feature.GetId());
        }

        template <class TBuilder>
        const TVector<float>& GetOrCreateFloatFeatureBorders(const TFloatValuesHolder& feature,
                                                             TBuilder&& builder) {
            CB_ENSURE(IsKnown(feature));
            const ui32 featureId = GetFeatureManagerId(feature);
            if (!Borders.has(featureId)) {
                Borders[featureId] = builder(GetFloatFeatureBinarization());
            }
            return Borders[featureId];
        }

        bool HasFloatFeatureBorders(const TFloatValuesHolder& feature) const {
            return Borders.has(GetId(feature));
        }

        ENanMode GetOrCreateNanMode(const TFloatValuesHolder& feature) {
            if (!NanModes.has(feature.GetId())) {
                NanModes[feature.GetId()] = ComputeNanMode(feature.GetValuesPtr(), feature.GetSize());
            }
            return NanModes.at(feature.GetId());
        }

        ENanMode GetNanMode(const ui32 featureId) const {
            ENanMode nanMode = ENanMode::Forbidden;
            if (FeatureManagerIdToDataProviderId.has(featureId)) {
                CB_ENSURE(IsFloat(featureId));
                const ui32 dataProviderId = FeatureManagerIdToDataProviderId[featureId];
                if (NanModes.has(dataProviderId)) {
                    nanMode = NanModes.at(dataProviderId);
                }
            }
            return nanMode;
        }

        const TVector<float>& GetFloatFeatureBorders(const TFloatValuesHolder& feature) const {
            CB_ENSURE(IsKnown(feature));
            ui32 id = GetId(feature);
            return Borders.at(id);
        }

        bool HasBorders(ui32 featureId) const {
            return Borders.has(featureId);
        }

        void SetBorders(ui32 featureId, TVector<float> borders) {
            CB_ENSURE(!HasBorders(featureId));
            Borders[featureId] = std::move(borders);
        }

        bool IsFloat(ui32 featureId) const {
            if (FeatureManagerIdToDataProviderId.has(featureId)) {
                return DataProviderFloatFeatureIdToFeatureManagerId.has(FeatureManagerIdToDataProviderId.at(featureId));
            } else {
                return false;
            }
        }

        bool IsCat(ui32 featureId) const {
            if (FeatureManagerIdToDataProviderId.has(featureId)) {
                return DataProviderCatFeatureIdToFeatureManagerId.has(FeatureManagerIdToDataProviderId.at(featureId));
            } else {
                return false;
            }
        }

        TVector<ui32> GetOneHotIds(const TVector<ui32>& ids) const {
            TVector<ui32> result;
            for (auto id : ids) {
                if (IsCat(id) && UseForOneHotEncoding(id)) {
                    result.push_back(id);
                }
            }
            return result;
        }

        bool UseForOneHotEncoding(ui32 featureId) const {
            CB_ENSURE(IsCat(featureId));
            const ui32 uniqueValues = GetUniqueValues(featureId);
            return (uniqueValues > 1) && uniqueValues <= CatFeatureOptions.OneHotMaxSize;
        }

        bool UseForCtr(ui32 featureId) const {
            CB_ENSURE(IsCat(featureId));
            return GetUniqueValues(featureId) > Max<ui32>(1u, CatFeatureOptions.OneHotMaxSize);
        }

        bool UseForTreeCtr(ui32 featureId) const {
            CB_ENSURE(IsCat(featureId));
            return GetUniqueValues(featureId) > Max<ui32>(1u, CatFeatureOptions.OneHotMaxSize) &&
                   (CatFeatureOptions.MaxTensorComplexity > 1);
        }

        bool IsTreeCtrsEnabled() const {
            return !DataProviderCatFeatureIdToFeatureManagerId.empty() &&
                   (CatFeatureOptions.MaxTensorComplexity > 1);
        }

        bool UseAsBaseTensorForTreeCtr(const TFeatureTensor& tensor) const {
            return (tensor.GetComplexity() < CatFeatureOptions.MaxTensorComplexity);
        }

        bool UseForTreeCtr(const TFeatureTensor& tensor) const {
            return (tensor.GetComplexity() <= CatFeatureOptions.MaxTensorComplexity);
        }

        bool IsCtr(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor);
            return InverseCtrs.has(featureId);
        }

        bool IsTreeCtr(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor);
            return IsCtr(featureId) && !GetCtr(featureId).IsSimple();
        }

        bool IsPermutationDependent(const TCtr& ctr) const {
            if (IsPermutationDependentCtrType(ctr.Configuration.Type)) {
                return true;
            }
            return HasPermutationDependentSplit(ctr.FeatureTensor.GetSplits());
        }

        bool HasPermutationDependentSplit(const TVector<TBinarySplit>& splits) const {
            for (auto& split : splits) {
                if (IsCtr(split.FeatureId) && IsPermutationDependent(GetCtr(split.FeatureId))) {
                    return true;
                }
            }
            return false;
        }

        const TCtr& GetCtr(ui32 featureId) const {
            CB_ENSURE(featureId < Cursor);
            return InverseCtrs.at(featureId);
        }

        ui32 GetFeatureCount() const {
            return Cursor;
        }

        ui32 RegisterDataProviderCatFeature(ui32 featureId) {
            CB_ENSURE(!DataProviderCatFeatureIdToFeatureManagerId.has(featureId));
            const ui32 id = RequestNewId();
            DataProviderCatFeatureIdToFeatureManagerId[featureId] = id;
            FeatureManagerIdToDataProviderId[id] = featureId;
            CatFeaturesPerfectHash.RegisterId(id);
            return id;
        }

        ui32 RegisterDataProviderFloatFeature(ui32 featureId) {
            CB_ENSURE(!DataProviderFloatFeatureIdToFeatureManagerId.has(featureId));
            const ui32 id = RequestNewId();
            DataProviderFloatFeatureIdToFeatureManagerId[featureId] = id;
            FeatureManagerIdToDataProviderId[id] = featureId;
            return id;
        }

        bool HasFloatFeatureBordersForDataProviderFeature(const ui32 dataProviderId) {
            const ui32 featureId = GetFeatureManagerIdForFloatFeature(dataProviderId);
            return Borders.has(featureId);
        }

        ui32 SetFloatFeatureBordersForDataProviderId(ui32 dataProviderId,
                                                     TVector<float>&& borders) {
            const ui32 id = GetFeatureManagerIdForFloatFeature(dataProviderId);
            Borders[id] = std::move(borders);
            return id;
        }

        ui32 SetFloatFeatureBorders(const TFloatValuesHolder& feature,
                                    TVector<float>&& borders) {
            CB_ENSURE(IsKnown(feature));

            const ui32 id = GetFeatureManagerId(feature);
            Borders[id] = std::move(borders);
            return id;
        }

        ui32 GetFeatureManagerIdForCatFeature(ui32 dataProviderId) const {
            CB_ENSURE(DataProviderCatFeatureIdToFeatureManagerId.has(dataProviderId),
                      "Error: feature #" << dataProviderId << " is not categorical");
            return DataProviderCatFeatureIdToFeatureManagerId.at(dataProviderId);
        }

        ui32 GetFeatureManagerIdForFloatFeature(ui32 dataProviderId) const {
            CB_ENSURE(DataProviderFloatFeatureIdToFeatureManagerId.has(dataProviderId),
                      "Error: feature #" << dataProviderId << " is not categorical");
            return DataProviderFloatFeatureIdToFeatureManagerId.at(dataProviderId);
        }

        ui32 GetFeatureManagerId(const IFeatureValuesHolder& feature) const {
            switch (feature.GetType()) {
                case EFeatureValuesType::BinarizedFloat:
                case EFeatureValuesType::Float: {
                    return GetFeatureManagerIdForFloatFeature(feature.GetId());
                }
                case EFeatureValuesType::Categorical: {
                    return GetFeatureManagerIdForCatFeature(feature.GetId());
                }
                default: {
                    ythrow TCatboostException() << "Unknown feature id " << feature.GetId();
                }
            }
        }

        ui32 GetDataProviderId(ui32 featureId) const {
            return FeatureManagerIdToDataProviderId.at(featureId);
        }

        bool IsKnown(const TCtr& ctr) const {
            return KnownCtrs.has(ctr);
        }

        ui32 AddCtr(const TCtr& ctr) {
            CB_ENSURE(!KnownCtrs.has(ctr));
            const ui32 id = RequestNewId();
            KnownCtrs[ctr] = id;
            InverseCtrs[id] = ctr;
            return id;
        }

        ui32 AddCtr(const TCtr& ctr,
                    TVector<float>&& borders) {
            ui32 id = AddCtr(ctr);
            Borders[id] = std::move(borders);
            return id;
        }

        const TVector<float>& GetBorders(ui32 id) const {
            return Borders.at(id);
        }

        ui32 CtrsPerTreeCtrFeatureTensor() const {
            ui32 totalCount = 0;
            auto treeCtrConfigs = CreateGrouppedTreeCtrConfigs();
            for (const auto& treeCtr : treeCtrConfigs) {
                totalCount += treeCtr.second.size();
            }
            return totalCount;
        }

        ui32 GetBinCount(ui32 localId) const {
            if (Borders.has(localId)) {
                return Borders.at(localId).size() + 1 + (GetNanMode(localId) != ENanMode::Forbidden);
            } else if (IsCat(localId)) {
                return GetUniqueValues(localId);
            } else if (InverseCtrs.has(localId)) {
                return GetBinarizationDescription(InverseCtrs[localId]).BorderCount + 1;
            } else if (IsFloat(localId)) {
                return 0;
            } else {
                ythrow TCatboostException() << "Error: unknown feature id #" << localId;
            }
        }

        const NCatboostOptions::TBinarizationOptions& GetFloatFeatureBinarization() const {
            return FloatFeaturesBinarization;
        }

        const NCatboostOptions::TBinarizationOptions& GetTargetBinarizationDescription() const {
            return CatFeatureOptions.TargetBorders;
        }

        const NCatboostOptions::TBinarizationOptions& GetBinarizationDescription(const TCtr& ctr) const {
            return GetCtrBinarizationForConfig(ctr.Configuration);
        }

        ui32 GetId(const IFeatureValuesHolder& feature) const {
            const ui32 featureId = feature.GetId();

            if (DataProviderFloatFeatureIdToFeatureManagerId.has(featureId)) {
                return DataProviderFloatFeatureIdToFeatureManagerId[featureId];
            } else if (DataProviderCatFeatureIdToFeatureManagerId.has(featureId)) {
                return DataProviderCatFeatureIdToFeatureManagerId[featureId];
            } else {
                ythrow TCatboostException() << "Error: unknown feature with id #" << feature.GetId();
            }
        }

        ui32 GetId(const TCtr& ctr) const {
            CB_ENSURE(KnownCtrs.has(ctr));
            return KnownCtrs[ctr];
        }

        TSet<ECtrType> GetKnownSimpleCtrTypes() const {
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

        TVector<ui32> CreateSimpleCtrsForType(ui32 featureId,
                                              ECtrType type) {
            CB_ENSURE(UseForCtr(featureId));
            TSet<ui32> resultIds;

            if (HasPerFeatureCtr(featureId)) {
                auto perFeatureCtrs = CreateGrouppedPerFeatureCtr(featureId);
                if (perFeatureCtrs.has(type)) {
                    CreateSimpleCtrs(featureId, perFeatureCtrs.at(type), &resultIds);
                }
            } else {
                auto simpleCtrConfigs = CreateGrouppedSimpleCtrConfigs();
                CB_ENSURE(simpleCtrConfigs.at(type), "Simple ctr type is not enabled " << type);
                CreateSimpleCtrs(featureId, simpleCtrConfigs.at(type), &resultIds);
            }

            return TVector<ui32>(resultIds.begin(), resultIds.end());
        }

        TVector<ui32> GetAllSimpleCtrs() const {
            TVector<ui32> ids;
            for (auto& knownCtr : KnownCtrs) {
                const TCtr& ctr = knownCtr.first;
                if (ctr.IsSimple()) {
                    ids.push_back(knownCtr.second);
                }
            }
            return ids;
        }

        TVector<ui32> CreateCombinationCtrForType(ECtrType type) {
            TSet<ui32> resultIds;

            for (auto& combination : UserCombinations) {
                if (combination.Description.Type != type) {
                    continue;
                }
                TMap<ECtrType, TSet<TCtrConfig>> configs;
                CreateCtrConfigsFromDescription(combination.Description, &configs);
                if (configs.has(type)) {
                    for (auto& ctrConfig : configs[type]) {
                        TCtr ctr;
                        ctr.FeatureTensor = combination.Tensor;
                        ctr.Configuration = ctrConfig;

                        if (!KnownCtrs.has(ctr)) {
                            AddCtr(ctr);
                        }
                        resultIds.insert(GetId(ctr));
                    }
                }
            }
            return TVector<ui32>(resultIds.begin(), resultIds.end());
        }

        void CreateSimpleCtrs(const ui32 featureId, const TSet<TCtrConfig>& configs, TSet<ui32>* resultIds) {
            for (const auto& ctrConfig : configs) {
                TCtr ctr;
                ctr.FeatureTensor.AddCatFeature(featureId);
                ctr.Configuration = ctrConfig;

                if (!KnownCtrs.has(ctr)) {
                    AddCtr(ctr);
                }
                resultIds->insert(GetId(ctr));
            }
        }

        TVector<TCtrConfig> CreateTreeCtrConfigs() const {
            TVector<TCtrConfig> result;
            auto treeCtrConfigs = CreateGrouppedTreeCtrConfigs();

            for (const auto& ctrConfigs : treeCtrConfigs) {
                auto ctrType = ctrConfigs.first;
                CB_ENSURE(IsSupportedCtrType(ctrType));
                for (const auto& ctrConfig : ctrConfigs.second) {
                    result.push_back(ctrConfig);
                }
            }
            return result;
        }

        TMap<ECtrType, TSet<TCtrConfig>> CreateGrouppedTreeCtrConfigs() const {
            TMap<ECtrType, TSet<TCtrConfig>> treeCtrConfigs;
            for (const auto& treeCtr : CatFeatureOptions.CombinationCtrs.Get()) {
                CreateCtrConfigsFromDescription(treeCtr, &treeCtrConfigs);
            }
            return treeCtrConfigs;
        }

        ui32 MaxTreeCtrBinFeaturesCount() const {
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

        TVector<ui32> GetCatFeatureIds() const {
            TVector<ui32> featureIds;

            for (const auto& feature : DataProviderCatFeatureIdToFeatureManagerId) {
                if (GetBinCount(feature.second)) {
                    featureIds.push_back(feature.second);
                }
            }
            Sort(featureIds.begin(), featureIds.end());
            return featureIds;
        }

        TVector<ui32> GetFloatFeatureIds() const {
            TVector<ui32> featureIds;

            for (const auto& feature : DataProviderFloatFeatureIdToFeatureManagerId) {
                if (GetBinCount(feature.second) > 1) {
                    featureIds.push_back(feature.second);
                }
            }
            return featureIds;
        }

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

        TVector<ui32> GetDataProviderFeatureIds() const {
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

        const NCatboostOptions::TBinarizationOptions& GetCtrBinarization(const TCtr& ctr) const {
            return GetCtrBinarizationForConfig(ctr.Configuration);
        }

        //store perfect hash by featureManager id
        const TMap<int, ui32>& GetCategoricalFeaturesPerfectHash(const ui32 featureId) const {
            CB_ENSURE(CatFeaturesPerfectHash.HasFeature(featureId));
            return CatFeaturesPerfectHash.GetFeatureIndex(featureId);
        };

        void UnloadCatFeaturePerfectHashFromRam() const {
            CatFeaturesPerfectHash.FreeRam();
        }

        bool UseFullSetForCatFeatureStatCtrs() {
            return CatFeatureOptions.CounterCalcMethod.Get() == ECounterCalc::Full;
        }

        TMap<ECtrType, TSet<TCtrConfig>> CreateGrouppedSimpleCtrConfigs() const {
            TMap<ECtrType, TSet<TCtrConfig>> simpleCtrs;

            for (const auto& simpleCtr : CatFeatureOptions.SimpleCtrs.Get()) {
                CreateCtrConfigsFromDescription(simpleCtr, &simpleCtrs);
            }
            return simpleCtrs;
        };

        TMap<ui32, TMap<ECtrType, TSet<TCtrConfig>>> CreateGrouppedPerFeatureCtrs() const {
            TMap<ui32, TMap<ECtrType, TSet<TCtrConfig>>> perFeatureCtrs;

            for (const auto& perFeatureCtr : CatFeatureOptions.PerFeatureCtrs.Get()) {
                CB_ENSURE(DataProviderCatFeatureIdToFeatureManagerId.has(perFeatureCtr.first),
                          "Error: Feature with id #" << perFeatureCtr.first << " is not categorical. Can't create ctr");
                const ui32 featureId = DataProviderCatFeatureIdToFeatureManagerId[perFeatureCtr.first];
                for (const auto& ctrDescription : perFeatureCtr.second) {
                    CreateCtrConfigsFromDescription(ctrDescription, &perFeatureCtrs[featureId]);
                }
            }
            return perFeatureCtrs;
        };

        bool HasPerFeatureCtr(ui32 featureId) const {
            ui32 featureIdInPool = GetDataProviderId(featureId);
            return CatFeatureOptions.PerFeatureCtrs->has(featureIdInPool);
        }

        TMap<ECtrType, TSet<TCtrConfig>> CreateGrouppedPerFeatureCtr(ui32 featureId) const {
            CB_ENSURE(IsCat(featureId), "Feature #" << featureId << " is not categorical. Can't create per feature CTRs");
            ui32 featureIdInPool = GetDataProviderId(featureId);
            CB_ENSURE(CatFeatureOptions.PerFeatureCtrs->has(featureIdInPool), "No perFeatureCtr for feature #" << featureIdInPool << " was found");
            TMap<ECtrType, TSet<TCtrConfig>> perFeatureCtr;
            for (const auto& ctrDescription : CatFeatureOptions.PerFeatureCtrs->at(featureIdInPool)) {
                CreateCtrConfigsFromDescription(ctrDescription, &perFeatureCtr);
            }
            return perFeatureCtr;
        };

        const NCatboostOptions::TCatFeatureParams& GetCatFeatureOptions() const {
            return CatFeatureOptions;
        }

        void AddCustomCtr(const TFeatureTensor& tensor, const NCatboostOptions::TCtrDescription& description) {
            UserCombinations.push_back(TUserDefinedCombination(tensor, description));
        }

    private:
        void CreateCtrConfigsFromDescription(const NCatboostOptions::TCtrDescription& ctrDescription,
                                             TMap<ECtrType, TSet<TCtrConfig>>* grouppedConfigs) const {
            for (const auto& prior : ctrDescription.GetPriors()) {
                CB_ENSURE(!TargetBorders.empty(), "Enable ctr description should be done after target borders are set");
                CB_ENSURE(ctrDescription.GetPriors().size(), "Set priors first");

                ECtrType type = ctrDescription.Type;
                TCtrConfig defaultConfig;

                defaultConfig.Prior = prior;
                defaultConfig.Type = type;
                defaultConfig.CtrBinarizationConfigId = GetOrCreateCtrBinarizationId(ctrDescription.CtrBinarization);
                CB_ENSURE(prior.size() == 2, "Error: currently priors are num and denum biases. Need 2 params in option");

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

        const NCatboostOptions::TBinarizationOptions& GetCtrBinarizationForConfig(const TCtrConfig& config) const {
            CB_ENSURE(config.CtrBinarizationConfigId < CtrBinarizationOptions.size(), "error: unknown ctr binarization id " << config.CtrBinarizationConfigId);
            return CtrBinarizationOptions[config.CtrBinarizationConfigId];
        }

        //stupid line-search here is not issue :)
        inline ui32 GetOrCreateCtrBinarizationId(const NCatboostOptions::TBinarizationOptions& binarization) const {
            for (ui32 i = 0; i < CtrBinarizationOptions.size(); ++i) {
                if (CtrBinarizationOptions[i] == binarization) {
                    return i;
                }
            }
            ui32 id = CtrBinarizationOptions.size();
            CtrBinarizationOptions.push_back(binarization);
            return id;
        }

        ui32 GetUniqueValues(ui32 featureId) const {
            CB_ENSURE(IsCat(featureId));
            return CatFeaturesPerfectHash.GetUniqueValues(featureId);
        }

        ui32 RequestNewId() {
            return Cursor++;
        }

        TBinarizedFeaturesManager& SetBinarization(ui64 featureId,
                                                   TVector<float>&& borders) {
            Borders[featureId] = std::move(borders);
            return *this;
        }

        friend class TCatFeaturesPerfectHashHelper;

        inline ENanMode ComputeNanMode(const float* values, ui32 size) const {
            if (FloatFeaturesBinarization.NanMode == ENanMode::Forbidden) {
                return ENanMode::Forbidden;
            }
            bool hasNans = false;
            for (ui32 i = 0; i < size; ++i) {
                if (IsNan(values[i])) {
                    hasNans = true;
                    break;
                }
            }
            if (hasNans) {
                return FloatFeaturesBinarization.NanMode;
            }
            return ENanMode::Forbidden;
        }

    private:
        mutable TMap<TCtr, ui32> KnownCtrs;
        mutable TMap<ui32, TCtr> InverseCtrs;

        mutable TMap<ui32, ui32> DataProviderFloatFeatureIdToFeatureManagerId;
        mutable TMap<ui32, ui32> DataProviderCatFeatureIdToFeatureManagerId;
        mutable TMap<ui32, ui32> FeatureManagerIdToDataProviderId;

        mutable ui32 Cursor = 0;

        mutable TVector<NCatboostOptions::TBinarizationOptions> CtrBinarizationOptions;

        TVector<float> TargetBorders;

        const NCatboostOptions::TCatFeatureParams& CatFeatureOptions;
        const NCatboostOptions::TBinarizationOptions& FloatFeaturesBinarization;

        //float and ctr features
        TMap<ui32, TVector<float>> Borders;
        TMap<ui32, ENanMode> NanModes;
        TCatFeaturesPerfectHash CatFeaturesPerfectHash;

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
    };
}
