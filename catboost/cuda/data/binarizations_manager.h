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

        bool IsKnown(const ui32 featuresProviderId) const;

        bool IsKnown(const IFeatureValuesHolder& feature) const;

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

        void SetOrCheckNanMode(const IFeatureValuesHolder& feature,
                               ENanMode nanMode);

        ENanMode GetOrComputeNanMode(const TFloatValuesHolder& feature);

        ENanMode GetNanMode(const ui32 featureId) const;

        const TVector<float>& GetFloatFeatureBorders(const TFloatValuesHolder& feature) const;

        bool HasBorders(ui32 featureId) const {
            return Borders.has(featureId);
        }

        void SetBorders(ui32 featureId, TVector<float> borders) {
            CB_ENSURE(!HasBorders(featureId));
            Borders[featureId] = std::move(borders);
        }

        bool IsFloat(ui32 featureId) const;

        bool IsCat(ui32 featureId) const;

        TVector<ui32> GetOneHotIds(const TVector<ui32>& ids) const;

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

        bool IsPermutationDependent(const TCtr& ctr) const;

        bool HasPermutationDependentSplit(const TVector<TBinarySplit>& splits) const;

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

        ui32 GetFeatureManagerIdForCatFeature(ui32 dataProviderId) const;

        ui32 GetFeatureManagerIdForFloatFeature(ui32 dataProviderId) const;

        ui32 GetFeatureManagerId(const IFeatureValuesHolder& feature) const;

        ui32 GetDataProviderId(ui32 featureId) const {
            return FeatureManagerIdToDataProviderId.at(featureId);
        }

        bool IsKnown(const TCtr& ctr) const {
            return KnownCtrs.has(ctr);
        }

        ui32 AddCtr(const TCtr& ctr);

        ui32 AddCtr(const TCtr& ctr,
                    TVector<float>&& borders) {
            ui32 id = AddCtr(ctr);
            Borders[id] = std::move(borders);
            return id;
        }

        const TVector<float>& GetBorders(ui32 id) const {
            return Borders.at(id);
        }

        ui32 CtrsPerTreeCtrFeatureTensor() const;

        ui32 GetBinCount(ui32 localId) const;

        const NCatboostOptions::TBinarizationOptions& GetFloatFeatureBinarization() const {
            return FloatFeaturesBinarization;
        }

        const NCatboostOptions::TBinarizationOptions& GetTargetBinarizationDescription() const {
            return CatFeatureOptions.TargetBorders;
        }

        const NCatboostOptions::TBinarizationOptions& GetBinarizationDescription(const TCtr& ctr) const {
            return GetCtrBinarizationForConfig(ctr.Configuration);
        }

        ui32 GetId(const IFeatureValuesHolder& feature) const;

        ui32 GetId(const TCtr& ctr) const {
            CB_ENSURE(KnownCtrs.has(ctr));
            return KnownCtrs[ctr];
        }

        TSet<ECtrType> GetKnownSimpleCtrTypes() const;

        TVector<ui32> CreateSimpleCtrsForType(ui32 featureId,
                                              ECtrType type);

        TVector<ui32> GetAllSimpleCtrs() const;

        TVector<ui32> CreateCombinationCtrForType(ECtrType type);

        void CreateSimpleCtrs(const ui32 featureId, const TSet<TCtrConfig>& configs, TSet<ui32>* resultIds);

        TVector<TCtrConfig> CreateTreeCtrConfigs() const;

        TMap<ECtrType, TSet<TCtrConfig>> CreateGrouppedTreeCtrConfigs() const;

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

        TMap<ECtrType, TSet<TCtrConfig>> CreateGrouppedSimpleCtrConfigs() const;

        TMap<ui32, TMap<ECtrType, TSet<TCtrConfig>>> CreateGrouppedPerFeatureCtrs() const;

        bool HasPerFeatureCtr(ui32 featureId) const {
            ui32 featureIdInPool = GetDataProviderId(featureId);
            return CatFeatureOptions.PerFeatureCtrs->has(featureIdInPool);
        }

        TMap<ECtrType, TSet<TCtrConfig>> CreateGrouppedPerFeatureCtr(ui32 featureId) const;

        const NCatboostOptions::TCatFeatureParams& GetCatFeatureOptions() const {
            return CatFeatureOptions;
        }

        void AddCustomCtr(const TFeatureTensor& tensor, const NCatboostOptions::TCtrDescription& description) {
            UserCombinations.push_back(TUserDefinedCombination(tensor, description));
        }

    private:
        void CreateCtrConfigsFromDescription(const NCatboostOptions::TCtrDescription& ctrDescription,
                                             TMap<ECtrType, TSet<TCtrConfig>>* grouppedConfigs) const;

        const NCatboostOptions::TBinarizationOptions& GetCtrBinarizationForConfig(const TCtrConfig& config) const {
            CB_ENSURE(config.CtrBinarizationConfigId < CtrBinarizationOptions.size(), "error: unknown ctr binarization id " << config.CtrBinarizationConfigId);
            return CtrBinarizationOptions[config.CtrBinarizationConfigId];
        }

        //stupid line-search here is not issue :)
        inline ui32 GetOrCreateCtrBinarizationId(const NCatboostOptions::TBinarizationOptions& binarization) const;

        ui32 GetUniqueValues(ui32 featureId) const {
            CB_ENSURE(IsCat(featureId));
            return CatFeaturesPerfectHash.GetUniqueValues(featureId);
        }

        ui32 RequestNewId() {
            return Cursor++;
        }

        TBinarizedFeaturesManager& SetBinarization(ui64 featureId,
                                                   TVector<float>&& borders);

        friend class TCatFeaturesPerfectHashHelper;

        inline ENanMode ComputeNanMode(const float* values, ui32 size) const;

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
