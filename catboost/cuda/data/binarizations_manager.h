#pragma once

#include "feature.h"
#include "binarization_config.h"
#include "cat_feature_perfect_hash.h"
#include <library/binsaver/bin_saver.h>
#include <util/generic/guid.h>

namespace NCatboostCuda
{
    class TFeatureManagerOptions
    {
    public:
        TFeatureManagerOptions(const TBinarizationConfiguration& binarizationConfiguration,
                               ui32 oneHotLimit)
                : BinarizationConfiguration(binarizationConfiguration)
                  , OneHotLimit(oneHotLimit)
        {
            EnabledCtrTypes.insert(ECtrType::Borders);
            EnabledCtrTypes.insert(ECtrType::FeatureFreq);
        }

        TFeatureManagerOptions() {
            EnabledCtrTypes.insert(ECtrType::Borders);
            EnabledCtrTypes.insert(ECtrType::FeatureFreq);
        }

        const TBinarizationConfiguration& GetBinarizationConfiguration() const
        {
            return BinarizationConfiguration;
        }

        TFeatureManagerOptions& SetTargetBinarization(ui32 discretiation)
        {
            BinarizationConfiguration.TargetBinarization.Discretization = discretiation;
            return *this;
        }

        ui32 GetOneHotLimit() const
        {
            return OneHotLimit;
        }

        ui32 GetMaxTensorComplexity() const
        {
            return MaxTensorComplexity;
        }

        void SetOneHotLimit(ui32 OneHotLimit)
        {
            TFeatureManagerOptions::OneHotLimit = OneHotLimit;
        }

        void SetMaxTensorComplexity(ui32 MaxTensorComplexity)
        {
            TFeatureManagerOptions::MaxTensorComplexity = MaxTensorComplexity;
        }

        bool IsCtrTypeEnabled(ECtrType ctrType) const
        {
            return EnabledCtrTypes.has(ctrType);
        }

        bool IsCustomCtrTypes() const
        {
            return CustomCtrTypes;
        }

        void EnableCtrType(ECtrType ctrType)
        {
            EnabledCtrTypes.insert(ctrType);
        }

        void DisableCtrType(ECtrType ctrType)
        {
            EnabledCtrTypes.erase(ctrType);
        }

        const TString& GetCatFeatureBinarizationTempName() const
        {
            return CatFeatureBinarizationTempName;
        }

        const yset<ui32>& GetIgnoredFeatures() const
        {
            return IgnoredFeatures;
        }

        template<class TConfig>
        friend
        class TOptionsBinder;


        template<class TConfig>
        friend
        class TOptionsJsonConverter;

        SAVELOAD(BinarizationConfiguration, OneHotLimit, MaxTensorComplexity,
                 EnabledCtrTypes, CustomCtrTypes, CatFeatureBinarizationTempName);

        void Validate()
        {
            CB_ENSURE(OneHotLimit < 255, "Maximum one hot limit is 255");
        }

    private:
        TBinarizationConfiguration BinarizationConfiguration;

        ui32 OneHotLimit = 0;
        ui32 MaxTensorComplexity = 4; //tensor complexity is number  of catFeatures, which could be used in one ctr
        yset<ui32> IgnoredFeatures;
        yset<ECtrType> EnabledCtrTypes;
        bool CustomCtrTypes = false;
        TString CatFeatureBinarizationTempName = TStringBuilder() << "cat_feature_index." << CreateGuidAsString() << ".tmp";

    };


//stores expression for binarized features calculations and mapping from this expression to unique ids
    class TBinarizedFeaturesManager
    {
    public:
        explicit TBinarizedFeaturesManager(const TFeatureManagerOptions& options)
                : FeatureManagerOptions(options)
                , CatFeaturesPerfectHash(options.GetCatFeatureBinarizationTempName())
        {
        }

        bool IsKnown(const ui32 featuresProviderId) const
        {
            return DataProviderFloatFeatureIdToFeatureManagerId.has(featuresProviderId) ||
                   DataProviderCatFeatureIdToFeatureManagerId.has(featuresProviderId);
        }

        bool IsKnown(const IFeatureValuesHolder& feature) const
        {
            return IsKnown(feature.GetId());
        }

        template<class TBuilder>
        const yvector<float>& GetOrCreateFloatFeatureBorders(const TFloatValuesHolder& feature,
                                                             TBuilder&& builder)
        {
            CB_ENSURE(IsKnown(feature));
            const ui32 featureId = GetFeatureManagerId(feature);
            if (!Borders.has(featureId))
            {
                Borders[featureId] = builder(GetDefaultFloatFeatureBinarizationDescription());
            }
            return Borders[featureId];
        }


        bool HasFloatFeatureBorders(const TFloatValuesHolder& feature) const
        {
            return Borders.has(GetId(feature));
        }

        const yvector<float>& GetFloatFeatureBorders(const TFloatValuesHolder& feature) const
        {
            CB_ENSURE(IsKnown(feature));
            ui32 id = GetId(feature);
            return Borders.at(id);
        }

        template<class TBuilder>
        yvector<float> GetOrBuildCtrBorders(const TCtr& ctr,
                                            TBuilder&& gridBuilder)
        {
            ui32 ctrId;

            if (!KnownCtrs.has(ctr))
            {
                ctrId = AddCtr(ctr);
            } else
            {
                ctrId = GetId(ctr);
            }

            if (!Borders.has(ctrId))
            {
                Borders[ctrId] = gridBuilder();
            }
            return Borders[ctrId];
        }

        bool IsFloat(ui32 featureId) const
        {
            if (FeatureManagerIdToDataProviderId.has(featureId))
            {
                return DataProviderFloatFeatureIdToFeatureManagerId.has(FeatureManagerIdToDataProviderId.at(featureId));
            } else
            {
                return false;
            }
        }

        bool IsCat(ui32 featureId) const
        {
            if (FeatureManagerIdToDataProviderId.has(featureId))
            {
                return DataProviderCatFeatureIdToFeatureManagerId.has(FeatureManagerIdToDataProviderId.at(featureId));
            } else
            {
                return false;
            }
        }

        yvector<ui32> GetOneHotIds(const yvector<ui32>& ids) const
        {
            yvector<ui32> result;
            for (auto id : ids)
            {
                if (IsCat(id) && UseForOneHotEncoding(id))
                {
                    result.push_back(id);
                }
            }
            return result;
        }

        bool UseForOneHotEncoding(ui32 featureId) const
        {
            CB_ENSURE(IsCat(featureId));
            return GetUniqueValues(featureId) <= FeatureManagerOptions.GetOneHotLimit();
        }

        bool UseForCtr(ui32 featureId) const
        {
            CB_ENSURE(IsCat(featureId));
            return GetUniqueValues(featureId) > FeatureManagerOptions.GetOneHotLimit();
        }

        bool UseForTreeCtr(ui32 featureId) const
        {
            CB_ENSURE(IsCat(featureId));
            return GetUniqueValues(featureId) > FeatureManagerOptions.GetOneHotLimit() &&
                   (FeatureManagerOptions.GetMaxTensorComplexity() > 1);
        }

        bool IsTreeCtrsEnabled() const
        {
            return !DataProviderCatFeatureIdToFeatureManagerId.empty() &&
                   (FeatureManagerOptions.GetMaxTensorComplexity() > 1);
        }

        bool UseAsBaseTensorForTreeCtr(const TFeatureTensor& tensor) const {
            return (tensor.GetComplexity() < FeatureManagerOptions.GetMaxTensorComplexity());
        }

        bool UseForTreeCtr(const TFeatureTensor& tensor) const {
            return (tensor.GetComplexity() <= FeatureManagerOptions.GetMaxTensorComplexity());
        }

        bool IsCtr(ui32 featureId) const
        {
            CB_ENSURE(featureId < Cursor);
            return InverseCtrs.has(featureId);
        }

        bool IsTreeCtr(ui32 featureId) const
        {
            CB_ENSURE(featureId < Cursor);
            return IsCtr(featureId) && !GetCtr(featureId).IsSimple();
        }

        bool IsPermutationDependent(const TCtr& ctr) const
        {
            if (IsPermutationDependentCtrType(ctr.Configuration.Type))
            {
                return true;
            }
            return HasPermutationDependentSplit(ctr.FeatureTensor.GetSplits());
        }

        bool HasPermutationDependentSplit(const yvector<TBinarySplit>& splits) const
        {
            for (auto& split : splits)
            {
                if (IsCtr(split.FeatureId) && IsPermutationDependent(GetCtr(split.FeatureId)))
                {
                    return true;
                }
            }
            return false;
        }

        const TCtr& GetCtr(ui32 featureId) const
        {
            CB_ENSURE(featureId < Cursor);
            return InverseCtrs.at(featureId);
        }

        ui32 GetFeatureCount() const
        {
            return Cursor;
        }

        ui32 RegisterDataProviderCatFeature(ui32 featureId)
        {
            CB_ENSURE(!DataProviderCatFeatureIdToFeatureManagerId.has(featureId));
            const ui32 id = RequestNewId();
            DataProviderCatFeatureIdToFeatureManagerId[featureId] = id;
            FeatureManagerIdToDataProviderId[id] = featureId;
            CatFeaturesPerfectHash.RegisterId(id);
            return id;
        }


        ui32 RegisterDataProviderFloatFeature(ui32 featureId)
        {
            CB_ENSURE(!DataProviderFloatFeatureIdToFeatureManagerId.has(featureId));
            const ui32 id = RequestNewId();
            DataProviderFloatFeatureIdToFeatureManagerId[featureId] = id;
            FeatureManagerIdToDataProviderId[id] = featureId;
            return id;
        }

        bool HasFloatFeatureBordersForDataProviderFeature(const ui32 dataProviderId)
        {
            const ui32 featureId = GetFeatureManagerIdForFloatFeature(dataProviderId);
            return Borders.has(featureId);
        }

        ui32 SetFloatFeatureBordersForDataProviderId(ui32 dataProviderId,
                                                     yvector<float>&& borders)
        {
            const ui32 id = GetFeatureManagerIdForFloatFeature(dataProviderId);
            Borders[id] = std::move(borders);
            return id;
        }

        ui32 SetFloatFeatureBorders(const TFloatValuesHolder& feature,
                                    yvector<float>&& borders)
        {
            CB_ENSURE(IsKnown(feature));

            const ui32 id = GetFeatureManagerId(feature);
            Borders[id] = std::move(borders);
            return id;
        }

        ui32 GetFeatureManagerIdForCatFeature(ui32 dataProviderId) const
        {
            CB_ENSURE(DataProviderCatFeatureIdToFeatureManagerId.has(dataProviderId),
                      "Error: feature #" << dataProviderId << " is not categorical");
            return DataProviderCatFeatureIdToFeatureManagerId.at(dataProviderId);
        }

        ui32 GetFeatureManagerIdForFloatFeature(ui32 dataProviderId) const
        {
            CB_ENSURE(DataProviderFloatFeatureIdToFeatureManagerId.has(dataProviderId),
                      "Error: feature #" << dataProviderId << " is not categorical");
            return DataProviderFloatFeatureIdToFeatureManagerId.at(dataProviderId);
        }

        ui32 GetFeatureManagerId(const IFeatureValuesHolder& feature) const
        {
            switch (feature.GetType())
            {
                case EFeatureValuesType::BinarizedFloat:
                case EFeatureValuesType::Float:
                {
                    return GetFeatureManagerIdForFloatFeature(feature.GetId());
                }
                case EFeatureValuesType::Categorical:
                {
                    return GetFeatureManagerIdForCatFeature(feature.GetId());
                }
                default:
                {
                    ythrow TCatboostException() << "Unknown feature id " << feature.GetId();
                }
            }
        }

        ui32 GetDataProviderId(ui32 featureId) const
        {
            return FeatureManagerIdToDataProviderId.at(featureId);
        }

        bool IsKnown(const TCtr& ctr) const
        {
            return KnownCtrs.has(ctr);
        }

        ui32 AddCtr(const TCtr& ctr)
        {
            CB_ENSURE(!KnownCtrs.has(ctr));
            const ui32 id = RequestNewId();
            KnownCtrs[ctr] = id;
            InverseCtrs[id] = ctr;
            return id;
        }

        ui32 AddCtr(const TCtr& ctr,
                    yvector<float>&& borders)
        {
            ui32 id = AddCtr(ctr);
            Borders[id] = std::move(borders);
            return id;
        }

        const yvector<float>& GetBorders(ui32 id) const
        {
            return Borders.at(id);
        }

        ui32 CtrsPerFeatureTensorCount() const
        {
            ui32 totalCount = 0;
            for (auto& ctrType : EnabledCtrTypes)
            {
                totalCount += GetDefaultConfigForType(ctrType).size();
            }
            return totalCount;
        }

        ui32 GetBinCount(ui32 localId) const
        {
            if (Borders.has(localId))
            {
                return Borders.at(localId).size() + 1;
            } else if (IsCat(localId))
            {
                return GetUniqueValues(localId);
            } else if (InverseCtrs.has(localId))
            {
                return GetBinarizationDescription(InverseCtrs[localId]).Discretization + 1;
            } else if (IsFloat(localId)) {
                return 0;
            } else
            {
                ythrow TCatboostException() << "Error: unknown feature id #" << localId;
            }
        }

        const TBinarizationDescription& GetBinarizationDescription(ui32 featureId) const
        {
            Y_UNUSED(featureId);
            return GetDefaultBinarization().DefaultFloatBinarization;
        }

        const TBinarizationDescription& GetDefaultFloatFeatureBinarizationDescription() const
        {
            return GetDefaultBinarization().DefaultFloatBinarization;
        }

        const TBinarizationDescription& GetTargetBinarizationDescription() const
        {
            return GetDefaultBinarization().TargetBinarization;
        }

        const TBinarizationDescription& GetBinarizationDescription(const TCtr& ctr) const
        {
            if (ctr.FeatureTensor.IsSimple())
            {
                return GetDefaultBinarization().DefaultCtrBinarization;
            } else
            {
                return GetDefaultBinarization().DefaultTreeCtrBinarization;
            }
        }

        ui32 GetId(const IFeatureValuesHolder& feature) const
        {
            const ui32 featureId = feature.GetId();

            if (DataProviderFloatFeatureIdToFeatureManagerId.has(featureId))
            {
                return DataProviderFloatFeatureIdToFeatureManagerId[featureId];
            } else if (DataProviderCatFeatureIdToFeatureManagerId.has(featureId))
            {
                return DataProviderCatFeatureIdToFeatureManagerId[featureId];
            } else
            {
                ythrow TCatboostException() << "Error: unknown feature with id #" << feature.GetId();
            }
        }

        ui32 GetId(const TCtr& ctr) const
        {
            CB_ENSURE(KnownCtrs.has(ctr));
            return KnownCtrs[ctr];
        }

        yset<ECtrType> GetEnabledCtrTypes() const
        {
            return EnabledCtrTypes;
        }

        const yvector<TCtrConfig>& GetDefaultConfigForType(ECtrType type) const
        {
            return DefaultCtrConfigsForType.at(type);
        }

        yvector<ui32> CreateSimpleCtrsForType(ui32 featureId,
                                              ECtrType type)
        {
            CB_ENSURE(UseForCtr(featureId));

            yvector<ui32> resultIds;

            for (const auto& ctrConfig : GetDefaultConfigForType(type))
            {
                TCtr ctr;
                ctr.FeatureTensor.AddCatFeature(featureId);
                ctr.Configuration = GetDefaultCtrConfigurationForTensor(ctrConfig,
                                                                        ctr.FeatureTensor);
                if (!KnownCtrs.has(ctr))
                {
                    AddCtr(ctr);
                }
                resultIds.push_back(GetId(ctr));
            }
            return resultIds;
        }

        yvector<TCtrConfig> CreateCtrsConfigsForTensor(const TFeatureTensor& tensor) const
        {
            yvector<TCtrConfig> result;

            for (const auto& ctrType : EnabledCtrTypes)
            {
                CB_ENSURE(IsSupportedCtrType(ctrType));
                for (const auto& ctrConfig : GetDefaultConfigForType(ctrType))
                {
                    result.push_back(GetDefaultCtrConfigurationForTensor(ctrConfig, tensor));
                }
            }
            return result;
        }

        TCtrConfig GetDefaultCtrConfigurationForTensor(const TCtrConfig& defaultConfig,
                                                       const TFeatureTensor& tensor) const
        {
            switch (defaultConfig.Type)
            {
                //feature-freq ctr should have non-informative prior by default and depend on unique feature values:)
                case ECtrType::FeatureFreq:
                {
                    TCtrConfig config = defaultConfig;
                    ui32 maxUniqueValues = GetMaxUniqueValues(tensor);
                    CB_ENSURE(config.Prior.size() == 1);
                    config.Prior.push_back((float) maxUniqueValues * config.Prior[0]);
                    return config;
                }
                default:
                {
                    return defaultConfig;
                }
            }
        }

        ui32 MaxTreeCtrBinFeaturesCount() const
        {
            ui32 total = 0;
            for (auto& type : EnabledCtrTypes)
            {
                ui32 binarization = GetDefaultTreeCtrBinarization(type).Discretization;
                total += DefaultCtrConfigsForType.at(type).size() * binarization;
            }
            return total;
        }

        ui64 GetMaxUniqueValues(const TFeatureTensor& tensor) const
        {
            ui64 uniqueValues = 1;
            for (const auto& cat : tensor.GetCatFeatures())
            {
                uniqueValues *= GetBinCount(cat);
            }
            uniqueValues <<= tensor.GetSplits().size();
            return uniqueValues;
        }

        yvector<ui32> GetCatFeatureIds() const
        {
            yvector<ui32> featureIds;

            for (const auto& feature : DataProviderCatFeatureIdToFeatureManagerId)
            {
                if (GetBinCount(feature.second))
                {
                    featureIds.push_back(feature.second);
                }
            }
            Sort(featureIds.begin(), featureIds.end());
            return featureIds;
        }

        yvector<ui32> GetFloatFeatureIds() const
        {
            yvector<ui32> featureIds;

            for (const auto& feature : DataProviderFloatFeatureIdToFeatureManagerId)
            {
                if (GetBinCount(feature.second))
                {
                    featureIds.push_back(feature.second);
                }
            }
            return featureIds;
        }

        ui32 FeatureCount() const
        {
            return Cursor;
        }

        bool HasTargetBinarization() const
        {
            return static_cast<bool>(!GetTargetBorders().empty());
        }

        TBinarizedFeaturesManager& SetTargetBorders(yvector<float>&& borders)
        {
            TargetBorders = borders;
            return *this;
        }

        const yvector<float>& GetTargetBorders() const
        {
            return TargetBorders;
        }

        yvector<ui32> GetDataProviderFeatureIds() const
        {
            yvector<ui32> features;
            for (auto id : GetFloatFeatureIds())
            {
                Y_ASSERT(GetBinCount(id));
                features.push_back(id);
            }
            for (auto id : GetCatFeatureIds())
            {
                Y_ASSERT(GetBinCount(id));
                features.push_back(id);
            }
            return features;
        }

        const TBinarizationDescription& GetCtrBinarization(const TCtr& ctr) const
        {
            if (ctr.Configuration.Type == ECtrType::FeatureFreq)
            {
                return ctr.IsSimple() ? GetDefaultBinarization().FreqCtrBinarization
                                      : GetDefaultBinarization().FreqTreeCtrBinarization;
            }
            if (ctr.FeatureTensor.IsSimple())
            {
                return GetDefaultBinarization().DefaultCtrBinarization;
            }
            return ctr.FeatureTensor.IsSimple() ? GetDefaultBinarization().DefaultCtrBinarization
                                                : GetDefaultBinarization().DefaultTreeCtrBinarization;
        }

        inline const TBinarizationDescription& GetDefaultTreeCtrBinarization(const ECtrType type) const
        {
            return type == ECtrType::FeatureFreq ? GetDefaultBinarization().FreqTreeCtrBinarization
                                                 : GetDefaultBinarization().DefaultTreeCtrBinarization;
        }

        TBinarizedFeaturesManager& EnableCtrType(ECtrType type,
                                                 yvector<float> prior)
        {
            CB_ENSURE(TargetBorders.size() != 0, "Enable ctr_description should be done after target borders are set");

            TCtrConfig defaultConfig;
            defaultConfig.Prior = prior;
            defaultConfig.Type = type;
            EnabledCtrTypes.insert(type);

            if (type == ECtrType::Buckets || type == ECtrType::Borders)
            {
                ui32 numBins = (type == ECtrType::Buckets)
                               ? TargetBorders.size() + 1
                               : TargetBorders.size();
                const ui32 numCtrs = numBins;

                if (prior.size() == 1)
                {
                    defaultConfig.Prior.resize(numBins, prior[0]);
                } else
                {
                    CB_ENSURE(prior.size() == numBins);
                }

                for (ui32 i = 0; i < numCtrs; ++i)
                {
                    //don't calc 0-class ctr for binary classification, it's unneeded
                    if (i == 0 && numBins == 2 && type == ECtrType::Buckets)
                    {
                        continue;
                    }
                    TCtrConfig config = defaultConfig;
                    config.ParamId = i;
                    DefaultCtrConfigsForType[type].push_back(config);
                }
            } else
            {
                DefaultCtrConfigsForType[type].push_back(defaultConfig);
            }
            return *this;
        }

        TBinarizedFeaturesManager& SetOneHotLimit(ui32 limit)
        {
            FeatureManagerOptions.SetOneHotLimit(limit);
            return *this;
        }

        //store perfect hash by featureManager id
        const ymap<int, ui32>& GetCategoricalFeaturesPerfectHash(const ui32 featureId) const
        {
            CB_ENSURE(CatFeaturesPerfectHash.HasFeature(featureId));
            return CatFeaturesPerfectHash.GetFeatureIndex(featureId);
        };

        void UnloadCatFeaturePerfectHashFromRam() const
        {
            CatFeaturesPerfectHash.FreeRam();
        }

        SAVELOAD(KnownCtrs, InverseCtrs,
                 DataProviderFloatFeatureIdToFeatureManagerId, DataProviderCatFeatureIdToFeatureManagerId,
                 FeatureManagerIdToDataProviderId, Cursor, FeatureManagerOptions,
                 DefaultCtrConfigsForType, Borders, CatFeaturesPerfectHash, TargetBorders);



    private:

        ui32 GetUniqueValues(ui32 featureId) const
        {
            CB_ENSURE(IsCat(featureId));
            return CatFeaturesPerfectHash.GetUniqueValues(featureId);
        }

        ui32 RequestNewId()
        {
            return Cursor++;
        }

        TBinarizedFeaturesManager& SetBinarization(ui64 featureId,
                                                   yvector<float>&& borders)
        {
            Borders[featureId] = std::move(borders);
            return *this;
        }

        const TBinarizationConfiguration& GetDefaultBinarization() const
        {
            return FeatureManagerOptions.GetBinarizationConfiguration();
        }

        friend class TCatFeaturesPerfectHashHelper;

    private:
        mutable ymap<TCtr, ui32> KnownCtrs;
        mutable ymap<ui32, TCtr> InverseCtrs;

        mutable ymap<ui32, ui32> DataProviderFloatFeatureIdToFeatureManagerId;
        mutable ymap<ui32, ui32> DataProviderCatFeatureIdToFeatureManagerId;
        mutable ymap<ui32, ui32> FeatureManagerIdToDataProviderId;

        mutable ui32 Cursor = 0;

        TFeatureManagerOptions FeatureManagerOptions;
        ymap<ECtrType, yvector<TCtrConfig>> DefaultCtrConfigsForType;

        //float and ctr features
        ymap<ui32, yvector<float>> Borders;
        TCatFeaturesPerfectHash CatFeaturesPerfectHash;
        yset<ECtrType> EnabledCtrTypes;
        yvector<float> TargetBorders;
    };
}
