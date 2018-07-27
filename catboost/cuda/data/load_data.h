#pragma once

#include "data_provider.h"
#include "grid_creator.h"
#include "binarizations_manager.h"
#include "data_utils.h"
#include "cat_feature_perfect_hash_helper.h"
#include "binarized_features_meta_info.h"

#include <catboost/cuda/utils/compression_helpers.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/utils/cpu_random.h>

#include <library/threading/local_executor/fwd.h>

#include <util/stream/file.h>
#include <util/system/spinlock.h>
#include <util/system/sem.h>
#include <util/random/shuffle.h>

namespace NCB {
    struct TPathWithScheme;
}

namespace NCatboostCuda {
    class TDataProviderBuilder: public NCB::IPoolBuilder {
    public:
        TDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
                             TDataProvider& dst,
                             bool isTest = false,
                             const int buildThreads = 1)
            : FeaturesManager(featureManager)
            , DataProvider(dst)
            , IsTest(isTest)
            , BuildThreads(buildThreads)
            , CatFeaturesPerfectHashHelper(FeaturesManager)
        {
        }

        template <class TContainer>
        TDataProviderBuilder& AddIgnoredFeatures(const TContainer& container) {
            for (const auto& f : container) {
                IgnoreFeatures.insert(f);
            }
            return *this;
        }

        TDataProviderBuilder& SetClassesWeights(const TVector<float>& classesWeights) {
            ClassesWeights = classesWeights;
            return *this;
        }

        TDataProviderBuilder& SetBinarizedFeaturesMetaInfo(const TBinarizedFloatFeaturesMetaInfo& binarizedFeaturesMetaInfo) {
            BinarizedFeaturesMetaInfo = binarizedFeaturesMetaInfo;
            return *this;
        }

        void Start(const TPoolMetaInfo& poolMetaInfo,
                   int docCount,
                   const TVector<int>& catFeatureIds) override;

        TDataProviderBuilder& SetShuffleFlag(bool shuffle, ui64 seed = 0) {
            ShuffleFlag = shuffle;
            Seed = seed;
            return *this;
        }

        void StartNextBlock(ui32 blockSize) override;

        float GetCatFeatureValue(const TStringBuf& feature) override {
            return ConvertCatFeatureHashToFloat(StringToIntHash(feature));
        }

        void AddCatFeature(ui32 localIdx,
                           ui32 featureId,
                           const TStringBuf& feature) override {
            if (IgnoreFeatures.count(featureId) == 0) {
                Y_ASSERT(FeatureTypes[featureId] == EFeatureValuesType::Categorical);
                WriteFloatOrCatFeatureToBlobImpl(localIdx,
                                                 featureId,
                                                 ConvertCatFeatureHashToFloat(StringToIntHash(feature)));
            }
        }

        void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) override {
            if (IgnoreFeatures.count(featureId) == 0) {
                switch (FeatureTypes[featureId]) {
                    case EFeatureValuesType::BinarizedFloat: {
                        ui8 binarizedFeature = Binarize<ui8>(Borders[featureId], feature);
                        WriteBinarizedFeatureToBlobImpl(localIdx, featureId, binarizedFeature);
                        break;
                    }
                    case EFeatureValuesType::Float: {
                        WriteFloatOrCatFeatureToBlobImpl(localIdx, featureId, feature);
                        break;
                    }
                    default: {
                        CB_ENSURE(false, "Unsupported type " << FeatureTypes[featureId]);
                    }
                }
            }
        }

        void AddBinarizedFloatFeature(ui32 localIdx, ui32 featureId, ui8 binarizedFeature) override {
            if (IgnoreFeatures.count(featureId) == 0) {
                CB_ENSURE(FeatureTypes[featureId] == EFeatureValuesType::BinarizedFloat, "FeatureValueType doesn't match: expect BinarizedFloat, got " << FeatureTypes[featureId]);
                WriteBinarizedFeatureToBlobImpl(localIdx, featureId, binarizedFeature);
            }
        }

        void AddAllFloatFeatures(ui32 localIdx, TConstArrayRef<float> features) override {
            CB_ENSURE(features.size() == FeatureBlobs.size(),
                      "Error: number of features should be equal to factor count");
            for (size_t featureId = 0; featureId < FeatureBlobs.size(); ++featureId) {
                if (IgnoreFeatures.count(featureId) == 0) {
                    if (FeatureTypes[featureId] == EFeatureValuesType::Categorical) {
                        WriteFloatOrCatFeatureToBlobImpl(localIdx, featureId, features[featureId]);
                    } else {
                        AddFloatFeature(localIdx, featureId, features[featureId]);
                    }
                }
            }
        }

        void AddTarget(ui32 localIdx, float value) override {
            DataProvider.Targets[GetLineIdx(localIdx)] = value;
        }

        void AddWeight(ui32 localIdx, float value) override {
            DataProvider.Weights[GetLineIdx(localIdx)] = value;
        }

        void AddQueryId(ui32 localIdx, TGroupId queryId) override {
            DataProvider.QueryIds[GetLineIdx(localIdx)] = queryId;
        }

        void AddSubgroupId(ui32 localIdx, TSubgroupId groupId) override {
            DataProvider.SubgroupIds[GetLineIdx(localIdx)] = groupId;
        }

        void AddBaseline(ui32 localIdx, ui32 baselineIdx, double value) override {
            DataProvider.Baseline[baselineIdx][GetLineIdx(localIdx)] = (float)value;
        }

        void AddDocId(ui32 localIdx, const TStringBuf& value) override {
            DataProvider.DocIds[GetLineIdx(localIdx)] = StringToIntHash(value);
        }

        void AddTimestamp(ui32 localIdx, ui64 timestamp) override {
            DataProvider.Timestamp[GetLineIdx(localIdx)] = timestamp;
        }

        void SetFeatureIds(const TVector<TString>& featureIds) override {
            FeatureNames = featureIds;
        }

        void SetPairs(const TVector<TPair>& pairs) override {
            CB_ENSURE(!IsDone, "Error: can't set pairs after finish");
            Pairs = pairs;
        }

        void SetFloatFeatures(const TVector<TFloatFeature>& floatFeatures) override {
            Y_UNUSED(floatFeatures);
            CB_ENSURE(false, "Not supported for regular pools");
        }

        int GetDocCount() const override {
            return DataProvider.Targets.size();
        }

        TConstArrayRef<float> GetWeight() const override {
            return MakeArrayRef(DataProvider.Weights.data(), DataProvider.Weights.size());
        }

        void GenerateDocIds(int offset) override {
            for (int ind = 0; ind < DataProvider.DocIds.ysize(); ++ind) {
                DataProvider.DocIds[ind] = offset + ind;
            }
        }

        void Finish() override;

        void RegisterFeaturesInFeatureManager(const TVector<TFeatureColumnPtr>& featureColumns) const {
            for (ui32 featureId = 0; featureId < featureColumns.size(); ++featureId) {
                if (!FeaturesManager.IsKnown(featureId)) {
                    if (FeatureTypes[featureId] == EFeatureValuesType::Categorical) {
                        FeaturesManager.RegisterDataProviderCatFeature(featureId);
                    } else {
                        FeaturesManager.RegisterDataProviderFloatFeature(featureId);
                    }
                }
            }
        }

    private:
        ui32 GetBytesPerFeature(ui32 featureId) const {
            return FeatureTypes.at(featureId) != EFeatureValuesType::BinarizedFloat ? 4 : 1;
        }

        void WriteBinarizedFeatureToBlobImpl(ui32 localIdx, ui32 featureId, ui8 feature);
        void WriteFloatOrCatFeatureToBlobImpl(ui32 localIdx, ui32 featureId, float feautre);

    private:
        inline ui32 GetLineIdx(ui32 localIdx) {
            return Cursor + localIdx;
        }

        inline TString GetFeatureName(ui32 featureId) {
            return FeatureNames.size() ? FeatureNames[featureId] : ToString<ui32>(featureId);
        }

        TBinarizedFeaturesManager& FeaturesManager;
        TDataProvider& DataProvider;
        bool IsTest;
        ui32 BuildThreads;
        TCatFeaturesPerfectHashHelper CatFeaturesPerfectHashHelper;

        bool ShuffleFlag = false;
        ui64 Seed = 0;
        ui32 Cursor = 0;
        bool IsDone = false;

        TBinarizedFloatFeaturesMetaInfo BinarizedFeaturesMetaInfo;
        TVector<TVector<ui8>> FeatureBlobs;
        TVector<EFeatureValuesType> FeatureTypes;
        TVector<TVector<float>> Borders;
        TVector<ENanMode> NanModes;

        TSet<ui32> IgnoreFeatures;
        TVector<TString> FeatureNames;

        TVector<float> ClassesWeights;
        TVector<TPair> Pairs;
    };

    void ReadPool(
        const ::NCB::TPathWithScheme& poolPath,
        const ::NCB::TPathWithScheme& pairsFilePath, // can be uninited
        const ::NCatboostOptions::TDsvPoolFormatParams& dsvPoolFormatParams,
        const TVector<int>& ignoredFeatures,
        bool verbose,
        const TVector<TString>& classNames,
        ::NPar::TLocalExecutor* localExecutor,
        TDataProviderBuilder* poolBuilder);
}
