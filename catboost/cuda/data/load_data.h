#pragma once

#include "data_provider.h"
#include "grid_creator.h"
#include "binarizations_manager.h"
#include "data_utils.h"
#include "cat_feature_perfect_hash_helper.h"

#include <catboost/cuda/utils/compression_helpers.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data_types/pair.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/utils/cpu_random.h>
#include <util/stream/file.h>
#include <util/system/spinlock.h>
#include <util/system/sem.h>
#include <util/random/shuffle.h>

namespace NCatboostCuda {
    class TDataProviderBuilder: public IPoolBuilder {
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

        void Start(const TPoolColumnsMetaInfo& metaInfo, int docCount) override {
            DataProvider.Features.clear();

            DataProvider.Baseline.clear();
            DataProvider.Baseline.resize(metaInfo.BaselineCount);

            Cursor = 0;
            IsDone = false;
            FeatureValues.clear();
            FeatureValues.resize(metaInfo.FactorCount);
            for (ui32 i = 0; i < metaInfo.FactorCount; ++i) {
                if (!IgnoreFeatures.has(i)) {
                    FeatureValues[i].reserve(docCount);
                }
            }

            CatFeatureIds = TSet<int>(metaInfo.CatFeatureIds.begin(), metaInfo.CatFeatureIds.end());
        }

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
                Y_ASSERT(CatFeatureIds.has(featureId));
                int hash = StringToIntHash(feature);
                //dirty c++ hack to store everything in float-vector
                AddFloatFeature(localIdx, featureId, ConvertCatFeatureHashToFloat(hash));
            }
        }

        void AddFloatFeature(ui32 localIdx, ui32 featureId, float feature) override {
            if (IgnoreFeatures.count(featureId) == 0) {
                auto& featureColumn = FeatureValues[featureId];
                featureColumn[GetLineIdx(localIdx)] = feature;
            }
        }

        void AddAllFloatFeatures(ui32 localIdx, const TVector<float>& features) override {
            CB_ENSURE(features.ysize() == FeatureValues.ysize(),
                      "Error: number of features should be equal to factor count");
            for (int featureId = 0; featureId < FeatureValues.ysize(); ++featureId) {
                if (IgnoreFeatures.count(featureId) == 0) {
                    auto& featureColumn = FeatureValues[featureId];
                    featureColumn[GetLineIdx(localIdx)] = features[featureId];
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

        void AddSubgroupId(ui32 localIdx, ui32 groupId) override {
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

        int GetDocCount() const override {
            return DataProvider.Targets.size();
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
                    if (CatFeatureIds.has(featureId)) {
                        FeaturesManager.RegisterDataProviderCatFeature(featureId);
                    } else {
                        FeaturesManager.RegisterDataProviderFloatFeature(featureId);
                    }
                }
            }
        }

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
        TVector<TVector<float>> FeatureValues;
        TSet<ui32> IgnoreFeatures;
        TVector<TString> FeatureNames;
        TSet<int> CatFeatureIds;

        TVector<float> ClassesWeights;
        TVector<TPair> Pairs;
    };
}
