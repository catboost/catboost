#pragma once

#include "externel_cat_values_holder.h"
#include <catboost/libs/data/pool.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/cat_feature_perfect_hash_helper.h>
#include <catboost/cuda/data/grid_creator.h>
#include <catboost/cuda/utils/cpu_random.h>

namespace NCatboostCuda {
    class TCpuPoolBasedDataProviderBuilder {
    public:
        TCpuPoolBasedDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
                                         bool hasQueries,
                                         const TPool& pool,
                                         bool isTest,
                                         TDataProvider& dst)
            : FeaturesManager(featureManager)
            , DataProvider(dst)
            , Pool(pool)
            , IsTest(isTest)
            , CatFeaturesPerfectHashHelper(FeaturesManager)
        {
            DataProvider.Targets = pool.Docs.Target;
            if (pool.Docs.Weight.size()) {
                DataProvider.Weights = pool.Docs.Weight;
            } else {
                DataProvider.Weights.resize(pool.Docs.Target.size(), 1.0f);
            }

            const ui32 numSamples = pool.Docs.GetDocCount();

            if (hasQueries) {
                DataProvider.QueryIds.resize(numSamples);
                for (ui32 i = 0; i < DataProvider.QueryIds.size(); ++i) {
                    DataProvider.QueryIds[i] = pool.Docs.QueryId[i];
                }
            }

            DataProvider.Baseline.resize(pool.Docs.Baseline.size());
            for (ui32 i = 0; i < pool.Docs.Baseline.size(); ++i) {
                auto& baseline = DataProvider.Baseline[i];
                auto& baselineSrc = pool.Docs.Baseline[i];
                baseline.resize(baselineSrc.size());
                for (ui32 j = 0; j < baselineSrc.size(); ++j) {
                    baseline[j] = baselineSrc[j];
                }
            }

            DataProvider.SubgroupIds = Pool.Docs.SubgroupId;
            DataProvider.Timestamp = Pool.Docs.Timestamp;
        }

        template <class TContainer>
        TCpuPoolBasedDataProviderBuilder& AddIgnoredFeatures(const TContainer& container) {
            for (const auto& f : container) {
                IgnoreFeatures.insert(f);
            }
            return *this;
        }

        TCpuPoolBasedDataProviderBuilder& SetClassesWeights(const TVector<float>& weights) {
            ClassesWeights = weights;
            return *this;
        }

        void Finish(ui32 binarizationThreads);

    private:
        void RegisterFeaturesInFeatureManager(const TSet<int>& catFeatureIds) const {
            const ui32 factorsCount = Pool.Docs.GetFactorsCount();
            for (ui32 featureId = 0; featureId < factorsCount; ++featureId) {
                if (!FeaturesManager.IsKnown(featureId)) {
                    if (catFeatureIds.has(featureId)) {
                        FeaturesManager.RegisterDataProviderCatFeature(featureId);
                    } else {
                        FeaturesManager.RegisterDataProviderFloatFeature(featureId);
                    }
                }
            }
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        TDataProvider& DataProvider;
        const TPool& Pool;
        bool IsTest;
        TCatFeaturesPerfectHashHelper CatFeaturesPerfectHashHelper;
        TSet<ui32> IgnoreFeatures;
        TVector<float> ClassesWeights;
    };

}
