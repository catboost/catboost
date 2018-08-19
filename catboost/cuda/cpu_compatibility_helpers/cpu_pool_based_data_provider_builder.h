#pragma once

#include "externel_cat_values_holder.h"
#include <catboost/libs/data/pool.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/cat_feature_perfect_hash_helper.h>
#include <catboost/cuda/data/grid_creator.h>
#include <catboost/libs/helpers/cpu_random.h>

namespace NCatboostCuda {
    class TCpuPoolBasedDataProviderBuilder {
    public:
        TCpuPoolBasedDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
                                         bool hasQueries,
                                         const TPool& pool,
                                         bool isTest,
                                         TDataProvider& dst);

        template <class TContainer>
        TCpuPoolBasedDataProviderBuilder& AddIgnoredFeatures(const TContainer& container) {
            for (const auto& f : container) {
                IgnoreFeatures.insert(f);
            }
            return *this;
        }

        TCpuPoolBasedDataProviderBuilder& SetTargetHelper(TSimpleSharedPtr<TClassificationTargetHelper> targetHelper) {
            TargetHelper = targetHelper;
            return *this;
        }

        void Finish(ui32 binarizationThreads);

    private:
        void RegisterFeaturesInFeatureManager(const TSet<int>& catFeatureIds) const {
            const ui32 factorsCount = Pool.Docs.GetEffectiveFactorCount();
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
        TSimpleSharedPtr<TClassificationTargetHelper> TargetHelper;
    };

}
