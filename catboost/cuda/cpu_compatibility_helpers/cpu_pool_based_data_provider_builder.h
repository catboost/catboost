#pragma once

#include "externel_cat_values_holder.h"
#include <catboost/libs/data/pool.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/data/data_provider.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/data/cat_feature_perfect_hash_helper.h>
#include <catboost/cuda/data/grid_creator.h>
#include <catboost/cuda/cuda_util/cpu_random.h>

namespace NCatboostCuda
{
    class TCpuPoolBasedDataProviderBuilder
    {
    public:

        TCpuPoolBasedDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
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
            DataProvider.Weights = pool.Docs.Weight;

            const ui32 numSamples = pool.Docs.GetDocCount();

            DataProvider.QueryIds.resize(numSamples);
            for (ui32 i = 0; i < DataProvider.QueryIds.size(); ++i)
            {
                DataProvider.QueryIds[i] = i;
            }

            DataProvider.Baseline.resize(pool.Docs.Baseline.size());
            for (ui32 i = 0; i < pool.Docs.Baseline.size(); ++i)
            {
                auto& baseline = DataProvider.Baseline[i];
                auto& baselineSrc = pool.Docs.Baseline[i];
                baseline.resize(baselineSrc.size());
                for (ui32 j = 0; j < baselineSrc.size(); ++j)
                {
                    baseline[j] = baselineSrc[j];
                }
            }
        }

        template<class TContainer>
        TCpuPoolBasedDataProviderBuilder& AddIgnoredFeatures(const TContainer& container)
        {
            for (auto& f : container)
            {
                IgnoreFeatures.insert(f);
            }
            return *this;
        }

        void Finish(ui32 binarizationThreads)
        {
            DataProvider.Order.resize(DataProvider.Targets.size());
            std::iota(DataProvider.Order.begin(),
                      DataProvider.Order.end(), 0);

            const ui32 featureCount = Pool.Docs.GetFactorsCount();
            const ui64 docCount = Pool.Docs.GetDocCount();


            DataProvider.CatFeatureIds.insert(Pool.CatFeatures.begin(), Pool.CatFeatures.end());
            if (!IsTest)
            {
                TOnCpuGridBuilderFactory gridBuilderFactory;
                FeaturesManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
                                                                 DataProvider.GetTargets())(
                        FeaturesManager.GetTargetBinarizationDescription()));
            }
            if (!IsTest)
            {
                RegisterFeaturesInFeatureManager(DataProvider.CatFeatureIds);
            }

            TAdaptiveLock binarizationLock;
            NPar::TLocalExecutor executor;
            executor.RunAdditionalThreads(binarizationThreads - 1);
            if (Pool.FeatureId.size())
            {
                CB_ENSURE(Pool.FeatureId.size() == featureCount);
                DataProvider.FeatureNames = Pool.FeatureId;
            } else {
                DataProvider.FeatureNames.clear();
                for (ui32 i = 0; i < featureCount; ++i) {
                    DataProvider.FeatureNames.push_back(ToString(i));
                }
            }

            NPar::ParallelFor(executor, 0, featureCount, [&](int featureId)
            {
                if (IgnoreFeatures.has(featureId))
                {
                    return;
                }
                if (DataProvider.CatFeatureIds.has(featureId))
                {
                    const bool shouldSkip = IsTest && CatFeaturesPerfectHashHelper.GetUniqueValues(featureId) == 0;

                    if (!shouldSkip)
                    {
                        auto bins = CatFeaturesPerfectHashHelper.UpdatePerfectHashAndBinarize(featureId,
                                                                                              ~Pool.Docs.Factors[featureId],
                                                                                              docCount);
                        Y_UNUSED(bins);
                    }
                } else
                {
                    if (!FeaturesManager.HasFloatFeatureBordersForDataProviderFeature(featureId))
                    {
                        const auto& config = FeaturesManager.GetDefaultFloatFeatureBinarizationDescription();
                        auto borders = BuildBorders(Pool.Docs.Factors[featureId], featureId /*seed */, config);
                        {
                            TGuard<TAdaptiveLock> guard(binarizationLock);
                            FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId, std::move(borders));
                        }
                    }
                }
            });

            for (ui32 featureId = 0; featureId < featureCount; ++featureId)
            {
                if (IgnoreFeatures.has(featureId)) {
                    continue;
                }
                TString featureName = !DataProvider.FeatureNames.empty() ? DataProvider.FeatureNames[featureId]
                                                                       : ToString(featureId);

                if (DataProvider.CatFeatureIds.has(featureId))
                {
                    const ui32 uniqueValues = CatFeaturesPerfectHashHelper.GetUniqueValues(featureId);
                    if (uniqueValues > 1)
                    {
                        DataProvider.Features.push_back(MakeHolder<TExternalCatFeatureValuesHolder>(featureId,
                                                                                                    docCount,
                                                                                                    ~Pool.Docs.Factors[featureId],
                                                                                                    uniqueValues,
                                                                                                    FeaturesManager,
                                                                                                    featureName));
                    }
                } else {
                    DataProvider.Features.push_back(MakeHolder<TFloatValuesHolder>(featureId,
                                                                                   ~Pool.Docs.Factors[featureId],
                                                                                   docCount,
                                                                                   featureName));
                }
            }

            GroupQueries(DataProvider.QueryIds,
                         &DataProvider.Queries);
            DataProvider.BuildIndicesRemap();
        }

    private:

        void RegisterFeaturesInFeatureManager(const yset<int>& catFeatureIds) const
        {
            const ui32 factorsCount = Pool.Docs.GetFactorsCount();
            for (ui32 featureId = 0; featureId < factorsCount; ++featureId)
            {
                if (!FeaturesManager.IsKnown(featureId))
                {
                    if (catFeatureIds.has(featureId))
                    {
                        FeaturesManager.RegisterDataProviderCatFeature(featureId);
                    } else
                    {
                        FeaturesManager.RegisterDataProviderFloatFeature(featureId);
                    }
                }
            }
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        TDataProvider& DataProvider;
        const TPool& Pool;
        bool IsTest ;
        TCatFeaturesPerfectHashHelper CatFeaturesPerfectHashHelper;
        yset<ui32> IgnoreFeatures;
    };

}
