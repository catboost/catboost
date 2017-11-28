#include "cpu_pool_based_data_provider_builder.h"


namespace NCatboostCuda
{

    void TCpuPoolBasedDataProviderBuilder::Finish(ui32 binarizationThreads)
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
        } else
        {
            DataProvider.FeatureNames.clear();
            for (ui32 i = 0; i < featureCount; ++i)
            {
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
                    const auto& config = FeaturesManager.GetFloatFeatureBinarization();
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
            if (IgnoreFeatures.has(featureId))
            {
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
            } else
            {
                DataProvider.Features.push_back(MakeHolder<TFloatValuesHolder>(featureId,
                                                                               ~Pool.Docs.Factors[featureId],
                                                                               docCount,
                                                                               featureName));
            }
        }

        GroupQueries(DataProvider.QueryIds,
                     &DataProvider.Queries);
        DataProvider.BuildIndicesRemap();

        if (ClassesWeights.size()) {
            Reweight(DataProvider.Targets, ClassesWeights, &DataProvider.Weights);
        }

    }
}
