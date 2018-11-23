#include "cpu_pool_based_data_provider_builder.h"

#include <catboost/libs/quantization/grid_creator.h>
#include <catboost/libs/quantization/utils.h>
#include <catboost/libs/pairs/util.h>


namespace NCatboostCuda {
    void TCpuPoolBasedDataProviderBuilder::Finish(ui32 binarizationThreads) {
        DataProvider.Order.resize(DataProvider.Targets.size());
        std::iota(DataProvider.Order.begin(),
                  DataProvider.Order.end(), 0);

        const ui32 featureCount = Pool.Docs.GetEffectiveFactorCount();
        const ui64 docCount = Pool.Docs.GetDocCount();

        if (TargetHelper) {
            TargetHelper->MakeTargetAndWeights(!IsTest, &DataProvider.Targets, &DataProvider.Weights);
            DataProvider.ClassificationTargetHelper = TargetHelper;
        }

        DataProvider.CatFeatureIds.insert(Pool.CatFeatures.begin(), Pool.CatFeatures.end());
        if (!IsTest) {
            NCB::TOnCpuGridBuilderFactory gridBuilderFactory;
            FeaturesManager.SetTargetBorders(NCB::TBordersBuilder(gridBuilderFactory,
                                                                  DataProvider.GetTargets())(FeaturesManager.GetTargetBinarizationDescription()));
        }
        if (!IsTest) {
            RegisterFeaturesInFeatureManager(DataProvider.CatFeatureIds);
        }

        TAdaptiveLock binarizationLock;
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(binarizationThreads - 1);

        CB_ENSURE(Pool.FeatureId.size() == featureCount);
        DataProvider.FeatureNames = Pool.FeatureId;

        NPar::ParallelFor(executor, 0, featureCount, [&](int featureId) {
            if (IgnoreFeatures.has(featureId)) {
                return;
            }
            if (DataProvider.CatFeatureIds.has(featureId)) {
                const bool shouldSkip = IsTest && CatFeaturesPerfectHashHelper.GetUniqueValues(featureId) == 0;

                if (!shouldSkip) {
                    auto bins = CatFeaturesPerfectHashHelper.UpdatePerfectHashAndBinarize(featureId,
                                                                                          ~Pool.Docs.Factors[featureId],
                                                                                          docCount);
                    Y_UNUSED(bins);
                }
            }
        });

        for (ui32 featureId = 0; featureId < featureCount; ++featureId) {
            if (IgnoreFeatures.has(featureId)) {
                continue;
            }
            TString featureName = !DataProvider.FeatureNames.empty() ? DataProvider.FeatureNames[featureId]
                                                                     : ToString(featureId);

            if (DataProvider.CatFeatureIds.has(featureId)) {
                const ui32 uniqueValues = CatFeaturesPerfectHashHelper.GetUniqueValues(featureId);
                if (uniqueValues > 1) {
                    DataProvider.Features.push_back(MakeHolder<TExternalCatFeatureValuesHolder>(featureId,
                                                                                                docCount,
                                                                                                ~Pool.Docs.Factors[featureId],
                                                                                                uniqueValues,
                                                                                                FeaturesManager,
                                                                                                featureName));
                }
            } else {
                auto holder = MakeHolder<TFloatValuesHolder>(featureId,
                                                             ~Pool.Docs.Factors[featureId],
                                                             docCount,
                                                             featureName);
                if (!IsTest) {
                    FeaturesManager.GetOrComputeNanMode(*holder);
                }
                DataProvider.Features.push_back(std::move(holder));
            }
        }

        if (!IsTest) {
            NPar::ParallelFor(executor, 0, featureCount, [&](int featureId) {
                if (IgnoreFeatures.has(featureId) || DataProvider.CatFeatureIds.has(featureId)) {
                    return;
                }

                if (!FeaturesManager.HasFloatFeatureBordersForDataProviderFeature(featureId)) {
                    NCatboostOptions::TBinarizationOptions config = FeaturesManager.GetFloatFeatureBinarization();
                    ENanMode nanMode = FeaturesManager.GetNanMode(featureId);
                    config.NanMode = nanMode;
                    auto borders = NCB::BuildBorders(Pool.Docs.Factors[featureId], featureId /*seed */, config);
                    with_lock (binarizationLock) {
                        FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId, std::move(borders));
                    }
                }
            });
        }

        DataProvider.BuildIndicesRemap();


        if (Pool.Pairs.size()) {
            //they are local, so we don't need shuffle
            CB_ENSURE(DataProvider.HasQueries(), "Error: for GPU pairwise learning you should provide query id column. Query ids will be used to split data between devices and for dynamic boosting learning scheme.");
            DataProvider.FillQueryPairs(Pool.Pairs);
        }
    }

    TCpuPoolBasedDataProviderBuilder::TCpuPoolBasedDataProviderBuilder(TBinarizedFeaturesManager& featureManager,
                                                                       bool hasQueries, const TPool& pool, bool isTest,
                                                                       const NCatboostOptions::TLossDescription& lossFunctionDescription,
                                                                       ui64 seed, TDataProvider& dst)
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

        if (IsPairLogit(lossFunctionDescription.GetLossFunction()) && DataProvider.GetPairs().empty()) {
            CB_ENSURE(
                    !DataProvider.GetTargets().empty(),
                    "Pool labels are not provided. Cannot generate pairs."
            );

            CATBOOST_WARNING_LOG << "No pairs provided for learn dataset. "
                                 << "Trying to generate pairs using dataset labels." << Endl;

            const auto& pairs = GeneratePairLogitPairs(
                    DataProvider.QueryIds,
                    DataProvider.Targets,
                    NCatboostOptions::GetMaxPairCount(lossFunctionDescription),
                    seed);
            DataProvider.FillQueryPairs(pairs);
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
}
