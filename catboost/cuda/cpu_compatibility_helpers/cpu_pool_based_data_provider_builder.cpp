#include "cpu_pool_based_data_provider_builder.h"

namespace NCatboostCuda {
    void TCpuPoolBasedDataProviderBuilder::Finish(ui32 binarizationThreads) {
        DataProvider.Order.resize(DataProvider.Targets.size());
        std::iota(DataProvider.Order.begin(),
                  DataProvider.Order.end(), 0);

        const ui32 featureCount = Pool.Docs.GetFactorsCount();
        const ui64 docCount = Pool.Docs.GetDocCount();

        DataProvider.CatFeatureIds.insert(Pool.CatFeatures.begin(), Pool.CatFeatures.end());
        if (!IsTest) {
            TOnCpuGridBuilderFactory gridBuilderFactory;
            FeaturesManager.SetTargetBorders(TBordersBuilder(gridBuilderFactory,
                                                             DataProvider.GetTargets())(
                FeaturesManager.GetTargetBinarizationDescription()));
        }
        if (!IsTest) {
            RegisterFeaturesInFeatureManager(DataProvider.CatFeatureIds);
        }

        TAdaptiveLock binarizationLock;
        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(binarizationThreads - 1);
        if (!Pool.FeatureId.empty()) {
            CB_ENSURE(Pool.FeatureId.size() == featureCount);
            DataProvider.FeatureNames = Pool.FeatureId;
        } else {
            DataProvider.FeatureNames.clear();
            for (ui32 i = 0; i < featureCount; ++i) {
                DataProvider.FeatureNames.push_back(ToString(i));
            }
        }

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
                    FeaturesManager.GetOrCreateNanMode(*holder);
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
                    auto borders = BuildBorders(Pool.Docs.Factors[featureId], featureId /*seed */, config);
                    with_lock (binarizationLock) {
                        FeaturesManager.SetFloatFeatureBordersForDataProviderId(featureId, std::move(borders));
                    }
                }
            });
        }

        DataProvider.BuildIndicesRemap();

        if (ClassesWeights.size()) {
            Reweight(DataProvider.Targets, ClassesWeights, &DataProvider.Weights);
        }

        if (Pool.Pairs.size()) {
            //they are local, so we don't need shuffle
            CB_ENSURE(DataProvider.HasQueries(), "Error: for GPU pairwise learning you should provide query id column. Query ids will be used to split data between devices and for dynamic boosting learning scheme.");
            DataProvider.FillQueryPairs(Pool.Pairs);
        }
    }
}
