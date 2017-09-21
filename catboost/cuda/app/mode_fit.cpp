#include "options_binding.h"
#include "application_options.h"

#include <catboost/cuda/data/load_data.h>
#include <catboost/cuda/methods/boosting.h>
#include <catboost/cuda/targets/mse.h>
#include <catboost/cuda/methods/oblivious_tree.h>
#include <catboost/cuda/targets/cross_entropy.h>
#include <catboost/cuda/data/cat_feature_binarization_helpers.h>
#include <catboost/cuda/cpu_compatibility_helpers/model_converter.h>
#include <catboost/cuda/cpu_compatibility_helpers/full_model_saver.h>
#include <catboost/libs/model/model.h>
#include <library/getopt/small/last_getopt.h>
#include <util/system/fs.h>

template <template <class TMapping, class> class TTargetTemplate, NCudaLib::EPtrType CatFeaturesStoragePtrType>
inline THolder<TAdditiveModel<TObliviousTreeModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                          const TBoostingOptions& boostingOptions,
                                                          const TObliviousTreeLearnerOptions& treeOptions,
                                                          const TTargetOptions& targetOptions,
                                                          const TDataProvider& learn,
                                                          const TDataProvider* test,
                                                          TRandom& random) {
    using TTaskDataSet = TDataSet<CatFeaturesStoragePtrType>;
    using TTarget = TTargetTemplate<NCudaLib::TMirrorMapping, TTaskDataSet>;

    TObliviousTree tree(featureManager, treeOptions);
    TDontLookAheadBoosting<TTargetTemplate, TObliviousTree, CatFeaturesStoragePtrType> boosting(featureManager,
                                                                                                boostingOptions,
                                                                                                targetOptions,
                                                                                                random,
                                                                                                tree);
    boosting.SetDataProvider(learn, test);

    using TMetricPrinter = TMetricLogger<TTarget, TObliviousTreeModel>;
    TIterationLogger<TTarget, TObliviousTreeModel> iterationPrinter;
    boosting.RegisterLearnListener(iterationPrinter);

    THolder<TMetricPrinter> learnPrinter;
    THolder<TMetricPrinter> testPrinter;

    if (boostingOptions.IsCalcScores()) {
        learnPrinter.Reset(new TMetricPrinter("Learn score: ", boostingOptions.GetLearnErrorLogPath()));
        if (test) {
            testPrinter.Reset(new TMetricPrinter("Test score: ", boostingOptions.GetTestErrorLogPath()));
        }
    }

    if (learnPrinter) {
        boosting.RegisterLearnListener(*learnPrinter);
    }

    if (testPrinter) {
        boosting.RegisterTestListener(*testPrinter);
    }

    return boosting.Run();
}

template <template <class TMapping, class> class TTargetTemplate>
inline THolder<TAdditiveModel<TObliviousTreeModel>> Train(TBinarizedFeaturesManager& featureManager,
                                                          const TBoostingOptions& boostingOptions,
                                                          const TObliviousTreeLearnerOptions& treeOptions,
                                                          const TTargetOptions& targetOptions,
                                                          const TDataProvider& learn,
                                                          const TDataProvider* test,
                                                          TRandom& random,
                                                          bool storeCatFeaturesInPinnedMemory) {
    if (storeCatFeaturesInPinnedMemory) {
        return Train<TTargetTemplate, NCudaLib::CudaHost>(featureManager, boostingOptions, treeOptions, targetOptions, learn, test, random);
    } else {
        return Train<TTargetTemplate, NCudaLib::CudaDevice>(featureManager, boostingOptions, treeOptions, targetOptions, learn, test, random);
    }
};

inline void MakeFullModel(const TString& coreModelPath,
                          const TPoolLoadOptions& poolLoadOptions,
                          ui32 numThreads,
                          const TString& fullModelPath) {
    TPool pool;

    ReadPool(poolLoadOptions.GetColumnDescriptionName(),
             poolLoadOptions.GetFeaturesFilename(),
             numThreads,
             false,
             poolLoadOptions.GetDelimiter(),
             poolLoadOptions.HasHeader(),
             poolLoadOptions.GetClassNames(),
             &pool);

    TCoreModel coreModel;
    {
        TIFStream modelInput(coreModelPath);
        coreModel.Load(&modelInput);
    }

    TCoreModelToFullModelConverter converter(coreModel, pool);
    converter.Save(fullModelPath);
}

int mode_fit(const int argc, const char** argv) {
    TFeatureManagerOptions featureManagerOptions;
    TPoolLoadOptions loadOptions;
    TObliviousTreeLearnerOptions treeConfig;
    TBoostingOptions boostingOptions;
    TTargetOptions targetOptions;
    TApplicationOptions applicationOptions;
    TString resultModelPath = "catboost.bin";

    {
        NLastGetopt::TOpts options = NLastGetopt::TOpts::Default();

        TOptionsBinder<TApplicationOptions>::Bind(applicationOptions, options);
        TOptionsBinder<TFeatureManagerOptions>::Bind(featureManagerOptions, options);
        TOptionsBinder<TPoolLoadOptions>::Bind(loadOptions, options);
        TOptionsBinder<TObliviousTreeLearnerOptions>::Bind(treeConfig, options);
        TOptionsBinder<TBoostingOptions>::Bind(boostingOptions, options);
        TOptionsBinder<TTargetOptions>::Bind(targetOptions, options);

        options
            .AddLongOption('m', "model-file")
            .StoreResult(&resultModelPath);

        NLastGetopt::TOptsParseResult parse(&options, argc, argv);
        Y_UNUSED(parse);
    }

    TString coreModelPath = TStringBuilder() << resultModelPath << ".core";

    if (targetOptions.GetTargetType() == ETargetFunction::RMSE) {
        treeConfig.SetLeavesEstimationIterations(1);
    }
    if (targetOptions.GetTargetType() == ETargetFunction::CrossEntropy) {
        featureManagerOptions.SetTargetBinarization(2);
    }

    {
        NPar::LocalExecutor().RunAdditionalThreads(applicationOptions.GetNumThreads());
        NCudaLib::SetApplicationConfig(applicationOptions.GetCudaApplicationConfig());
        StartCudaManager();
        {
            if (NCudaLib::GetCudaManager().GetDeviceCount() > 1) {
                NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaDevice, NCudaLib::CudaHost>();
                NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaDevice, NCudaLib::CudaDevice>();
                NCudaLib::GetLatencyAndBandwidthStats<NCudaLib::CudaHost, NCudaLib::CudaDevice>();
            }
            auto& profiler = NCudaLib::GetCudaManager().GetProfiler();

            if (applicationOptions.IsProfile()) {
                profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::ImplicitLabelSync);
            } else {
                profiler.SetDefaultProfileMode(NCudaLib::EProfileMode::NoProfile);
            }

            TBinarizedFeaturesManager featuresManager(featureManagerOptions);
            TDataProvider dataProvider;
            THolder<TDataProvider> testProvider;
            const auto& catFeatureBinarizationTempFilename = loadOptions.GetCatFeatureBinarizationTempName();
            {
                MATRIXNET_INFO_LOG << "Loading data..." << Endl;

                TOnGpuGridBuilderFactory gridBuilderFactory;

                TDataProviderBuilder dataProviderBuilder(featuresManager,
                                                         gridBuilderFactory,
                                                         dataProvider);

                dataProviderBuilder
                    .AddIgnoredFeatures(loadOptions.GetIgnoredFeatures())
                    .SetShuffleFlag(boostingOptions.HasTime() ? false : true);

                {
                    auto loadTimeGuard = profiler.Profile("Load learn data");
                    ReadPool(loadOptions.GetColumnDescriptionName(),
                             loadOptions.GetFeaturesFilename(),
                             applicationOptions.GetNumThreads(),
                             true,
                             loadOptions.GetDelimiter(),
                             loadOptions.HasHeader(),
                             loadOptions.GetClassNames(),
                             &dataProviderBuilder);
                }

                if (loadOptions.GetTestFilename()) {
                    MATRIXNET_INFO_LOG << "Loading test..." << Endl;
                    auto loadTimeGuard = profiler.Profile("Load test data");

                    testProvider.Reset(new TDataProvider());
                    TDataProviderBuilder testBuilder(featuresManager,
                                                     gridBuilderFactory,
                                                     *testProvider,
                                                     true);
                    testBuilder.AddIgnoredFeatures(loadOptions.GetIgnoredFeatures())
                        .SetShuffleFlag(false);

                    {
                        TDataProviderBuilder::TSimpleCatFeatureBinarizationInfo info;
                        dataProviderBuilder.MoveBinarizationTo(info);
                        testBuilder.SetExistingCatFeaturesBinarization(std::move(info));
                    }

                    ReadPool(loadOptions.GetColumnDescriptionName(),
                             loadOptions.GetTestFilename(),
                             applicationOptions.GetNumThreads(),
                             true,
                             loadOptions.GetDelimiter(),
                             loadOptions.HasHeader(),
                             loadOptions.GetClassNames(),
                             &testBuilder);

                    {
                        TDataProviderBuilder::TSimpleCatFeatureBinarizationInfo info;
                        testBuilder.MoveBinarizationTo(info);
                        TCatFeatureBinarizationHelpers::SaveCatFeatureBinarization(info,
                                                                                   catFeatureBinarizationTempFilename);
                    }
                } else {
                    TDataProviderBuilder::TSimpleCatFeatureBinarizationInfo info;
                    dataProviderBuilder.MoveBinarizationTo(info);
                    TCatFeatureBinarizationHelpers::SaveCatFeatureBinarization(info,
                                                                               catFeatureBinarizationTempFilename);
                }
            }

            if (featureManagerOptions.IsCtrTypeEnabled(ECtrType::FeatureFreq)) {
                yvector<float> prior = {0.5f};
                featuresManager.EnableCtrType(ECtrType::FeatureFreq, prior);
            }

            const bool isFloatTargetMeanCtrEnabled = (!featureManagerOptions.IsCustomCtrTypes() && targetOptions.GetTargetType() == ETargetFunction::RMSE) || featureManagerOptions.IsCtrTypeEnabled(ECtrType::FloatTargetMeanValue);
            if (isFloatTargetMeanCtrEnabled) {
                yvector<float> prior = {0.0, 3.0};
                featuresManager.EnableCtrType(ECtrType::FloatTargetMeanValue, prior);
            }

            if (targetOptions.GetTargetType() == ETargetFunction::RMSE) {
                yvector<float> prior = {0.5f};
                if (featureManagerOptions.IsCtrTypeEnabled(ECtrType::Borders)) {
                    featuresManager.EnableCtrType(ECtrType::Borders, prior);
                }
                if (featureManagerOptions.IsCtrTypeEnabled(ECtrType::Buckets)) {
                    featuresManager.EnableCtrType(ECtrType::Buckets, prior);
                }
            } else {
                if (featureManagerOptions.IsCtrTypeEnabled(ECtrType::Borders)) {
                    MATRIXNET_WARNING_LOG << "Warning: borders ctr aren't supported for target " << targetOptions.GetTargetType() << ". Change type for buckets" << Endl;
                    featureManagerOptions.DisableCtrType(ECtrType::Borders);
                    featureManagerOptions.EnableCtrType(ECtrType::Buckets);
                }
                if (featureManagerOptions.IsCtrTypeEnabled(ECtrType::Buckets)) {
                    yvector<float> prior = {0.5, 0.5};
                    featuresManager.EnableCtrType(ECtrType::Buckets, prior);

                    prior = {1.0, 0.0};
                    featuresManager.EnableCtrType(ECtrType::Buckets, prior);

                    prior = {0.0, 1.0};
                    featuresManager.EnableCtrType(ECtrType::Buckets, prior);
                }
            }

            //don't make several permutations in matrixnet-like mode if we don't have ctrs
            {
                bool hasCtrs = false;
                for (auto catFeature : featuresManager.GetCatFeatureIds()) {
                    if (featuresManager.UseForCtr(catFeature) || featuresManager.UseForTreeCtr(catFeature)) {
                        hasCtrs = true;
                        break;
                    }
                }
                if (boostingOptions.DisableDontLookAhead() && !hasCtrs && boostingOptions.GetPermutationCount() > 1) {
                    MATRIXNET_INFO_LOG << "No catFeatures for ctrs found and don't look ahead is disabled. Fallback to one permutation" << Endl;
                    boostingOptions.SetPermutationCount(1);
                }
            }

            TRandom random(applicationOptions.GetSeed());

            THolder<TAdditiveModel<TObliviousTreeModel>> model;
            const bool storeCatFeaturesInPinnedMemory = boostingOptions.UseCpuRamForCatFeaturesDataSet();

            switch (targetOptions.GetTargetType()) {
                case ETargetFunction::RMSE: {
                    model = Train<TL2>(featuresManager, boostingOptions, treeConfig, targetOptions, dataProvider,
                                       testProvider.Get(), random, storeCatFeaturesInPinnedMemory);
                    break;
                }
                case ETargetFunction::CrossEntropy:
                case ETargetFunction::Logloss: {
                    model = Train<TCrossEntropy>(featuresManager, boostingOptions, treeConfig, targetOptions,
                                                 dataProvider, testProvider.Get(), random,
                                                 storeCatFeaturesInPinnedMemory);
                    break;
                }
            }

            auto coreModel = ConvertToCoreModel(featuresManager,
                                                dataProvider,
                                                catFeatureBinarizationTempFilename,
                                                *model);

            TOFStream modelOutput(coreModelPath);
            coreModel.Save(&modelOutput);

            if (NFs::Exists(catFeatureBinarizationTempFilename)) {
                NFs::Remove(catFeatureBinarizationTempFilename);
            }
        }
        StopCudaManager();
    }

    MakeFullModel(coreModelPath,
                  loadOptions,
                  applicationOptions.GetNumThreads(),
                  resultModelPath);

    return 0;
}
