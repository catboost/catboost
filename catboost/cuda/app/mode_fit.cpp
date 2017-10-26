#include "options_binding.h"

#include <catboost/cuda/data/load_data.h>
#include <catboost/cuda/train_lib/train.h>
#include <catboost/cuda/train_lib/model_helpers.h>
#include <library/getopt/small/last_getopt.h>

using namespace NCatboostCuda;

int mode_fit(const int argc, const char* argv[]) {
    TTrainCatBoostOptions catBoostOptions;
    TPoolLoadOptions poolLoadOptions;

    {
        NLastGetopt::TOpts options = NLastGetopt::TOpts::Default();

        TOptionsBinder<TTrainCatBoostOptions>::Bind(catBoostOptions, options);
        TOptionsBinder<TPoolLoadOptions>::Bind(poolLoadOptions, options);
        NLastGetopt::TOptsParseResult parse(&options, argc, argv);
        Y_UNUSED(parse);
    }

    const auto& resultModelPath = catBoostOptions.OutputFilesOptions.GetResultModelPath();
    TString coreModelPath = TStringBuilder() << resultModelPath << ".core";

    const int numThreads = catBoostOptions.ApplicationOptions.GetNumThreads();
    {
        NPar::LocalExecutor().RunAdditionalThreads(numThreads - 1);
        TBinarizedFeaturesManager featuresManager(catBoostOptions.FeatureManagerOptions);
        TDataProvider dataProvider;
        THolder<TDataProvider> testProvider;
        {
            MATRIXNET_INFO_LOG << "Loading data..." << Endl;

            TDataProviderBuilder dataProviderBuilder(featuresManager,
                                                     dataProvider,
                                                     false,
                                                     numThreads);

            dataProviderBuilder
                .AddIgnoredFeatures(catBoostOptions.FeatureManagerOptions.GetIgnoredFeatures())
                .SetShuffleFlag(!catBoostOptions.BoostingOptions.HasTime());

            {
                ReadPool(poolLoadOptions.GetColumnDescriptionName(),
                         poolLoadOptions.GetFeaturesFilename(),
                         "",
                         numThreads,
                         true,
                         poolLoadOptions.GetDelimiter(),
                         poolLoadOptions.HasHeader(),
                         poolLoadOptions.GetClassNames(),
                         &dataProviderBuilder);
            }

            if (poolLoadOptions.GetTestFilename()) {
                MATRIXNET_INFO_LOG << "Loading test..." << Endl;
                testProvider.Reset(new TDataProvider());
                TDataProviderBuilder testBuilder(featuresManager,
                                                 *testProvider,
                                                 true,
                                                 numThreads);
                testBuilder
                    .AddIgnoredFeatures(catBoostOptions.FeatureManagerOptions.GetIgnoredFeatures())
                    .SetShuffleFlag(false);

                ReadPool(poolLoadOptions.GetColumnDescriptionName(),
                         poolLoadOptions.GetTestFilename(),
                         "",
                         numThreads,
                         true,
                         poolLoadOptions.GetDelimiter(),
                         poolLoadOptions.HasHeader(),
                         poolLoadOptions.GetClassNames(),
                         &testBuilder);
            }
        }
        featuresManager.UnloadCatFeaturePerfectHashFromRam();
        UpdatePinnedMemorySizeOption(dataProvider, testProvider.Get(), featuresManager, catBoostOptions);
        UpdateOptionsAndEnableCtrTypes(catBoostOptions, featuresManager);
        {
            auto coreModel = TrainModel(catBoostOptions, dataProvider, testProvider.Get(), featuresManager);
            TOFStream modelOutput(coreModelPath);
            coreModel.Save(&modelOutput);
        }
    }

    MakeFullModel(coreModelPath,
                  poolLoadOptions,
                  numThreads,
                  resultModelPath);

    return 0;
}
