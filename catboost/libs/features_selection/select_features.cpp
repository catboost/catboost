#include "select_features.h"

#include "recursive_features_elimination.h"

#include <catboost/libs/data/borders_io.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/train_lib/dir_helper.h>
#include <catboost/libs/train_lib/options_helper.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/full_model_saver.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/private/libs/options/plain_options_helper.h>


using namespace NJson;
using namespace NCatboostOptions;


namespace NCB {
    static void CheckOptions(
        const TCatBoostOptions& catBoostOptions,
        const TFeaturesSelectOptions& featuresSelectOptions,
        const TDataProviders& pools
    ) {
        if (catBoostOptions.GetTaskType() == ETaskType::GPU) {
            CB_ENSURE(
                TTrainerFactory::Has(ETaskType::GPU),
                "Can't load GPU learning library. "
                "Module was not compiled or driver is incompatible with package. "
                "Please install latest NVDIA driver and check again");
        }

        const auto& featuresForSelect = featuresSelectOptions.FeaturesForSelect.Get();
        CB_ENSURE(featuresSelectOptions.NumberOfFeaturesToSelect.IsSet(), "You should specify the number of features to select");
        CB_ENSURE(featuresSelectOptions.NumberOfFeaturesToSelect.Get() > 0, "Number of features to select should be positive");
        CB_ENSURE(featuresForSelect.size() > 0, "You should specify features to select from");
        CB_ENSURE(
            static_cast<int>(featuresForSelect.size()) >= featuresSelectOptions.NumberOfFeaturesToSelect,
            "It is impossible to select " << featuresSelectOptions.NumberOfFeaturesToSelect << " features from " << featuresForSelect.size() << " features"
        );
        const ui32 featureCount = pools.Learn->MetaInfo.GetFeatureCount();
        for (const ui32 feature : featuresForSelect) {
            CB_ENSURE(feature < featureCount, "Tested feature " << feature << " is not present; dataset contains only " << featureCount << " features");
            CB_ENSURE(
                !pools.Learn->MetaInfo.FeaturesLayout->GetExternalFeatureMetaInfo(feature).IsIgnored,
                "Tested feature " << feature << " should not be ignored"
            );
        }
    }


    static TTrainingDataProviders QuantizePools(
        const TPoolLoadParams* poolLoadParams,
        const TOutputFilesOptions& outputFileOptions,
        const TDataProviders& pools,
        TCatBoostOptions* catBoostOptions,
        TLabelConverter* labelConverter,
        TRestorableFastRng64* rand,
        NPar::ILocalExecutor* executor
    ) {
        const auto learnFeaturesLayout = pools.Learn->MetaInfo.FeaturesLayout;

        // maybe PrepareQuantizationParameters ??
        // create here to possibly load borders
        auto quantizedFeaturesInfo = MakeIntrusive<TQuantizedFeaturesInfo>(
            *learnFeaturesLayout,
            catBoostOptions->DataProcessingOptions->IgnoredFeatures.Get(),
            catBoostOptions->DataProcessingOptions->FloatFeaturesBinarization.Get(),
            catBoostOptions->DataProcessingOptions->PerFloatFeatureQuantization.Get(),
            catBoostOptions->DataProcessingOptions->TextProcessingOptions.Get(),
            catBoostOptions->DataProcessingOptions->EmbeddingProcessingOptions.Get(),
            /*allowNansInTestOnly*/true
        );
        if (poolLoadParams && poolLoadParams->BordersFile) {
            LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
                poolLoadParams->BordersFile,
                quantizedFeaturesInfo.Get());
        }

        for (auto testPoolIdx : xrange(pools.Test.size())) {
            const auto& testPool = *pools.Test[testPoolIdx];
            if (testPool.GetObjectCount() == 0) {
                continue;
            }
            CheckCompatibleForApply(
                *learnFeaturesLayout,
                *testPool.MetaInfo.FeaturesLayout,
                TStringBuilder() << "test dataset #" << testPoolIdx);
        }

        TString tmpDir;
        if (outputFileOptions.AllowWriteFiles()) {
            NCB::NPrivate::CreateTrainDirWithTmpDirIfNotExist(outputFileOptions.GetTrainDir(), &tmpDir);
        }

        const bool haveLearnFeaturesInMemory = HaveFeaturesInMemory(
            *catBoostOptions,
            poolLoadParams ? MakeMaybe(poolLoadParams->LearnSetPath) : Nothing()
        );

        TTrainingDataProviders trainingData = GetTrainingData(
            pools,
            /*trainDataCanBeEmpty*/ false,
            /* borders */ Nothing(), // borders are already loaded to quantizedFeaturesInfo
            /*ensureConsecutiveIfDenseLearnFeaturesDataForCpu*/ haveLearnFeaturesInMemory,
            outputFileOptions.AllowWriteFiles(),
            tmpDir,
            quantizedFeaturesInfo,
            catBoostOptions,
            labelConverter,
            executor,
            rand
        );
        CheckConsistency(trainingData);

        return trainingData;
    }


    TFeaturesSelectionSummary SelectFeatures(
        TCatBoostOptions catBoostOptions,
        TOutputFilesOptions outputFileOptions,
        const TPoolLoadParams* poolLoadParams,
        const TFeaturesSelectOptions& featuresSelectOptions,
        const TDataProviders& pools,
        TFullModel* dstModel,
        const TVector<TEvalResult*>& evalResultPtrs,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory,
        NPar::ILocalExecutor* executor
    ) {
        TSetLogging inThisScope(catBoostOptions.LoggingLevel);

        CheckOptions(
            catBoostOptions,
            featuresSelectOptions,
            pools
        );

        TLabelConverter labelConverter;
        TRestorableFastRng64 rand(catBoostOptions.RandomSeed);

        auto trainingData = QuantizePools(
            poolLoadParams,
            outputFileOptions,
            pools,
            &catBoostOptions,
            &labelConverter,
            &rand,
            executor
        );

        const bool haveLearnFeaturesInMemory = HaveFeaturesInMemory(
            catBoostOptions,
            poolLoadParams ? MakeMaybe(poolLoadParams->LearnSetPath) : Nothing()
        );
        // TODO(ilyzhin) support distributed training with quantized pool
        CB_ENSURE(haveLearnFeaturesInMemory, "Features selection doesn't support distributed training with quantized pool yet.");

        THolder<TMasterContext> masterContext;

        if (catBoostOptions.SystemOptions->IsMaster()) {
            masterContext.Reset(new TMasterContext(catBoostOptions.SystemOptions));
            if (!haveLearnFeaturesInMemory) {
                TVector<TObjectsGrouping> testObjectsGroupings;
                for (const auto& testDataset : trainingData.Test) {
                    testObjectsGroupings.push_back(*(testDataset->ObjectsGrouping));
                }
                SetTrainDataFromQuantizedPools(
                    *poolLoadParams,
                    catBoostOptions,
                    TObjectsGrouping(*trainingData.Learn->ObjectsGrouping),
                    std::move(testObjectsGroupings),
                    *trainingData.Learn->MetaInfo.FeaturesLayout,
                    labelConverter,
                    &rand
                );
            } else {
                SetTrainDataFromMaster(
                    trainingData,
                    ParseMemorySizeDescription(catBoostOptions.SystemOptions->CpuUsedRamLimit.Get()),
                    executor
                );
            }
        }

        SetDataDependentDefaults(
            trainingData.Learn->MetaInfo,
            trainingData.Test.size() > 0 ?
                TMaybe<NCB::TDataMetaInfo>(trainingData.Test[0]->MetaInfo) :
                Nothing(),
            /*continueFromModel*/ false,
            /*continueFromProgress*/ false,
            &outputFileOptions,
            &catBoostOptions
        );

        InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric,
                                     &catBoostOptions.MetricOptions->EvalMetric);

        TFeaturesSelectionSummary summary = DoRecursiveFeaturesElimination(
            catBoostOptions,
            outputFileOptions,
            featuresSelectOptions,
            pools,
            labelConverter,
            trainingData,
            dstModel,
            evalResultPtrs,
            metricsAndTimeHistory,
            executor
        );

        const auto featuresNames = pools.Learn->MetaInfo.FeaturesLayout->GetExternalFeatureIds();
        for (auto featureIdx : summary.SelectedFeatures) {
            summary.SelectedFeaturesNames.push_back(featuresNames[featureIdx]);
        }
        for (auto featureIdx : summary.EliminatedFeatures) {
            summary.EliminatedFeaturesNames.push_back(featuresNames[featureIdx]);
        }

        return summary;
    }


    NJson::TJsonValue SelectFeatures(
        const NJson::TJsonValue& plainJsonParams,
        const TDataProviders& pools,
        TFullModel* dstModel,
        const TVector<TEvalResult*>& testApproxes,
        TMetricsAndTimeLeftHistory* metricsAndTimeHistory
    ) {
        NJson::TJsonValue catBoostJsonOptions;
        NJson::TJsonValue outputOptionsJson;
        NJson::TJsonValue featuresSelectJsonOptions;
        PlainJsonToOptions(plainJsonParams, &catBoostJsonOptions, &outputOptionsJson, &featuresSelectJsonOptions);
        ConvertFeaturesForSelectFromStringToIndices(pools.Learn.Get()->MetaInfo, &featuresSelectJsonOptions);

        const auto taskType = GetTaskType(catBoostJsonOptions);
        TCatBoostOptions catBoostOptions(taskType);
        catBoostOptions.Load(catBoostJsonOptions);
        TOutputFilesOptions outputFileOptions;
        outputFileOptions.Load(outputOptionsJson);
        TFeaturesSelectOptions featuresSelectOptions;
        featuresSelectOptions.Load(featuresSelectJsonOptions);
        featuresSelectOptions.CheckAndUpdateSteps();

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(catBoostOptions.SystemOptions->NumThreads - 1);

        const auto summary = SelectFeatures(
            catBoostOptions,
            outputFileOptions,
            /*poolLoadParams*/ nullptr,
            featuresSelectOptions,
            pools,
            dstModel,
            testApproxes,
            metricsAndTimeHistory,
            &executor
        );
        return ToJson(summary);
    }
}
