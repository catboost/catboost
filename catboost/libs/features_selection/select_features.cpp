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
        const TPoolLoadParams& poolLoadParams,
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
            CB_ENSURE(Count(poolLoadParams.IgnoredFeatures, feature) == 0, "Tested feature " << feature << " should not be ignored");
        }
        const auto nFeaturesToEliminate = (int)featuresSelectOptions.FeaturesForSelect->size() - featuresSelectOptions.NumberOfFeaturesToSelect;
        CB_ENSURE(
            featuresSelectOptions.Steps <= nFeaturesToEliminate,
            "Features selection steps should not be greater than number of features to eliminate."
        );
    }


    static TTrainingDataProviders QuantizePools(
        const TPoolLoadParams& poolLoadParams,
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
        if (poolLoadParams.BordersFile) {
            LoadBordersAndNanModesFromFromFileInMatrixnetFormat(
                poolLoadParams.BordersFile,
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

        const bool haveLearnFeaturesInMemory = HaveLearnFeaturesInMemory(&poolLoadParams, *catBoostOptions);

        TTrainingDataProviders trainingData = GetTrainingData(
            pools,
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
        const TPoolLoadParams& poolLoadParams,
        const TFeaturesSelectOptions& featuresSelectOptions,
        const TDataProviders& pools,
        NPar::ILocalExecutor* executor
    ) {
        CheckOptions(
            catBoostOptions,
            poolLoadParams,
            featuresSelectOptions,
            pools
        );

        InitializeEvalMetricIfNotSet(catBoostOptions.MetricOptions->ObjectiveMetric,
                                    &catBoostOptions.MetricOptions->EvalMetric);

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

        const bool haveLearnFeaturesInMemory = HaveLearnFeaturesInMemory(&poolLoadParams, catBoostOptions);
        // TODO(ilyzhin) support distributed training with quantized pool
        CB_ENSURE(haveLearnFeaturesInMemory, "Features selection doesn't support distributed training with quantized pool yet.");
        if (catBoostOptions.SystemOptions->IsMaster()) {
            InitializeMaster(catBoostOptions.SystemOptions);
            if (!haveLearnFeaturesInMemory) {
                SetTrainDataFromQuantizedPool(
                    poolLoadParams,
                    catBoostOptions,
                    *trainingData.Learn->ObjectsGrouping,
                    *trainingData.Learn->MetaInfo.FeaturesLayout,
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

        TFeaturesSelectionSummary summary = DoRecursiveFeaturesElimination(
            catBoostOptions,
            outputFileOptions,
            featuresSelectOptions,
            pools,
            labelConverter,
            trainingData,
            executor
        );

        if (catBoostOptions.SystemOptions->IsMaster()) {
            FinalizeMaster(catBoostOptions.SystemOptions);
        }

        return summary;
    }
}
