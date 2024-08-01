#include "select_features.h"

#include "recursive_features_elimination.h"

#include <catboost/libs/data/borders_io.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/helpers/restorable_rng.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/train_lib/dir_helper.h>
#include <catboost/libs/train_lib/options_helper.h>
#include <catboost/libs/train_lib/trainer_env.h>
#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/full_model_saver.h>
#include <catboost/private/libs/algo/preprocess.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/private/libs/options/defaults_helper.h>
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
                "Please install latest NVDIA driver and check again"
            );
        }

        auto checkCountConsistency = [] (
            auto entriesForSelectSize,
            const TOption<int>& numberOfEntriesToSelect,
            TStringBuf entriesName
        ) {
            CB_ENSURE(
                numberOfEntriesToSelect.IsSet(),
                "You should specify the number of " << entriesName << " to select"
            );
            CB_ENSURE(
                numberOfEntriesToSelect.Get() > 0,
                "Number of " << entriesName << " to select should be positive"
            );
            CB_ENSURE(entriesForSelectSize > 0, "You should specify " << entriesName << " to select from");
            CB_ENSURE(
                static_cast<int>(entriesForSelectSize) >= numberOfEntriesToSelect.Get(),
                "It is impossible to select " << numberOfEntriesToSelect.Get() << ' ' << entriesName
                << " from " << entriesForSelectSize << ' ' << entriesName
            );
        };


        if (featuresSelectOptions.Grouping.Get() == EFeaturesSelectionGrouping::Individual) {
            const auto& featuresForSelect = featuresSelectOptions.FeaturesForSelect.Get();

            checkCountConsistency(
                featuresForSelect.size(),
                featuresSelectOptions.NumberOfFeaturesToSelect,
                "features"
            );

            const ui32 featureCount = pools.Learn->MetaInfo.GetFeatureCount();
            for (const ui32 feature : featuresForSelect) {
                CB_ENSURE(
                    feature < featureCount,
                    "Tested feature " << feature << " is not present; dataset contains only " << featureCount
                    << " features"
                );
            }
        } else { // ByTags
            const auto& featuresTagsForSelect = featuresSelectOptions.FeaturesTagsForSelect.Get();

            checkCountConsistency(
                featuresTagsForSelect.size(),
                featuresSelectOptions.NumberOfFeaturesTagsToSelect,
                "features tags"
            );

            const auto& datasetFeaturesTags = pools.Learn->MetaInfo.FeaturesLayout->GetTagToExternalIndices();
            for (const auto& featuresTag : featuresTagsForSelect) {
                CB_ENSURE(
                    datasetFeaturesTags.contains(featuresTag),
                    "Tested features tag \"" << featuresTag << "\" is not present in dataset features tags"
                );
            }
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
                quantizedFeaturesInfo.Get()
            );
        }

        for (auto testPoolIdx : xrange(pools.Test.size())) {
            const auto& testPool = *pools.Test[testPoolIdx];
            if (testPool.GetObjectCount() == 0) {
                continue;
            }
            CheckCompatibleForApply(
                *learnFeaturesLayout,
                *testPool.MetaInfo.FeaturesLayout,
                TStringBuilder() << "test dataset #" << testPoolIdx
            );
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

    static TDataProviderPtr LoadSubsetForFstrCalc(
        const TDataProviderPtr srcPool,
        const TCatBoostOptions& catBoostOptions,
        const TPoolLoadParams* poolLoadParams,
        NPar::ILocalExecutor* executor
    ) {
        CATBOOST_DEBUG_LOG << "Loading fstr pool..." << Endl;
        const ui32 totalDocumentCount = srcPool->GetObjectCount();
        const ui32 minSubsetDocumentCount = SafeIntegerCast<ui32>(
            GetMaxObjectCountForFstrCalc(
                totalDocumentCount,
                SafeIntegerCast<i64>(srcPool->ObjectsData->GetFeaturesLayout()->GetExternalFeatureCount())
            )
        );

        ui32 subsetDocumentCount = 0;
        if (srcPool->ObjectsGrouping->IsTrivial()) {
            subsetDocumentCount = minSubsetDocumentCount;
        } else {
            ui32 lastGroupIdx = 0;
            while (srcPool->ObjectsGrouping->GetGroup(lastGroupIdx).End < minSubsetDocumentCount) {
                lastGroupIdx += 1;
            }
            subsetDocumentCount = srcPool->ObjectsGrouping->GetGroup(lastGroupIdx).End;
        }

        auto classLabels = catBoostOptions.DataProcessingOptions->ClassLabels.Get();
        return ReadDataset(
            catBoostOptions.GetTaskType(),
            poolLoadParams->LearnSetPath,
            poolLoadParams->PairsFilePath,
            poolLoadParams->GraphFilePath,
            poolLoadParams->GroupWeightsFilePath,
            poolLoadParams->TimestampsFilePath,
            poolLoadParams->BaselineFilePath,
            poolLoadParams->FeatureNamesPath,
            poolLoadParams->PoolMetaInfoPath,
            poolLoadParams->ColumnarPoolFormatParams,
            poolLoadParams->IgnoredFeatures,
            catBoostOptions.DataProcessingOptions->HasTimeFlag.Get() ? EObjectsOrder::Ordered : EObjectsOrder::Undefined,
            TDatasetSubset::MakeRange(0, subsetDocumentCount),
            /*loadSampleIds*/ false,
            catBoostOptions.DataProcessingOptions->ForceUnitAutoPairWeights,
            &classLabels,
            executor
        );
    }


    TFeaturesSelectionSummary SelectFeatures(
        TCatBoostOptions catBoostOptions,
        TOutputFilesOptions outputFileOptions,
        const TPoolLoadParams* poolLoadParams,
        const TFeaturesSelectOptions& featuresSelectOptions,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
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

        TDataProviderPtr srcPool = pools.Test.empty() ? pools.Learn : pools.Test[0];
        TMaybe<TPathWithScheme> srcPoolPath = poolLoadParams
            ? MakeMaybe(pools.Test.empty() ? poolLoadParams->LearnSetPath : poolLoadParams->TestSetPaths[0])
            : Nothing();
        const bool haveFeaturesInMemory = HaveFeaturesInMemory(catBoostOptions, srcPoolPath);
        TDataProviderPtr fstrPool = haveFeaturesInMemory
            ? GetSubsetForFstrCalc(srcPool, executor)
            : LoadSubsetForFstrCalc(srcPool, catBoostOptions, poolLoadParams, executor);
        CATBOOST_DEBUG_LOG << "Fstr pool size: " << fstrPool->GetObjectCount() << Endl;

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

        InitializeEvalMetricIfNotSet(
            catBoostOptions.MetricOptions->ObjectiveMetric,
            &catBoostOptions.MetricOptions->EvalMetric
        );

        UpdateMetricPeriodOption(catBoostOptions, &outputFileOptions);

        TFeaturesSelectionSummary summary = DoRecursiveFeaturesElimination(
            catBoostOptions,
            outputFileOptions,
            featuresSelectOptions,
            evalMetricDescriptor,
            fstrPool,
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
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
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

        auto trainerEnv = NCB::CreateTrainerEnv(catBoostOptions);

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(catBoostOptions.SystemOptions->NumThreads - 1);

        const auto summary = SelectFeatures(
            catBoostOptions,
            outputFileOptions,
            /*poolLoadParams*/ nullptr,
            featuresSelectOptions,
            evalMetricDescriptor,
            pools,
            dstModel,
            testApproxes,
            metricsAndTimeHistory,
            &executor
        );
        return ToJson(summary);
    }
}
