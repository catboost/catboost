#include "modes.h"

#if defined(USE_MPI)
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#endif

#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/data/loader.h>
#include <catboost/libs/data/order.h>
#include <catboost/libs/features_selection/select_features.h>
#include <catboost/libs/features_selection/selection_results.h>
#include <catboost/libs/train_lib/trainer_env.h>

#include <catboost/private/libs/algo/data.h>
#include <catboost/private/libs/algo/helpers.h>
#include <catboost/private/libs/algo/preprocess.cpp>
#include <catboost/private/libs/app_helpers/bind_options.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/features_select_options.h>
#include <catboost/private/libs/options/load_options.h>
#include <catboost/private/libs/options/output_file_options.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/options/pool_metainfo_options.h>

#include <library/cpp/json/json_writer.h>

using namespace NCB;
using namespace NCatboostOptions;


static void LoadOptions(
    int argc,
    const char* argv[],
    TCatBoostOptions* catBoostOptions,
    TOutputFilesOptions* outputOptions,
    TPoolLoadParams* poolLoadParams,
    TFeaturesSelectOptions* featuresSelectOptions
) {
    NJson::TJsonValue catBoostFlatJsonOptions;
    TString paramsFile;
    ParseFeaturesSelectCommandLine(
        argc,
        argv,
        &catBoostFlatJsonOptions,
        &paramsFile,
        poolLoadParams
    );

    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputOptionsJson;
    NJson::TJsonValue featuresSelectJsonOptions;
    InitOptions(paramsFile, &catBoostJsonOptions, &outputOptionsJson, &featuresSelectJsonOptions);
    LoadPoolMetaInfoOptions(poolLoadParams->PoolMetaInfoPath, &catBoostJsonOptions);

    ConvertIgnoredFeaturesFromStringToIndices(*poolLoadParams, &catBoostFlatJsonOptions);
    NCatboostOptions::PlainJsonToOptions(catBoostFlatJsonOptions, &catBoostJsonOptions, &outputOptionsJson, &featuresSelectJsonOptions);
    ConvertFeaturesForSelectFromStringToIndices(*poolLoadParams, &featuresSelectJsonOptions);
    ConvertParamsToCanonicalFormat(*poolLoadParams, &catBoostJsonOptions);
    CopyIgnoredFeaturesToPoolParams(catBoostJsonOptions, poolLoadParams);

    const auto taskType = GetTaskType(catBoostFlatJsonOptions);
    *catBoostOptions = TCatBoostOptions(taskType);
    catBoostOptions->Load(catBoostJsonOptions);
    outputOptions->Load(outputOptionsJson);
    featuresSelectOptions->Load(featuresSelectJsonOptions);
    featuresSelectOptions->CheckAndUpdateSteps();
}


static TDataProviders LoadPools(
    const TCatBoostOptions& catBoostOptions,
    const TPoolLoadParams& poolLoadParams,
    TOption<bool>* hasTimeFlag,
    NPar::ILocalExecutor* executor
) {
    TVector<NJson::TJsonValue> classLabels = catBoostOptions.DataProcessingOptions->ClassLabels;
    const auto objectsOrder = hasTimeFlag->Get() ? EObjectsOrder::Ordered : EObjectsOrder::Undefined;
    CB_ENSURE(poolLoadParams.TestSetPaths.size() <= 1, "Features selection mode doesn't support several eval sets.");
    const bool haveLearnFeaturesInMemory = HaveFeaturesInMemory(
        catBoostOptions,
        poolLoadParams.LearnSetPath);
    TVector<TDatasetSubset> testDatasetSubsets;
    for (const auto& testSetPath : poolLoadParams.TestSetPaths) {
        testDatasetSubsets.push_back(
            TDatasetSubset::MakeColumns(HaveFeaturesInMemory(catBoostOptions, testSetPath)));
    }
    auto pools = NCB::ReadTrainDatasets(
        catBoostOptions.GetTaskType(),
        poolLoadParams,
        objectsOrder,
        /*readTestData*/true,
        TDatasetSubset::MakeColumns(haveLearnFeaturesInMemory),
        testDatasetSubsets,
        catBoostOptions.DataProcessingOptions->ForceUnitAutoPairWeights,
        &classLabels,
        executor,
        /*profile*/nullptr
    );
    CB_ENSURE(pools.Learn != nullptr, "Train data must be provided");

    const auto learnDataOrder = pools.Learn->ObjectsData->GetOrder();
    if (learnDataOrder == EObjectsOrder::Ordered) {
        *hasTimeFlag = true;
    }
    pools.Learn = ReorderByTimestampLearnDataIfNeeded(catBoostOptions, pools.Learn, executor);

    TRestorableFastRng64 rand(catBoostOptions.RandomSeed);
    pools.Learn = ShuffleLearnDataIfNeeded(catBoostOptions, pools.Learn, executor, &rand);

    return pools;
}


void SaveSummaryToFile(const TFeaturesSelectionSummary& summary, const TString& resultPath) {
    auto summaryJson = ToJson(summary);
    TOFStream out(resultPath);
    NJson::TJsonWriterConfig config;
    config.FormatOutput = true;
    config.SortKeys = true;
    WriteJson(&out, &summaryJson, config);
}


int mode_select_features(int argc, const char* argv[]) {
    ConfigureMalloc();

    #if defined(USE_MPI)
    char** args = const_cast<char**>(argv);
        auto& mpiManager = NCudaLib::GetMpiManager();
    mpiManager.Start(&argc, &args);
        CATBOOST_DEBUG_LOG << (mpiManager.IsMaster() ? "MASTER" : "SLAVE") << Endl;
    if (!mpiManager.IsMaster()) {
        CATBOOST_DEBUG_LOG << "Running MPI slave" << Endl;
        RunSlave();
        return 0;
    }
    #endif

    TCatBoostOptions catBoostOptions(ETaskType::CPU /* no matter */);
    TOutputFilesOptions outputOptions;
    TPoolLoadParams poolLoadParams;
    TFeaturesSelectOptions featuresSelectOptions;
    LoadOptions(
        argc,
        argv,
        &catBoostOptions,
        &outputOptions,
        &poolLoadParams,
        &featuresSelectOptions
    );

    TSetLogging inThisScope(catBoostOptions.LoggingLevel);

    auto trainerEnv = NCB::CreateTrainerEnv(catBoostOptions);

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(catBoostOptions.SystemOptions->NumThreads - 1);

    CATBOOST_INFO_LOG << "Loading pools..." << Endl;
    auto pools = LoadPools(
        catBoostOptions,
        poolLoadParams,
        &catBoostOptions.DataProcessingOptions->HasTimeFlag,
        &executor
    );

    TVector<TEvalResult> evalResults(pools.Test.size());

    const TFeaturesSelectionSummary summary = SelectFeatures(
        catBoostOptions,
        outputOptions,
        &poolLoadParams,
        featuresSelectOptions,
        /*evalMetricDescriptor*/ Nothing(),
        pools,
        /*dstModel*/ nullptr,
        /*evalResults*/ GetMutablePointers(evalResults),
        /*metricsAndTimeHistory*/ nullptr,
        &executor
    );
    SaveSummaryToFile(summary, featuresSelectOptions.ResultPath);

    CATBOOST_INFO_LOG << "Selected features:";
    for (auto feature : summary.SelectedFeatures) {
        CATBOOST_INFO_LOG << " " << feature;
    }
    CATBOOST_INFO_LOG << Endl;


    #if defined(USE_MPI)
    if (mpiManager.IsMaster()) {
        CATBOOST_INFO_LOG << "Stopping MPI slaves" << Endl;
        mpiManager.Stop();
    }
    #endif

    return 0;
}
