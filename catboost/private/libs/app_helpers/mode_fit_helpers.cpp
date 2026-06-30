#include "mode_fit_helpers.h"

#include "bind_options.h"

#include <catboost/libs/data/baseline.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/monotone_constraints.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/feature_penalties_options.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/options/pool_metainfo_options.h>
#include <catboost/libs/train_lib/train_model.h>
#include <catboost/libs/train_lib/trainer_env.h>

#if defined(USE_MPI)
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>

#include <catboost/libs/logging/logging.h>
#endif

#include <library/cpp/json/json_reader.h>

#include <util/generic/ptr.h>

using namespace NCB;


int NCB::ModeFitImpl(int argc, const char* argv[]) {
    #if defined(USE_MPI)
    char** args = const_cast<char**>(argv);
        auto& mpiManager = NCudaLib::GetMpiManager();
    mpiManager.Start(&argc, &args);
    if (!mpiManager.IsMaster()) {
        CATBOOST_DEBUG_LOG << "Running MPI slave" << Endl;
        RunSlave();
        return 0;
    }
    #endif
    {
        NCatboostOptions::TPoolLoadParams poolLoadParams;
        TString paramsFile;
        NJson::TJsonValue catBoostFlatJsonOptions(NJson::JSON_MAP);
        ParseCommandLine(argc, argv, &catBoostFlatJsonOptions, &paramsFile, &poolLoadParams);
        NJson::TJsonValue catBoostJsonOptions;
        NJson::TJsonValue outputOptionsJson;
        InitOptions(paramsFile, &catBoostJsonOptions, &outputOptionsJson);
        NCatboostOptions::LoadPoolMetaInfoOptions(poolLoadParams.PoolMetaInfoPath, &catBoostJsonOptions);
        ConvertIgnoredFeaturesFromStringToIndices(poolLoadParams, &catBoostFlatJsonOptions);
        ConvertFixedBinarySplitsFromStringToIndices(poolLoadParams, &catBoostFlatJsonOptions);
        NCatboostOptions::PlainJsonToOptions(catBoostFlatJsonOptions, &catBoostJsonOptions, &outputOptionsJson);
        ConvertParamsToCanonicalFormat(poolLoadParams, &catBoostJsonOptions);

        // need json w/o feature names dependent params or LoadOptions can fail (because feature names could be extracted from Pool data)
        NJson::TJsonValue catBoostJsonOptionsWithoutFeatureNamesDependentParams = catBoostJsonOptions;
        ExtractFeatureNamesDependentParams(&catBoostJsonOptionsWithoutFeatureNamesDependentParams);
        auto options = NCatboostOptions::LoadOptions(catBoostJsonOptionsWithoutFeatureNamesDependentParams);
        auto trainerEnv = NCB::CreateTrainerEnv(options);

        CopyIgnoredFeaturesToPoolParams(catBoostJsonOptions, &poolLoadParams);
        NCatboostOptions::TOutputFilesOptions outputOptions;
        outputOptions.Load(outputOptionsJson);

        TrainModel(poolLoadParams, outputOptions, catBoostJsonOptions);
    }

    #if defined(USE_MPI)
    if (mpiManager.IsMaster()) {
        CATBOOST_INFO_LOG << "Stopping MPI slaves" << Endl;
        mpiManager.Stop();
    }
    #endif
    return 0;
}
