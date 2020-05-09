#include "modes.h"
#include "bind_options.h"

#include <catboost/private/libs/algo/helpers.h>
#include <catboost/libs/data/baseline.h>
#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/monotone_constraints.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/feature_penalties_options.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/libs/train_lib/train_model.h>


#if defined(USE_MPI)
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>

#include <catboost/libs/logging/logging.h>
#endif

#include <library/cpp/json/json_reader.h>

#include <util/generic/ptr.h>


using namespace NCB;


int mode_fit(int argc, const char* argv[]) {
    ConfigureMalloc();

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
    NCatboostOptions::TPoolLoadParams poolLoadParams;
    TString paramsFile;
    NJson::TJsonValue catBoostFlatJsonOptions(NJson::JSON_MAP);
    ParseCommandLine(argc, argv, &catBoostFlatJsonOptions, &paramsFile, &poolLoadParams);
    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputOptionsJson;
    InitOptions(paramsFile, &catBoostJsonOptions, &outputOptionsJson);
    ConvertIgnoredFeaturesFromStringToIndices(poolLoadParams, &catBoostFlatJsonOptions);
    NCatboostOptions::PlainJsonToOptions(catBoostFlatJsonOptions, &catBoostJsonOptions, &outputOptionsJson);
    ConvertParamsToCanonicalFormat(poolLoadParams, &catBoostJsonOptions);
    CopyIgnoredFeaturesToPoolParams(catBoostJsonOptions, &poolLoadParams);
    NCatboostOptions::TOutputFilesOptions outputOptions;
    outputOptions.Load(outputOptionsJson);
    //Cout << LabeledOutput(outputOptions.UseBestModel.IsSet()) << Endl;

    TrainModel(poolLoadParams, outputOptions, catBoostJsonOptions);

    #if defined(USE_MPI)
    if (mpiManager.IsMaster()) {
        CATBOOST_INFO_LOG << "Stopping MPI slaves" << Endl;
        mpiManager.Stop();
    }
    #endif
    return 0;
}

