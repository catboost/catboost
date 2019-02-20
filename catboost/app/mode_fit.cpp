#include "modes.h"
#include "bind_options.h"

#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/train_lib/train_model.h>

#if defined(USE_MPI)
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>

#include <catboost/libs/logging/logging.h>
#endif

#include <library/json/json_reader.h>

#include <util/generic/ptr.h>
#include <util/stream/fwd.h>
#include <util/system/fs.h>



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
    NCatboostOptions::TPoolLoadParams poolLoadOptions;
    TString paramsFile;
    NJson::TJsonValue catBoostFlatJsonOptions;
    ParseCommandLine(argc, argv, &catBoostFlatJsonOptions, &paramsFile, &poolLoadOptions);
    NJson::TJsonValue catBoostJsonOptions;
    NJson::TJsonValue outputOptionsJson;
    if (!paramsFile.empty()) {
        CB_ENSURE(NFs::Exists(paramsFile), "Params file does not exist " << paramsFile);
        TIFStream in(paramsFile);
        NJson::TJsonValue fromFileParams;
        CB_ENSURE(NJson::ReadJsonTree(&in, &fromFileParams), "can't parse params file");
        NCatboostOptions::PlainJsonToOptions(fromFileParams, &catBoostJsonOptions, &outputOptionsJson);
    }
    NCatboostOptions::PlainJsonToOptions(catBoostFlatJsonOptions, &catBoostJsonOptions, &outputOptionsJson);

    poolLoadOptions.IgnoredFeatures = GetOptionIgnoredFeatures(catBoostJsonOptions);

    auto taskType = NCatboostOptions::GetTaskType(catBoostJsonOptions);
    poolLoadOptions.Validate(taskType);

    THolder<IModelTrainer> modelTrainerHolder;
    NCatboostOptions::TOutputFilesOptions outputOptions;
    if (!outputOptionsJson.Has("train_dir")) {
        outputOptionsJson["train_dir"] = ".";
    }
    outputOptions.Load(outputOptionsJson);
    //Cout << LabeledOutput(outputOptions.UseBestModel.IsSet()) << Endl;

    TrainModel(poolLoadOptions, outputOptions, catBoostJsonOptions);

    #if defined(USE_MPI)
    if (mpiManager.IsMaster()) {
        CATBOOST_INFO_LOG << "Stopping MPI slaves" << Endl;
        mpiManager.Stop();
    }
    #endif
    return 0;
}

