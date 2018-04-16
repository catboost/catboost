#include "modes.h"
#include "bind_options.h"
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/train_lib/train_model.h>

#if defined(USE_MPI)
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/mpi/mpi_manager.h>
#endif

int mode_fit(int argc, const char* argv[]) {
    ConfigureMalloc();

    #if defined(USE_MPI)
    char** args = const_cast<char**>(argv);
    auto& mpiManager = NCudaLib::GetMpiManager();
    mpiManager.Start(&argc, &args);
    if (!mpiManager.IsMaster()) {
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
    poolLoadOptions.Validate();

    auto taskType = NCatboostOptions::GetTaskType(catBoostJsonOptions);
    THolder<IModelTrainer> modelTrainerHolder;
    NCatboostOptions::TOutputFilesOptions outputOptions(taskType);
    if (!outputOptionsJson.Has("train_dir")) {
        outputOptionsJson["train_dir"] = ".";
    }
    outputOptions.Load(outputOptionsJson);

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(!isGpuDeviceType, "GPU support was not compiled");
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }
    modelTrainerHolder->TrainModel(poolLoadOptions, outputOptions, catBoostJsonOptions);

    #if defined(USE_MPI)
    if (mpiManager.IsMaster()) {
        mpiManager.Stop();
    }
    #endif
    return 0;
}

