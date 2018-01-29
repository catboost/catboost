#include "modes.h"
#include "bind_options.h"
#include <catboost/libs/algo/helpers.h>
#include <catboost/libs/options/plain_options_helper.h>
#include <catboost/libs/train_lib/train_model.h>


int mode_fit(const int argc, const char* argv[]) {
    ConfigureMalloc();

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
    poolLoadOptions.Validate();

    auto taskType = NCatboostOptions::GetTaskType(catBoostJsonOptions);
    THolder<IModelTrainer> modelTrainerHolder;
    NCatboostOptions::TOutputFilesOptions outputOptions(taskType);
    outputOptions.Load(outputOptionsJson);

    const bool isGpuDeviceType = taskType == ETaskType::GPU;
    if (isGpuDeviceType && TTrainerFactory::Has(ETaskType::GPU)) {
        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::GPU);
    } else {
        CB_ENSURE(!isGpuDeviceType, "GPU Device not found.");

        modelTrainerHolder = TTrainerFactory::Construct(ETaskType::CPU);
    }
    modelTrainerHolder->TrainModel(poolLoadOptions, outputOptions, catBoostJsonOptions);
    return 0;
}

