#include "cmd_line.h"
#include "output_fstr.h"

#include <catboost/libs/algo/cv_data_partition.h>
#include <catboost/libs/algo/calc_fstr.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/algo/train_model.h>
#include <catboost/libs/algo/params.h>
#include <catboost/libs/algo/tree_print.h>
#include <catboost/libs/algo/eval_helpers.h>
#include <catboost/libs/model/model.h>

#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/logging/logging.h>

#include <library/grid_creator/binarization.h>
#include <library/json/json_reader.h>

#include <library/malloc/api/malloc.h>

#include <util/string/cast.h>
#include <util/stream/file.h>
#include <util/system/hp_timer.h>
#include <util/system/fs.h>
#include <util/system/info.h>

#include <iostream>

static void MoveFiles(const TString& trainDir, TCmdLineParams* params) {
    TFsPath trainDirPath(trainDir);
    if (!trainDir.empty() && !trainDirPath.Exists()) {
        trainDirPath.MkDir();
    }
    CB_ENSURE(!params->ModelFileName.empty(), "empty model filename");

    params->ModelFileName = TOutputFiles::AlignFilePath(trainDir, params->ModelFileName);
    if (!params->FstrRegularFileName.empty()) {
        params->FstrRegularFileName = TOutputFiles::AlignFilePath(trainDir, params->FstrRegularFileName);
    }
    if (!params->FstrInternalFileName.empty()) {
        params->FstrInternalFileName = TOutputFiles::AlignFilePath(trainDir, params->FstrInternalFileName);
    }
    if (params->EvalFileName) {
        params->EvalFileName = TOutputFiles::AlignFilePath(trainDir, params->EvalFileName);
    }
}


static void AddParamsFromFile(const TString& paramsPath, NJson::TJsonValue* trainJson) {
    TIFStream reader(paramsPath);
    NJson::TJsonValue fileJson;
    CB_ENSURE(NJson::ReadJsonTree(&reader, &fileJson), "can't parse params file");
    CB_ENSURE(fileJson.IsMap(), "wrong params file format, should be map param_name:param_arg");
    for (const auto& param : fileJson.GetMapSafe()) {
        if (!trainJson->GetMapSafe().has(param.first)) {
            trainJson->GetMapSafe()[param.first] = param.second;
        }
    }
}


static void LoadParams(int argc, const char* argv[],
                       NJson::TJsonValue* trainJson,
                       TCmdLineParams* params,
                       int* threadCount,
                       TString* dataDir,
                       bool* verbose) {
    TString paramsPath;
    ParseCommandLine(argc, argv, trainJson, params, &paramsPath);
    if (!paramsPath.empty()) {
        AddParamsFromFile(paramsPath, trainJson);
    }
    if (trainJson->Has("train_dir")) {
        *dataDir = (*trainJson)["train_dir"].GetStringSafe();
    }
    if (trainJson->Has("verbose")) {
        *verbose = (*trainJson)["verbose"].GetBooleanSafe();
    }
    if (trainJson->Has("thread_count")) {
        *threadCount = (*trainJson)["thread_count"].GetIntegerSafe();
    }
}

int mode_fit(int argc, const char* argv[]) {

#if !(defined(__APPLE__) && defined(__MACH__)) // there is no LF for MacOS
    if (!NMalloc::MallocInfo().SetParam("LB_LIMIT_TOTAL_SIZE", "1000000")) {
        MATRIXNET_WARNING_LOG << "link me with lfalloc please" << Endl;
    }
#endif

    THPTimer runTimer;

    TCmdLineParams params;
    NJson::TJsonValue trainJson;
    int threadCount = Min(8, (int)NSystemInfo::CachedNumberOfCpus());
    TString dataDir;
    bool verbose = true;
    LoadParams(argc, argv, &trainJson, &params, &threadCount, &dataDir, &verbose);

    TProfileInfo profile(true);

    TPool learnPool;
    if (!params.LearnFile.empty()) {
        ReadPool(params.CdFile, params.LearnFile, threadCount, verbose, &learnPool, params.Delimiter, params.HasHeaders);
        profile.AddOperation("Build learn pool");
    }

    TPool testPool;
    if (!params.TestFile.empty()) {
        ReadPool(params.CdFile, params.TestFile, threadCount, verbose, &testPool, params.Delimiter, params.HasHeaders);
        profile.AddOperation("Build test pool");
    }

    if (params.CvParams.FoldCount != 0) {
        Y_VERIFY(params.CvParams.FoldIdx != -1);
        BuildCvPools(
            params.CvParams.FoldIdx,
            params.CvParams.FoldCount,
            params.CvParams.Inverted,
            params.CvParams.RandSeed,
            &learnPool,
            &testPool);
        profile.AddOperation("Build cv pools");
    }
    MoveFiles(dataDir, &params);

    yvector<yvector<double>> testApprox;
    TrainModel(trainJson, Nothing(), Nothing(), learnPool, testPool, params.ModelFileName, nullptr, &testApprox);

    SetVerboseLogingMode();
    if (params.EvalFileName) {
        MATRIXNET_INFO_LOG << "Writing test eval to: " << params.EvalFileName << Endl;
        OutputTestEval(testApprox, params.EvalFileName, testPool.Docs, true);
    } else {
        MATRIXNET_INFO_LOG << "Skipping test eval output" << Endl;
    }
    profile.AddOperation("Train model");

    if (trainJson.Has("detailed_profile") || trainJson.Has("developer_mode")) {
        profile.PrintState();
    }

    if (!params.FstrRegularFileName.empty() || !params.FstrInternalFileName.empty()){
        TFullModel model = ReadModel(params.ModelFileName);
        CalcAndOutputFstr(model, learnPool, &params.FstrRegularFileName, &params.FstrInternalFileName, threadCount);
    }

    MATRIXNET_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
    SetSilentLogingMode();

    return 0;
}
