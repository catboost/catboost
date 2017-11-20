#include "cmd_line.h"
#include "output_fstr.h"

#include <catboost/libs/algo/cv_data_partition.h>
#include <catboost/libs/algo/train_model.h>
#include <catboost/libs/algo/tree_print.h>
#include <catboost/libs/helpers/mem_usage.h>
#include <catboost/libs/helpers/eval_helpers.h>
#include <catboost/libs/fstr/calc_fstr.h>
#include <catboost/libs/data/load_data.h>
#include <catboost/libs/params/params.h>
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
                       bool* verbose,
                       TVector<TString>* classNames) {
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
    if (trainJson->Has("class_names")) {
        classNames->clear();
        if ((*trainJson)["class_names"].IsArray()) {
            for (const auto& value : (*trainJson)["class_names"].GetArraySafe()) {
                classNames->push_back(value.GetStringSafe());
            }
        } else {
            classNames->push_back((*trainJson)["class_names"].GetStringSafe());
        }
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
    bool verbose = false;
    TVector<TString> classNames;
    LoadParams(argc, argv, &trainJson, &params, &threadCount, &dataDir, &verbose, &classNames);

    TProfileInfo profile(true);

    TPool learnPool;
    if (!params.LearnFile.empty()) {
        ReadPool(params.CdFile, params.LearnFile, params.LearnPairsFile, threadCount, verbose,
                 params.Delimiter, params.HasHeaders, classNames, &learnPool);
        profile.AddOperation("Build learn pool");
    }

    TPool testPool;
    if (!params.TestFile.empty()) {
        ReadPool(params.CdFile, params.TestFile, params.TestPairsFile, threadCount, verbose,
                 params.Delimiter, params.HasHeaders, classNames, &testPool);
        profile.AddOperation("Build test pool");
    }

    if (params.CvParams.FoldCount != 0) {
        CB_ENSURE(params.TestFile.empty(), "Test file is not supported in cross-validation mode");
        CB_ENSURE(params.LearnPairsFile.empty() && params.TestPairsFile.empty(),
                  "Pairs are not supported in cross-validation mode");
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

    TEvalResult evalResult;
    bool allowClearPool = params.FstrRegularFileName.empty() && params.FstrInternalFileName.empty();
    TrainModel(trainJson, Nothing(), Nothing(), learnPool, allowClearPool, testPool, params.ModelFileName,
                   nullptr, &evalResult);

    SetVerboseLogingMode();
    if (params.EvalFileName) {
        MATRIXNET_INFO_LOG << "Writing test eval to: " << params.EvalFileName << Endl;
        TOFStream fileStream(params.EvalFileName);
        evalResult.PostProcess(threadCount);
        evalResult.OutputToFile(testPool.Docs.Id, &fileStream, true, &testPool.Docs.Target);
    } else {
        MATRIXNET_INFO_LOG << "Skipping test eval output" << Endl;
    }
    profile.AddOperation("Train model");

    if (trainJson.Has("detailed_profile") || trainJson.Has("developer_mode")) {
        profile.LogState();
    }

    if (!params.FstrRegularFileName.empty() || !params.FstrInternalFileName.empty()) {
        TFullModel model = ReadModel(params.ModelFileName);
        CalcAndOutputFstr(model, learnPool, &params.FstrRegularFileName, &params.FstrInternalFileName, threadCount);
    }

    MATRIXNET_INFO_LOG << runTimer.Passed() / 60 << " min passed" << Endl;
    SetSilentLogingMode();

    return 0;
}
