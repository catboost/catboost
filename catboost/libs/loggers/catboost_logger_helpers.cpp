#include "catboost_logger_helpers.h"
#include "logger.h"

#include <catboost/libs/helpers/json_helpers.h>

#include <type_traits>


TTimeInfo::TTimeInfo(const TProfileResults& profileResults)
    : IterationTime(profileResults.CurrentTime)
    , PassedTime(profileResults.PassedTime)
    , RemainingTime(profileResults.RemainingTime)
{
}

NJson::TJsonValue TMetricsAndTimeLeftHistory::SaveMetrics() const {
    auto saveToJson = [&](const auto& field) {
        NJson::TJsonValue dst;
        TJsonFieldHelper<std::remove_const_t<std::remove_reference_t<decltype(field)>>>::Write(field, &dst);
        return dst;
    };

    NJson::TJsonValue result(NJson::JSON_MAP);
    result["learn_metrics_history"] = saveToJson(LearnMetricsHistory);
    result["test_metrics_history"] = saveToJson(TestMetricsHistory);
    if (BestIteration) {
        result["best_iteration"] = saveToJson(*BestIteration);
    }
    result["learn_best_error"] = saveToJson(LearnBestError);
    result["test_best_error"] = saveToJson(TestBestError);

    return result;
}

TMetricsAndTimeLeftHistory TMetricsAndTimeLeftHistory::LoadMetrics(const NJson::TJsonValue& rhs) {
    const auto& rhsMap = rhs.GetMap();

    auto loadFromJson = [&] (TStringBuf name, auto* field) {
        TJsonFieldHelper<std::remove_reference_t<decltype(*field)>>::Read(
            rhsMap.at(name),
            field
        );
    };

    TMetricsAndTimeLeftHistory result;
    loadFromJson("learn_metrics_history",  &result.LearnMetricsHistory);
    loadFromJson("test_metrics_history",  &result.TestMetricsHistory);
    if (rhsMap.contains("best_iteration")) {
        result.BestIteration = rhsMap.at("best_iteration").GetUIntegerSafe();
    }
    loadFromJson("learn_best_error", &result.LearnBestError);
    loadFromJson("test_best_error", &result.TestBestError);

    return result;
}


void TMetricsAndTimeLeftHistory::TryUpdateBestError(const IMetric& metric, double error, THashMap<TString, double>& bestError, bool updateBestIteration) {
    TString metricDescription = metric.GetDescription();
    bool shouldUpdate = false;
    if (!bestError.contains(metricDescription)) {
        shouldUpdate = true;
    } else {
        double currentBestError = bestError.at(metricDescription);
        float bestValue = 0;
        EMetricBestValue metricBestValueType;
        metric.GetBestValue(&metricBestValueType, &bestValue);
        shouldUpdate |= (metricBestValueType == EMetricBestValue::Min && error < currentBestError);
        shouldUpdate |= (metricBestValueType == EMetricBestValue::Max && error > currentBestError);
        shouldUpdate |= (metricBestValueType == EMetricBestValue::FixedValue &&
                         Abs(error - static_cast<double>(bestValue)) < Abs(currentBestError - static_cast<double>(bestValue)));
    }
    if (shouldUpdate) {
        bestError[metricDescription] = error;
        if (updateBestIteration) {
            BestIteration = TestMetricsHistory.size() - 1;
        }
    }
}

void TMetricsAndTimeLeftHistory::AddLearnError(const IMetric& metric, double error) {
    LearnMetricsHistory.back()[metric.GetDescription()] = error;
    TryUpdateBestError(metric, error, LearnBestError, false);
}

void TMetricsAndTimeLeftHistory::AddTestError(size_t testIdx, const IMetric& metric, double error, bool updateBestIteration) {
    if (testIdx >= TestMetricsHistory.back().size()) {
        TestMetricsHistory.back().resize(testIdx + 1);
    }
    TestMetricsHistory.back()[testIdx][metric.GetDescription()] = error;
    if (testIdx >= TestBestError.size()) {
        TestBestError.resize(testIdx + 1);
    }
    TryUpdateBestError(metric, error, TestBestError[testIdx], updateBestIteration);
}

TString TOutputFiles::AlignFilePath(const TString& baseDir, const TString& fileName, const TString& namePrefix) {
    const TFsPath filePath(fileName);
    if (filePath.IsAbsolute()) {
        return JoinFsPaths(filePath.Dirname(), namePrefix + filePath.Basename());
    }
    return JoinFsPaths(baseDir, namePrefix + fileName);
}

TString TOutputFiles::AlignFilePathAndCreateDir(const TString& baseDir, const TString& fileName, const TString& namePrefix) {
    const TString result = AlignFilePath(baseDir, fileName, namePrefix);
    const TFsPath resultingPath(result);
    const TString dirName = resultingPath.Dirname();
    const TFsPath dirPath(dirName);
    if (!dirName.empty() && !dirPath.Exists()) {
        dirPath.MkDirs();
    }
    return result;
}

void TOutputFiles::InitializeFiles(const NCatboostOptions::TOutputFilesOptions& params, const TString& namesPrefix) {
    if (!params.AllowWriteFiles()) {
        Y_ASSERT(TimeLeftLogFile.empty());
        Y_ASSERT(LearnErrorLogFile.empty());
        Y_ASSERT(TestErrorLogFile.empty());
        Y_ASSERT(SnapshotFile.empty());
        return;
    }

    const auto& trainDir = params.GetTrainDir();
    TFsPath trainDirPath(trainDir);
    if (!trainDir.empty() && !trainDirPath.Exists()) {
        trainDirPath.MkDirs();
    }
    NamesPrefix = namesPrefix;
    CB_ENSURE(!params.GetTimeLeftLogFilename().empty(), "empty time_left filename");
    TimeLeftLogFile = TOutputFiles::AlignFilePathAndCreateDir(trainDir, params.GetTimeLeftLogFilename(), NamesPrefix);

    CB_ENSURE(!params.GetLearnErrorFilename().empty(), "empty learn_error filename");
    LearnErrorLogFile = TOutputFiles::AlignFilePathAndCreateDir(trainDir, params.GetLearnErrorFilename(), NamesPrefix);
    if (params.GetTestErrorFilename()) {
        TestErrorLogFile = TOutputFiles::AlignFilePathAndCreateDir(trainDir, params.GetTestErrorFilename(), NamesPrefix);
    }
    if (params.SaveSnapshot()) {
        SnapshotFile = TOutputFiles::AlignFilePathAndCreateDir(trainDir, params.GetSnapshotFilename(), NamesPrefix);
    }
    const TString& jsonLogFilename = params.GetJsonLogFilename();
    CB_ENSURE(!jsonLogFilename.empty(), "empty json_log filename");
    JsonLogFile = TOutputFiles::AlignFilePathAndCreateDir(trainDir, jsonLogFilename, "");

    const TString& profileLogFilename = params.GetProfileLogFilename();
    CB_ENSURE(!profileLogFilename.empty(), "empty profile_log filename");
    ProfileLogFile = TOutputFiles::AlignFilePathAndCreateDir(trainDir, profileLogFilename, "");

    ExperimentName = params.GetName();
    TrainDir = trainDir;
}


void AddFileLoggers(
        bool detailedProfile,
        const TString& learnErrorLogFile,
        const TString& testErrorLogFile,
        const TString& timeLogFile,
        const TString& jsonLogFile,
        const TString& profileLogFile,
        const TString& trainDir,
        const NJson::TJsonValue& metaJson,
        int metricPeriod,
        TLogger* logger
) {
    TIntrusivePtr<ILoggingBackend> jsonLoggingBackend = new TJsonLoggingBackend(jsonLogFile, metaJson, metricPeriod);
    for (auto& jsonToken : metaJson["learn_sets"].GetArraySafe()) {
        TString token = jsonToken.GetString();
        logger->AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TErrorFileLoggingBackend(learnErrorLogFile)));
        logger->AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TTensorBoardLoggingBackend(JoinFsPaths(trainDir, token))));
        logger->AddBackend(token, jsonLoggingBackend);
    }
    TIntrusivePtr<ILoggingBackend> errorFileLoggingBackend;
    for (auto& jsonToken : metaJson["test_sets"].GetArraySafe()) {
        TString token = jsonToken.GetString();
        if (!errorFileLoggingBackend) {
            errorFileLoggingBackend.Reset(new TErrorFileLoggingBackend(testErrorLogFile));
        }
        logger->AddBackend(token, errorFileLoggingBackend);
        logger->AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TTensorBoardLoggingBackend(JoinFsPaths(trainDir, token))));
        logger->AddBackend(token, jsonLoggingBackend);
    }
    logger->AddProfileBackend(TIntrusivePtr<ILoggingBackend>(new TTimeFileLoggingBackend(timeLogFile)));
    logger->AddProfileBackend(jsonLoggingBackend);
    if (detailedProfile) {
        logger->AddProfileBackend(TIntrusivePtr<ILoggingBackend>(new TProfileLoggingBackend(profileLogFile)));
        logger->AddProfileBackend(TIntrusivePtr<ILoggingBackend>(new TJsonProfileLoggingBackend(profileLogFile + ".json")));
    }
    TString parametersToken = metaJson["parameters"].GetString();
    logger->AddBackend(parametersToken, jsonLoggingBackend);
}

void AddConsoleLogger(
        const TString& learnToken,
        const TVector<const TString>& testTokens,
        bool hasTrain,
        int metricPeriod,
        int iterationCount,
        TLogger* logger
) {
    TIntrusivePtr<ILoggingBackend> consoleLoggingBackend = new TConsoleLoggingBackend(/*detailedProfile=*/false, metricPeriod, iterationCount);
    if (hasTrain) {
        logger->AddBackend(learnToken, consoleLoggingBackend);
    }
    for (const TString& testToken : testTokens) {
        logger->AddBackend(testToken, consoleLoggingBackend);
    }
    logger->AddProfileBackend(consoleLoggingBackend);
}

void Log(
        int iteration,
        const TVector<TString>& metricsDescription,
        const TVector<THashMap<TString, double>>& learnErrorsHistory, // [iter][metric]
        const TVector<TVector<THashMap<TString, double>>>& testErrorsHistory, // [iter][test][metric]
        TMaybe<double> bestErrorValue,
        TMaybe<int> bestIteration,
        const TProfileResults& profileResults,
        const TString& learnToken,
        const TVector<const TString>& testTokens,
        bool outputErrors,
        TLogger* logger
) {
    TOneInterationLogger oneIterLogger(*logger);

    if (outputErrors) {
        if (iteration < learnErrorsHistory.ysize()) {
            const THashMap<TString, double>& learnErrors = learnErrorsHistory[iteration];
            for (int metricIdx = 0; metricIdx < metricsDescription.ysize(); ++metricIdx) {
                const TString& metricDescription = metricsDescription[metricIdx];
                if (learnErrors.contains(metricDescription)) {
                    oneIterLogger.OutputMetric(learnToken, TMetricEvalResult(metricDescription, learnErrors.at(metricDescription), metricIdx == 0));
                }
            }
        }
        if (iteration < testErrorsHistory.ysize()) {
            const int testCount = testErrorsHistory[iteration].ysize();
            for (int testIdx = 0; testIdx < testCount; ++testIdx) {
                const TString& token = testTokens[testIdx];
                const THashMap<TString, double>& testErrors = testErrorsHistory[iteration][testIdx];
                CB_ENSURE(
                    testErrors.size() == metricsDescription.size(),
                    "Wrong number of calculated metrics (" << testErrors.size() << "), expected "
                    << metricsDescription.size()
                );

                for (int metricIdx = 0; metricIdx < metricsDescription.ysize(); ++metricIdx) {
                    const TString& metricDescription = metricsDescription[metricIdx];

                    if (testErrors.contains(metricDescription)) {
                        double testError = testErrors.at(metricDescription);
                        bool isMainMetric = metricIdx == 0;

                        if ((testIdx == testCount - 1) && bestErrorValue) {
                            // Only last test should be followed by 'best:'
                            oneIterLogger.OutputMetric(token, TMetricEvalResult(metricDescription, testError, *bestErrorValue, *bestIteration, isMainMetric));
                        } else {
                            oneIterLogger.OutputMetric(token, TMetricEvalResult(metricDescription + ":" + ToString(testIdx), testError, isMainMetric));
                        }
                    }
                }
            }
        }
    }

    oneIterLogger.OutputProfile(profileResults);
}


NJson::TJsonValue GetJsonMeta(
        int iterationCount,
        const TString& optionalExperimentName,
        const TVector<const IMetric*>& metrics,
        const TVector<TString>& learnSetNames,
        const TVector<TString>& testSetNames,
        const TString& parametersName,
        ELaunchMode launchMode
) {
    NJson::TJsonValue meta;
    meta["iteration_count"] = iterationCount;
    meta["name"] = optionalExperimentName;

    meta.InsertValue("learn_sets", NJson::JSON_ARRAY);
    for (auto& name : learnSetNames) {
        meta["learn_sets"].AppendValue(name);
    }

    meta.InsertValue("test_sets", NJson::JSON_ARRAY);
    for (auto& name : testSetNames) {
        meta["test_sets"].AppendValue(name);
    }

    meta.InsertValue("learn_metrics", NJson::JSON_ARRAY);
    meta.InsertValue("test_metrics", NJson::JSON_ARRAY);
    for (const auto& loss : metrics) {

        NJson::TJsonValue metricJson;
        metricJson.InsertValue("name", loss->GetDescription());

        EMetricBestValue bestValueType;
        float bestValue;
        loss->GetBestValue(&bestValueType, &bestValue);
        TString bestValueString;
        if (bestValueType != EMetricBestValue::FixedValue) {
            metricJson.InsertValue("best_value", ToString(bestValueType));
        } else {
            metricJson.InsertValue("best_value", bestValue);
        }

        const TMap<TString, TString>& hints = loss->GetHints();
        if (!learnSetNames.empty() && (!hints.contains("skip_train") || hints.at("skip_train") == "false")) {
            meta["learn_metrics"].AppendValue(metricJson);
        }
        if (!testSetNames.empty()) {
            meta["test_metrics"].AppendValue(metricJson);
        }
    }

    meta.InsertValue("parameters", parametersName);
    meta.InsertValue("launch_mode", ToString<ELaunchMode>(launchMode));
    return meta;
}


TString GetTrainModelLearnToken() {
    return "learn";
}

TString GetParametersToken() {
    return "parameters";
}

TVector<const TString> GetTrainModelTestTokens(int testCount) {
    TString testTokenPrefix = "test";
    TVector<const TString> testTokens;
    for (int testIdx = 0; testIdx < testCount; ++testIdx) {
        TString testToken = testTokenPrefix + (testIdx > 0 ? ToString(testIdx) : "");
        testTokens.push_back(testToken);
    }
    return testTokens;
}


void InitializeFileLoggers(
        const NCatboostOptions::TCatBoostOptions& catboostOptions,
        const TOutputFiles& outputFiles,
        const TVector<const IMetric*>& metrics,
        const TString& learnToken,
        const TVector<const TString>& testTokens,
        int metricPeriod,
        TLogger* logger) {
    TVector<TString> metricDescriptions = GetMetricsDescription(metrics);

    TVector<TString> learnSetNames = {outputFiles.NamesPrefix + learnToken};
    TVector<TString> testSetNames;
    for (int testIdx = 0; testIdx < testTokens.ysize(); ++testIdx) {
        testSetNames.push_back({outputFiles.NamesPrefix + testTokens[testIdx]});
    }

    AddFileLoggers(
            catboostOptions.IsProfile,
            outputFiles.LearnErrorLogFile,
            outputFiles.TestErrorLogFile,
            outputFiles.TimeLeftLogFile,
            outputFiles.JsonLogFile,
            outputFiles.ProfileLogFile,
            outputFiles.TrainDir,
            GetJsonMeta(
                    catboostOptions.BoostingOptions->IterationCount.Get(),
                    outputFiles.ExperimentName,
                    metrics,
                    learnSetNames,
                    testSetNames,
                    /*parametersName=*/ "",
                    ELaunchMode::Train),
            metricPeriod,
            logger
    );


}


void InitializeFileLoggers(
    const NCatboostOptions::TOutputFilesOptions& outputFileOptions,
    const NJson::TJsonValue& metricsMetaJson,
    const TString& namesPrefix,
    bool isDetailedProfile,
    TLogger* logger
) {
    TOutputFiles outputFiles(outputFileOptions, namesPrefix);
    AddFileLoggers(
        isDetailedProfile,
        outputFiles.LearnErrorLogFile,
        outputFiles.TestErrorLogFile,
        outputFiles.TimeLeftLogFile,
        outputFiles.JsonLogFile,
        outputFiles.ProfileLogFile,
        outputFileOptions.GetTrainDir(),
        metricsMetaJson,
        outputFileOptions.GetMetricPeriod(),
        logger
    );
}
