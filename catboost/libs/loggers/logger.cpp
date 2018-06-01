#include "logger.h"

void WriteHistory(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<TVector<double>>>& testErrorsHistory, // [iter][test][metric]
    const TVector<TVector<double>>& timeHistory,
    const TString& learnToken,
    const TVector<const TString>& testTokens,
    TLogger* logger
) {
    for (int iteration = 0; iteration < timeHistory.ysize(); ++iteration) {
        TOneInterationLogger oneIterLogger(*logger);
        if (iteration < learnErrorsHistory.ysize()) {
            const TVector<double>& learnErrors = learnErrorsHistory[iteration];
            for (int i = 0; i < learnErrors.ysize(); ++i) {
                oneIterLogger.OutputMetric(learnToken, TMetricEvalResult(metricsDescription[i], learnErrors[i], i == 0));
            }
        }
        if (iteration < testErrorsHistory.ysize()) {
            const int testCount = testErrorsHistory[0].ysize();
            for (int testIdx = 0; testIdx < testCount; ++testIdx) {
                const TVector<double>& testErrors = testErrorsHistory[iteration][testIdx];
                for (int i = 0; i < testErrors.ysize(); ++i) {
                    oneIterLogger.OutputMetric(testTokens[testIdx], TMetricEvalResult(metricsDescription[i], testErrors[i], i == 0));
                }
            }
        }
        oneIterLogger.OutputProfile(TProfileResults(timeHistory[iteration][0], timeHistory[iteration][1]));
    }
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
    const TVector<TString>& metricsDescription,
    const TVector<bool>& skipMetricOnTrain,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<TVector<double>>>& testErrorsHistory, // [iter][test][metric]
    double bestErrorValue,
    int bestIteration,
    const TProfileResults& profileResults,
    const TString& learnToken,
    const TVector<const TString>& testTokens,
    bool outputErrors,
    TLogger* logger
) {
    TOneInterationLogger oneIterLogger(*logger);
    int iteration = profileResults.PassedIterations - 1;
    if (outputErrors && iteration < learnErrorsHistory.ysize()) {
        const TVector<double>& learnErrors = learnErrorsHistory[iteration];
        size_t metricIdx = 0;
        for (int i = 0; i < learnErrors.ysize(); ++i) {
            while (skipMetricOnTrain[metricIdx]) {
                ++metricIdx;
            }
            oneIterLogger.OutputMetric(learnToken, TMetricEvalResult(metricsDescription[metricIdx], learnErrors[i], metricIdx == 0));
            ++metricIdx;
        }
    }
    if (outputErrors && iteration < testErrorsHistory.ysize()) {
        const int testCount = testErrorsHistory[iteration].ysize();
        for (int testIdx = 0; testIdx < testCount; ++testIdx) {
            const int metricCount = testErrorsHistory[iteration][testIdx].ysize();
            for (int metricIdx = 0; metricIdx < metricCount; ++metricIdx) {
                double testError = testErrorsHistory[iteration][testIdx][metricIdx];
                bool isMainMetric = metricIdx == 0;
                const TString& token = testTokens[testIdx];
                if (testIdx == 0) {
                    // Only test 0 should be followed by 'best:'
                    oneIterLogger.OutputMetric(token, TMetricEvalResult(metricsDescription[metricIdx], testError, bestErrorValue, bestIteration, isMainMetric));
                } else {
                    oneIterLogger.OutputMetric(token, TMetricEvalResult(metricsDescription[metricIdx] + ":" + ToString(testIdx), testError, isMainMetric));
                }
            }
        }
    }

    oneIterLogger.OutputProfile(profileResults);
}

void LogAverages(const TProfileResults& profileResults) {
    MATRIXNET_NOTICE_LOG << Endl << "Average times:" << Endl;
    if (profileResults.PassedIterations == 0) {
        MATRIXNET_NOTICE_LOG << Endl << "No iterations recorded" << Endl;
        return;
    }

    double time = 0;
    for (const auto& it : profileResults.OperationToTimeInAllIterations) {
        time += it.second;
    }
    time /= profileResults.PassedIterations;
    MATRIXNET_NOTICE_LOG << "Iteration time: " << FloatToString(time, PREC_NDIGITS, 3) << " sec" << Endl;

    for (const auto& it : profileResults.OperationToTimeInAllIterations) {
        MATRIXNET_NOTICE_LOG << it.first << ": "
            << FloatToString(it.second / profileResults.PassedIterations, PREC_NDIGITS, 3) << " sec" << Endl;
    }
    MATRIXNET_NOTICE_LOG << Endl;
}
