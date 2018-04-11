#include "logger.h"

void WriteHistory(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<double>>& testErrorsHistory,
    const TVector<TVector<double>>& timeHistory,
    const TString& learnToken,
    const TString& testToken,
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
            const TVector<double>& testErrors = testErrorsHistory[iteration];
            for (int i = 0; i < testErrors.ysize(); ++i) {
                oneIterLogger.OutputMetric(testToken, TMetricEvalResult(metricsDescription[i], testErrors[i], i == 0));
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
    for (auto& jsonToken : metaJson["test_sets"].GetArraySafe()) {
        TString token = jsonToken.GetString();
        logger->AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TErrorFileLoggingBackend(testErrorLogFile)));
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
    const TString& testToken,
    bool hasTrain,
    bool hasTest,
    int metricPeriod,
    TLogger* logger
) {
    TIntrusivePtr<ILoggingBackend> consoleLoggingBackend = new TConsoleLoggingBackend(/*detailedProfile=*/false, metricPeriod);
    if (hasTrain) {
        logger->AddBackend(learnToken, consoleLoggingBackend);
    }
    if (hasTest) {
        logger->AddBackend(testToken, consoleLoggingBackend);
    }
    logger->AddProfileBackend(consoleLoggingBackend);
}

void Log(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<double>>& testErrorsHistory,
    double bestErrorValue,
    int bestIteration,
    const TProfileResults& profileResults,
    const TString& learnToken,
    const TString& testToken,
    TLogger* logger
) {
    TOneInterationLogger oneIterLogger(*logger);
    int iteration = profileResults.PassedIterations - 1;
    if (iteration < learnErrorsHistory.ysize()) {
        const TVector<double>& learnErrors = learnErrorsHistory[iteration];
        for (int i = 0; i < learnErrors.ysize(); ++i) {
            oneIterLogger.OutputMetric(learnToken, TMetricEvalResult(metricsDescription[i], learnErrors[i], i == 0));
        }

    }
    if (iteration < testErrorsHistory.ysize()) {
        const TVector<double>& testErrors = testErrorsHistory[iteration];
        for (int i = 0; i < testErrors.ysize(); ++i) {
            oneIterLogger.OutputMetric(testToken, TMetricEvalResult(metricsDescription[i], testErrors[i], bestErrorValue, bestIteration, i == 0));
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
