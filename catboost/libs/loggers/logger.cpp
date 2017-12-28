#include "logger.h"

static void WriteHistory(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<double>>& testErrorsHistory,
    const TVector<TVector<double>> timeHistory,
    TLogger* logger
) {
    for (int iteration = 0; iteration < timeHistory.ysize(); ++iteration) {
        TOneInterationLogger oneIterLogger(*logger);
        if (iteration < learnErrorsHistory.ysize()) {
            const TVector<double>& learnErrors = learnErrorsHistory[iteration];
            for (int i = 0; i < learnErrors.ysize(); ++i) {
                oneIterLogger.OutputMetric("learn", TMetricEvalResult(metricsDescription[i], learnErrors[i], i == 0));
            }

        }
        if (iteration < testErrorsHistory.ysize()) {
            const TVector<double>& testErrors = testErrorsHistory[iteration];
            for (int i = 0; i < testErrors.ysize(); ++i) {
                oneIterLogger.OutputMetric("test", TMetricEvalResult(metricsDescription[i], testErrors[i], i == 0));
            }
        }
        oneIterLogger.OutputProfile(TProfileResults(timeHistory[iteration][0], timeHistory[iteration][1]));
    }
}

TLogger CreateLogger(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<double>>& testErrorsHistory,
    const TVector<TVector<double>>& timeHistory,
    const TString learnErrorLogFile,
    const TString testErrorLogFile,
    const TString timeLogFile,
    const TString trainDir,
    const bool allowWriteFiles,
    const bool detailedProfile,
    const bool hasTrain,
    const bool hasTest
) {
    TLogger logger;
    TIntrusivePtr<ILoggingBackend> consoleLoggingBackend = new TConsoleLoggingBackend(detailedProfile);
    if (hasTrain) {
        logger.AddBackend("learn", consoleLoggingBackend);
        if (allowWriteFiles) {
            logger.AddBackend("learn", TIntrusivePtr<ILoggingBackend>(new TErrorFileLoggingBackend(learnErrorLogFile)));
            logger.AddBackend("learn", TIntrusivePtr<ILoggingBackend>(new TTensorBoardLoggingBackend(JoinFsPaths(trainDir, "train"))));
        }
    }
    if (hasTest) {
        logger.AddBackend("test", consoleLoggingBackend);
        if (allowWriteFiles) {
            logger.AddBackend("test", TIntrusivePtr<ILoggingBackend>(new TErrorFileLoggingBackend(testErrorLogFile)));
            logger.AddBackend("test", TIntrusivePtr<ILoggingBackend>(new TTensorBoardLoggingBackend(JoinFsPaths(trainDir, "test"))));
        }
    }
    logger.AddProfileBackend(consoleLoggingBackend);
    if (allowWriteFiles) {
        logger.AddProfileBackend(TIntrusivePtr<ILoggingBackend>(new TTimeFileLoggingBackend(timeLogFile)));
    }

    WriteHistory(
        metricsDescription,
        learnErrorsHistory,
        testErrorsHistory,
        timeHistory,
        &logger
    );
    return logger;
}

void Log(
    const TVector<TString>& metricsDescription,
    const TVector<TVector<double>>& learnErrorsHistory,
    const TVector<TVector<double>>& testErrorsHistory,
    const double bestErrorValue,
    const int bestIteration,
    const TProfileResults& profileResults,
    TLogger* logger
) {
    TOneInterationLogger oneIterLogger(*logger);
    int iteration = profileResults.PassedIterations - 1;
    if (iteration < learnErrorsHistory.ysize()) {
        const TVector<double>& learnErrors = learnErrorsHistory[iteration];
        for (int i = 0; i < learnErrors.ysize(); ++i) {
            oneIterLogger.OutputMetric("learn", TMetricEvalResult(metricsDescription[i], learnErrors[i], i == 0));
        }

    }
    if (iteration < testErrorsHistory.ysize()) {
        const TVector<double>& testErrors = testErrorsHistory[iteration];
        for (int i = 0; i < testErrors.ysize(); ++i) {
            oneIterLogger.OutputMetric("test", TMetricEvalResult(metricsDescription[i], testErrors[i], bestErrorValue, bestIteration, i == 0));
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
