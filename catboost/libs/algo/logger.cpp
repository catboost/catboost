#include "logger.h"

static void OutputLineToErrLog(const TVector<double>& history, const int iteration, TOFStream* errLog) {
    *errLog << iteration;
    for (const auto& error : history) {
        *errLog << "\t" << error;
    }
    *errLog << Endl;
}

static void OutputLineToTensorBoardLog(const TVector<double>& history, const int iteration,
        const TVector<THolder<IMetric>>& errors, TTensorBoardLogger* tensorBoardLogger) {
    for (int i = 0; i < history.ysize(); ++i) {
        tensorBoardLogger->AddScalar(errors[i]->GetDescription(), iteration, history[i]);
    }
}

static THolder<TOFStream> CreateErrLog(const TVector<THolder<IMetric>>& errors, const TVector<TVector<double>>& history, const TString& logName) {
    THolder<TOFStream> errLog = new TOFStream(logName);
    *errLog << "iter";
    for (const auto& error : errors) {
        *errLog << "\t" << error->GetDescription();
    }
    *errLog << Endl;
    for (int iteration = 0; iteration < history.ysize(); ++iteration) {
        OutputLineToErrLog(history[iteration], iteration, errLog.Get());
    }
    return errLog;
}

static THolder<TTensorBoardLogger> CreateTensorBoardLog(const TVector<THolder<IMetric>>& errors,
        const TVector<TVector<double>>& history, const TString& logName) {
    THolder<TTensorBoardLogger> tensorBoardLogger = new TTensorBoardLogger(logName);
    for (int iteration = 0; iteration < history.ysize(); ++iteration) {
        OutputLineToTensorBoardLog(history[iteration], iteration, errors, tensorBoardLogger.Get());
    }
    return tensorBoardLogger;
}


THolder<TLogger> CreateLogger(const TVector<THolder<IMetric>>& errors, TLearnContext& ctx, const bool hasTest) {
    THolder<TLogger> logger = new TLogger();
    logger->LearnErrLog = CreateErrLog(errors, ctx.LearnProgress.LearnErrorsHistory, ctx.Files.LearnErrorLogFile);
    if (hasTest) {
        logger->TestErrLog = CreateErrLog(errors, ctx.LearnProgress.TestErrorsHistory, ctx.Files.TestErrorLogFile);
    }
    logger->LearnTensorBoardLogger = CreateTensorBoardLog(errors, ctx.LearnProgress.LearnErrorsHistory, JoinFsPaths(ctx.Params.TrainDir, "train"));
    logger->TestTensorBoardLogger = CreateTensorBoardLog(errors, ctx.LearnProgress.TestErrorsHistory, JoinFsPaths(ctx.Params.TrainDir, "test"));
    return logger;
}

void Log(int iteration, const TVector<double>& errorsHistory, const TVector<THolder<IMetric>>& errors, TLogger* logger, const EPhase phase) {
    TOFStream* errLog = logger->LearnErrLog.Get();
    TTensorBoardLogger* tensorBoardLogger = logger->LearnTensorBoardLogger.Get();
    if (phase == EPhase::Test) {
        errLog = logger->TestErrLog.Get();
        tensorBoardLogger = logger->TestTensorBoardLogger.Get();
    }
    OutputLineToErrLog(errorsHistory, iteration, errLog);
    OutputLineToTensorBoardLog(errorsHistory, iteration, errors, tensorBoardLogger);
}
