#include "logger.h"

static THolder<TOFStream> InitErrLog(const yvector<THolder<IMetric>>& errors, const yvector<yvector<double>>& history, const TString& logName) {
    THolder<TOFStream> errLog = new TOFStream(logName);
    *errLog << "iter";
    for (const auto& error : errors) {
        *errLog << "\t" << error->GetDescription();
    }
    *errLog << Endl;
    for (const auto& errors : history) {
        for (const auto& error : errors) {
            *errLog << "\t" << error;
        }
        *errLog << Endl;
    }
    return errLog;
}

THolder<TLogger> InitLogger(const yvector<THolder<IMetric>>& errors, TLearnContext& ctx, const bool hasTest) {
    THolder<TLogger> logger = new TLogger();
    logger->LearnErrLog = InitErrLog(errors, ctx.LearnProgress.LearnErrorsHistory, ctx.Files.LearnErrorLogFile);
    if (hasTest) {
        logger->TestErrLog = InitErrLog(errors, ctx.LearnProgress.TestErrorsHistory, ctx.Files.TestErrorLogFile);
    }
    logger->LearnTensorBoardLogger = new TTensorBoardLogger(JoinFsPaths(ctx.Params.TrainDir, "train"));
    logger->TestTensorBoardLogger = new TTensorBoardLogger(JoinFsPaths(ctx.Params.TrainDir, "test"));
    return logger;
}

void Log(int iteration, const yvector<double>& errorsHistory, const yvector<THolder<IMetric>>& errors, TLogger* logger, const EPhase phase) {
    TOFStream* errLog = logger->LearnErrLog.Get();
    TTensorBoardLogger* tensorBoardLogger = logger->LearnTensorBoardLogger.Get();
    if (phase == EPhase::Test) {
        errLog = logger->TestErrLog.Get();
        tensorBoardLogger = logger->TestTensorBoardLogger.Get();
    }
    *errLog << iteration;
    for (int i = 0; i < errorsHistory.ysize(); ++i) {
        *errLog << "\t" << errorsHistory[i];
        tensorBoardLogger->AddScalar(errors[i]->GetDescription(), iteration, errorsHistory[i]);
    }
    *errLog << Endl;
}
