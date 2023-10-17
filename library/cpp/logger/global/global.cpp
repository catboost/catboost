#include "global.h"

static void DoInitGlobalLog(THolder<TGlobalLog> logger, THolder<ILoggerFormatter> formatter) {
    TLoggerOperator<TGlobalLog>::Set(logger.Release());
    if (!formatter) {
        formatter.Reset(CreateRtyLoggerFormatter());
    }
    TLoggerFormatterOperator::Set(formatter.Release());
}

void DoInitGlobalLog(const TString& logType, const int logLevel, const bool rotation, const bool startAsDaemon, THolder<ILoggerFormatter> formatter, bool threaded) {
    DoInitGlobalLog(
        MakeHolder<TGlobalLog>(
            CreateLogBackend(
                NLoggingImpl::PrepareToOpenLog(logType, logLevel, rotation, startAsDaemon),
                (ELogPriority)logLevel,
                threaded)),
        std::move(formatter));
}

void DoInitGlobalLog(THolder<TLogBackend> backend, THolder<ILoggerFormatter> formatter) {
    DoInitGlobalLog(THolder(new TGlobalLog(std::move(backend))), std::move(formatter));
}

bool GlobalLogInitialized() {
    return TLoggerOperator<TGlobalLog>::Usage();
}

template <>
TGlobalLog* CreateDefaultLogger<TGlobalLog>() {
    return new TGlobalLog("console", TLOG_INFO);
}

template <>
TNullLog* CreateDefaultLogger<TNullLog>() {
    return new TNullLog("null");
}

NPrivateGlobalLogger::TVerifyEvent::~TVerifyEvent() {
    const TString info = Str();
    FATAL_LOG << info << Endl;
    Y_ABORT("%s", info.data());
}
