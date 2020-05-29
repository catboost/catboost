#include "global.h"

void DoInitGlobalLog(const TString& logType, const int logLevel, const bool rotation, const bool startAsDaemon) {
    NLoggingImpl::InitLogImpl<TGlobalLog>(logType, logLevel, rotation, startAsDaemon);
}

void DoInitGlobalLog(THolder<TLogBackend> backend) {
    TLoggerOperator<TGlobalLog>::Set(new TGlobalLog(std::move(backend)));
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
    Y_FAIL("%s", info.data());
}
