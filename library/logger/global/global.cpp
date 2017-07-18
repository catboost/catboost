#include "global.h"

void DoInitGlobalLog(const TString& logType, const int logLevel, const bool rotation, const bool startAsDaemon) {
    NLoggingImpl::InitLogImpl<TGlobalLog>(logType, logLevel, rotation, startAsDaemon);
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
