#pragma once

#include <library/logger/global/global.h>
#include <util/generic/singleton.h>

class TMatrixnetLogSettings {
    Y_DECLARE_SINGLETON_FRIEND();
    TMatrixnetLogSettings() = default;

public:
    using TSelf = TMatrixnetLogSettings;

    inline static TSelf& GetRef() {
        return *Singleton<TSelf>();
    }

    bool OutputExtendedInfo = false;
    TLogPriority LogPriority = TLOG_WARNING;
};

struct TMatrixnetMessageFormater {
    static bool CheckLoggingContext(TLog& logger, const TLogRecordContext& context);
    static TSimpleSharedPtr<TLogElement> StartRecord(TLog& logger, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier);
};

inline void SetVerboseLogingMode() {
    TMatrixnetLogSettings::GetRef().LogPriority = TLOG_INFO;
}

inline void SetSilentLogingMode() {
    TMatrixnetLogSettings::GetRef().LogPriority = TLOG_WARNING;
}

using TCustomLoggingFunction = void(*)(const char*, size_t len);

void SetCustomLoggingFunction(TCustomLoggingFunction func);
void RestoreOriginalLogger();

#define MATRIXNET_FATAL_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TMatrixnetMessageFormater, TLOG_CRIT, "CRITICAL_INFO")
#define MATRIXNET_ERROR_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TMatrixnetMessageFormater, TLOG_ERR, "ERROR")
#define MATRIXNET_WARNING_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TMatrixnetMessageFormater, TLOG_WARNING, "WARNING")
#define MATRIXNET_NOTICE_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TMatrixnetMessageFormater, TLOG_NOTICE, "NOTICE")
#define MATRIXNET_INFO_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TMatrixnetMessageFormater, TLOG_INFO, "INFO")
#define MATRIXNET_DEBUG_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TMatrixnetMessageFormater, TLOG_DEBUG, "DEBUG")
