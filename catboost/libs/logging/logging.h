#pragma once

#include "logging_level.h"

#include <library/logger/backend.h>
#include <util/generic/singleton.h>
#include <util/system/src_location.h>
#include <util/stream/tempbuf.h>
#include <util/generic/yexception.h>

class TCatboostLog;

class TCatboostLogEntry : public TTempBufOutput {
public:
    TCatboostLogEntry(TCatboostLog* parent, const TSourceLocation& sourceLocation, TStringBuf customMessage, ELogPriority priority);
    void DoFlush() override;

    template <class T>
    inline TCatboostLogEntry& operator<<(const T& t) {
        static_cast<IOutputStream&>(*this) << t;

        return *this;
    }

    ~TCatboostLogEntry() override;
private:
    TCatboostLog* Parent;
public:
    const TSourceLocation SourceLocation;
    const TStringBuf CustomMessage;
    const ELogPriority Priority;
};

class TCatboostLog {
public:
    TCatboostLog();
    ~TCatboostLog();
    void Output(const TCatboostLogEntry& entry);
    void ResetBackend(THolder<TLogBackend>&& lowPriorityBackend, THolder<TLogBackend>&& highPriorityBackend);
    void RestoreDefaultBackend();
private:
    class TImpl;
    THolder<TImpl> ImplHolder;
};

class TMatrixnetLogSettings {
    Y_DECLARE_SINGLETON_FRIEND();
    TMatrixnetLogSettings() = default;

public:
    using TSelf = TMatrixnetLogSettings;
    TCatboostLog Log;
    inline static TSelf& GetRef() {
        return *Singleton<TSelf>();
    }

    bool OutputExtendedInfo = false;
    ELogPriority LogPriority = TLOG_WARNING;
};

inline void SetLogingLevel(ELoggingLevel level) {
    switch (level) {
        case ELoggingLevel::Silent:{
            TMatrixnetLogSettings::GetRef().LogPriority = TLOG_WARNING;
            break;
        }
        case ELoggingLevel::Verbose: {
            TMatrixnetLogSettings::GetRef().LogPriority = TLOG_NOTICE;
            break;
        }
        case ELoggingLevel::Info: {
            TMatrixnetLogSettings::GetRef().LogPriority = TLOG_INFO;
            break;
        }
        case ELoggingLevel::Debug: {
            TMatrixnetLogSettings::GetRef().LogPriority = TLOG_DEBUG;
            break;
        }
        default:{
            ythrow yexception() << "Unknown logging level " << level;
        }
    }
}

inline void SetSilentLogingMode() {
    SetLogingLevel(ELoggingLevel::Silent);
}

inline void SetVerboseLogingMode() {
    SetLogingLevel(ELoggingLevel::Debug);
}

using TCustomLoggingFunction = void(*)(const char*, size_t len);

void SetCustomLoggingFunction(TCustomLoggingFunction lowPriorityFunc, TCustomLoggingFunction highPriorityFunc);
void RestoreOriginalLogger();

namespace NPrivateCatboostLogger {
    struct TEatStream {
        Y_FORCE_INLINE bool operator|(const IOutputStream&) const {
            return true;
        }
    };
}

#define MATRIXNET_GENERIC_LOG(level, msg) (TMatrixnetLogSettings::GetRef().LogPriority >= level) && NPrivateCatboostLogger::TEatStream() | TCatboostLogEntry(&TMatrixnetLogSettings::GetRef().Log, __LOCATION__, msg, level)

#define MATRIXNET_FATAL_LOG MATRIXNET_GENERIC_LOG(TLOG_CRIT, "CRITICAL_INFO")
#define MATRIXNET_ERROR_LOG MATRIXNET_GENERIC_LOG(TLOG_ERR, "ERROR")
#define MATRIXNET_WARNING_LOG MATRIXNET_GENERIC_LOG(TLOG_WARNING, "WARNING")
#define MATRIXNET_NOTICE_LOG MATRIXNET_GENERIC_LOG(TLOG_NOTICE, "NOTICE")
#define MATRIXNET_INFO_LOG MATRIXNET_GENERIC_LOG(TLOG_INFO, "INFO")
#define MATRIXNET_DEBUG_LOG MATRIXNET_GENERIC_LOG(TLOG_DEBUG, "DEBUG")
