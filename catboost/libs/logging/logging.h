#pragma once

#include "logging_level.h"

#include <library/cpp/logger/backend.h>
#include <util/generic/singleton.h>
#include <util/system/src_location.h>
#include <util/stream/tempbuf.h>
#include <util/generic/yexception.h>
#include <atomic>

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
    size_t GetRegularMessageStartOffset() const {
        return RegularMessageStartOffset;
    }
    ~TCatboostLogEntry() override;
private:
    TCatboostLog* Parent;
    size_t RegularMessageStartOffset = 0;
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
    void ResetTraceBackend(THolder<TLogBackend>&& lowPriorityBackend = THolder<TLogBackend>());
    void RestoreDefaultBackend();

    void SetOutputExtendedInfo(bool value) {
        OutputExtendedInfo = value;
    }

    inline bool NeedExtendedInfo() const {
        return OutputExtendedInfo || HaveTraceLog;
    }

    inline bool FastLogFilter(ELogPriority priority) const {
        return HaveTraceLog || LogPriority >= priority;
    }

    void SetLogPriority(ELogPriority logPriority) {
        LogPriority = logPriority;
    }

    ELogPriority GetLogPriority() const {
        return LogPriority;
    }

private:
    bool OutputExtendedInfo = false;
    bool HaveTraceLog = false;
    ELogPriority LogPriority = TLOG_WARNING;
    class TImpl;
    THolder<TImpl> ImplHolder;

    std::atomic<bool> IsCustomBackendSpecified;
};

class TCatBoostLogSettings {
    Y_DECLARE_SINGLETON_FRIEND()
    TCatBoostLogSettings() = default;

public:
    using TSelf = TCatBoostLogSettings;
    TCatboostLog Log;
    inline static TSelf& GetRef() {
       return *Singleton<TSelf>();
    }
};

class TSetLogging { /* in current scope */
public:
    explicit TSetLogging(ELoggingLevel level)
        : SavedLogPriority(TCatBoostLogSettings::GetRef().Log.GetLogPriority())
    {
        ELogPriority newLogPriority = GetLogPriorityForLoggingLevel(level);
        RestoreLogPriority = SavedLogPriority != newLogPriority;
        TCatBoostLogSettings::GetRef().Log.SetLogPriority(newLogPriority);
    }

    ~TSetLogging() {
        if (RestoreLogPriority) {
            TCatBoostLogSettings::GetRef().Log.SetLogPriority(SavedLogPriority);
        }
    }

private:
    ELogPriority GetLogPriorityForLoggingLevel(ELoggingLevel level) {
        ELogPriority logPriority = TLOG_EMERG;
        switch (level) {
            case ELoggingLevel::Silent: {
                logPriority = TLOG_WARNING;
                break;
            }
            case ELoggingLevel::Verbose: {
                logPriority = TLOG_NOTICE;
                break;
            }
            case ELoggingLevel::Info: {
                logPriority = TLOG_INFO;
                break;
            }
            case ELoggingLevel::Debug: {
                logPriority = TLOG_DEBUG;
                break;
            }
            default:{
                ythrow yexception() << "Unknown logging level " << level;
            }
        }
        return logPriority;
    }

private:
    ELogPriority SavedLogPriority;
    bool RestoreLogPriority = false;
};

class TSetLoggingSilent : TSetLogging {
public:
    TSetLoggingSilent()
    : TSetLogging(ELoggingLevel::Silent)
    {}
};

class TSetLoggingVerbose : TSetLogging {
public:
    TSetLoggingVerbose()
    : TSetLogging(ELoggingLevel::Debug)
    {}
};

class TSetLoggingVerboseOrSilent : TSetLogging {
public:
    explicit TSetLoggingVerboseOrSilent(bool verbose)
    : TSetLogging(verbose ? ELoggingLevel::Debug : ELoggingLevel::Silent)
    {}
};

void ResetTraceBackend(const TString& string);

using TCustomLoggingObject = void*;
using TCustomLoggingFunction = void(*)(const char*, size_t len, TCustomLoggingObject);


void SetCustomLoggingFunction(
    TCustomLoggingFunction normalPriorityFunc,
    TCustomLoggingFunction errorPriorityFunc,
    TCustomLoggingObject normalPriorityObj = nullptr,
    TCustomLoggingObject errorPriorityObj = nullptr
);
void RestoreOriginalLogger();

namespace NPrivateCatboostLogger {
    struct TEatStream {
        Y_FORCE_INLINE bool operator|(const IOutputStream&) const {
            return true;
        }
    };
}

#define CATBOOST_GENERIC_LOG(level, msg) (TCatBoostLogSettings::GetRef().Log.FastLogFilter(level)) && NPrivateCatboostLogger::TEatStream() | TCatboostLogEntry(&TCatBoostLogSettings::GetRef().Log, __LOCATION__, msg, level)

#define CATBOOST_FATAL_LOG CATBOOST_GENERIC_LOG(TLOG_CRIT, "CRITICAL_INFO")
#define CATBOOST_ERROR_LOG CATBOOST_GENERIC_LOG(TLOG_ERR, "ERROR")
#define CATBOOST_WARNING_LOG CATBOOST_GENERIC_LOG(TLOG_WARNING, "WARNING")
#define CATBOOST_NOTICE_LOG CATBOOST_GENERIC_LOG(TLOG_NOTICE, "NOTICE")
#define CATBOOST_INFO_LOG CATBOOST_GENERIC_LOG(TLOG_INFO, "INFO")
#define CATBOOST_DEBUG_LOG CATBOOST_GENERIC_LOG(TLOG_DEBUG, "DEBUG")
