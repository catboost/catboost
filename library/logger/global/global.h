#pragma once

#include "common.h"
#include "rty_formater.h"

// ATTENTION! MUST CALL DoInitGlobalLog BEFORE USAGE

bool GlobalLogInitialized();
void DoInitGlobalLog(const TString& logType, const int logLevel, const bool rotation, const bool startAsDaemon);

inline void InitGlobalLog2Null() {
    DoInitGlobalLog("null", TLOG_EMERG, false, false);
}

inline void InitGlobalLog2Console(int loglevel = TLOG_INFO) {
    DoInitGlobalLog("console", loglevel, false, false);
}

class TGlobalLog: public TLog {
public:
    TGlobalLog(const TString& logType, TLogPriority priority = LOG_MAX_PRIORITY)
        : TLog(logType, priority)
    {
    }
};

template <>
TGlobalLog* CreateDefaultLogger<TGlobalLog>();

class TNullLog: public TLog {
public:
    TNullLog(const TString& logType, TLogPriority priority = LOG_MAX_PRIORITY)
        : TLog(logType, priority)
    {
    }
};

template <>
TNullLog* CreateDefaultLogger<TNullLog>();

template <>
class TSingletonTraits<TLoggerOperator<TGlobalLog>::TPtr> {
public:
    static const size_t Priority = NLoggingImpl::SingletonPriority;
};

template <>
class TSingletonTraits<TLoggerOperator<TNullLog>::TPtr> {
public:
    static const size_t Priority = NLoggingImpl::SingletonPriority;
};

#define FATAL_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TRTYLogPreprocessor, TLOG_CRIT, "CRITICAL_INFO")
#define ERROR_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TRTYLogPreprocessor, TLOG_ERR, "ERROR")
#define WARNING_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TRTYLogPreprocessor, TLOG_WARNING, "WARNING")
#define NOTICE_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TRTYLogPreprocessor, TLOG_NOTICE, "NOTICE")
#define INFO_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TRTYLogPreprocessor, TLOG_INFO, "INFO")
#define DEBUG_LOG SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TRTYLogPreprocessor, TLOG_DEBUG, "DEBUG")

#define TEMPLATE_LOG(logLevel) SINGLETON_CHECKED_GENERIC_LOG(TGlobalLog, TRTYLogPreprocessor, logLevel, ~ToString(logLevel))

#define IS_LOG_ACTIVE(logLevel) (TLoggerOperator<TGlobalLog>::Log().FiltrationLevel() >= logLevel)

#define RTY_MEM_LOG(Action) \
    { NOTICE_LOG << "RESOURCES On " << Action << ": " << NLoggingImpl::GetSystemResources() << Endl; };

#define VERIFY_WITH_LOG(expr, msg, ...)                       \
    do {                                                      \
        if (Y_UNLIKELY(!(expr))) {                            \
            FATAL_LOG << Sprintf(msg, ##__VA_ARGS__) << Endl; \
            Y_VERIFY(false, msg, ##__VA_ARGS__);              \
        };                                                    \
    } while (0);

namespace NPrivateGlobalLogger {
    class TVerifyEvent: public TStringStream {
    public:
        ~TVerifyEvent() {
            const TString info = Str();
            FATAL_LOG << info << Endl;
            Y_VERIFY(false, "%s", ~info);
        }
        template <class T>
        inline TVerifyEvent& operator<<(const T& t) {
            static_cast<TOutputStream&>(*this) << t;

            return *this;
        }
    };
}

#define CHECK_WITH_LOG(expr) \
    Y_UNLIKELY(!(expr)) && NPrivateGlobalLogger::TEatStream() | NPrivateGlobalLogger::TVerifyEvent() << __LOCATION__ << ": " << #expr << "(verification failed!): "

#define CHECK_EQ_WITH_LOG(a, b) CHECK_WITH_LOG((a) == (b)) << a << " != " << b;
#define CHECK_LEQ_WITH_LOG(a, b) CHECK_WITH_LOG((a) <= (b)) << a << " > " << b;

#define FAIL_LOG(msg, ...) VERIFY_WITH_LOG(false, msg, ##__VA_ARGS__)
#define S_FAIL_LOG CHECK_WITH_LOG(false)
