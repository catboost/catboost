#pragma once

#include <util/datetime/base.h>

#include <util/folder/path.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/generic/ptr.h>
#include <util/generic/yexception.h>
#include <util/string/printf.h>
#include <util/system/src_location.h>

#include <library/cpp/logger/log.h>

template <class T>
T* CreateDefaultLogger() {
    return nullptr;
}

template <class T>
class TLoggerOperator {
public:
    struct TPtr {
        TPtr()
            : Log(CreateDefaultLogger<T>())
        {
        }

        THolder<T> Log;
    };

    inline static bool Usage() {
        return Singleton<TPtr>()->Log.Get();
    }

    inline static TLog& Log() {
        Y_ASSERT(Usage());
        return *Singleton<TPtr>()->Log.Get();
    }

    inline static T* Get() {
        return Singleton<TPtr>()->Log.Get();
    }

    inline static void Set(T* log) {
        Singleton<TPtr>()->Log.Reset(log);
    }
};

namespace NLoggingImpl {
    const size_t SingletonPriority = 500;

    TString GetLocalTimeSSimple();

    template <class TLoggerType>
    void InitLogImpl(TString logType, const int logLevel, const bool rotation, const bool startAsDaemon) {
        if (logLevel < 0 || logLevel > (int)LOG_MAX_PRIORITY)
            ythrow yexception() << "Incorrect priority";
        if (rotation && TFsPath(logType).Exists()) {
            TString newPath = Sprintf("%s_%s_%" PRIu64, logType.data(), NLoggingImpl::GetLocalTimeSSimple().data(), static_cast<ui64>(Now().MicroSeconds()));
            TFsPath(logType).RenameTo(newPath);
        }
        if (startAsDaemon && (logType == "console" || logType == "cout" || logType == "cerr")) {
            logType = "null";
        }
        TLoggerOperator<TLoggerType>::Set(new TLoggerType(logType, (ELogPriority)logLevel));
    }
}

struct TLogRecordContext {
    TLogRecordContext(const TSourceLocation& sourceLocation, const char* customMessage, ELogPriority priority);

    TSourceLocation SourceLocation;
    TStringBuf CustomMessage;
    ELogPriority Priority;
};

template <class... R>
struct TLogRecordPreprocessor;

template <>
struct TLogRecordPreprocessor<> {
    inline static bool CheckLoggingContext(TLog& /*log*/, const TLogRecordContext& /*context*/) {
        return true;
    }

    inline static TSimpleSharedPtr<TLogElement> StartRecord(TLog& log, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier) {
        if (earlier)
            return earlier;
        TSimpleSharedPtr<TLogElement> result(new TLogElement(&log));
        *result << context.Priority;
        return result;
    }
};

template <class H, class... R>
struct TLogRecordPreprocessor<H, R...> {
    inline static bool CheckLoggingContext(TLog& log, const TLogRecordContext& context) {
        return H::CheckLoggingContext(log, context) && TLogRecordPreprocessor<R...>::CheckLoggingContext(log, context);
    }

    inline static TSimpleSharedPtr<TLogElement> StartRecord(TLog& log, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier) {
        TSimpleSharedPtr<TLogElement> first = H::StartRecord(log, context, earlier);
        return TLogRecordPreprocessor<R...>::StartRecord(log, context, first);
    }
};

struct TLogFilter {
    static bool CheckLoggingContext(TLog& log, const TLogRecordContext& context);
    static TSimpleSharedPtr<TLogElement> StartRecord(TLog& log, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier);
};

class TNullLog;

template <class TPreprocessor>
TSimpleSharedPtr<TLogElement> GetLoggerForce(TLog& log, const TLogRecordContext& context) {
    if (TSimpleSharedPtr<TLogElement> result = TPreprocessor::StartRecord(log, context, nullptr))
        return result;
    return new TLogElement(&TLoggerOperator<TNullLog>::Log());
}

namespace NPrivateGlobalLogger {
    struct TEatStream {
        Y_FORCE_INLINE bool operator|(const IOutputStream&) const {
            return true;
        }
    };
}

#define LOGGER_GENERIC_LOG_CHECKED(logger, preprocessor, level, message) (*GetLoggerForce<preprocessor>(logger, TLogRecordContext(__LOCATION__, message, level)))
#define LOGGER_CHECKED_GENERIC_LOG(logger, preprocessor, level, message) \
    (preprocessor::CheckLoggingContext(logger, TLogRecordContext(__LOCATION__, message, level))) && NPrivateGlobalLogger::TEatStream() | (*(preprocessor::StartRecord(logger, TLogRecordContext(__LOCATION__, message, level), nullptr)))

#define SINGLETON_GENERIC_LOG_CHECKED(type, preprocessor, level, message) LOGGER_GENERIC_LOG_CHECKED(TLoggerOperator<type>::Log(), preprocessor, level, message)
#define SINGLETON_CHECKED_GENERIC_LOG(type, preprocessor, level, message) LOGGER_CHECKED_GENERIC_LOG(TLoggerOperator<type>::Log(), preprocessor, level, message)
