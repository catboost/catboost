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

namespace NLoggingImpl {
    const size_t SingletonPriority = 500;
}

template <class T>
T* CreateDefaultLogger() {
    return nullptr;
}

namespace NLoggingImpl {
    template<class T, class TTraits>
    class TOperatorBase {
        struct TPtr {
            TPtr()
                : Instance(TTraits::CreateDefault())
            {
            }

            THolder<T> Instance;
        };

    public:
        inline static bool Usage() {
            return SingletonWithPriority<TPtr, SingletonPriority>()->Instance.Get();
        }

        inline static T* Get() {
            return SingletonWithPriority<TPtr, SingletonPriority>()->Instance.Get();
        }

        inline static void Set(T* v) {
            SingletonWithPriority<TPtr, SingletonPriority>()->Instance.Reset(v);
        }
    };

    template<class T>
    struct TLoggerTraits {
        static T* CreateDefault() {
            return CreateDefaultLogger<T>();
        }
    };
}

template <class T>
class TLoggerOperator : public NLoggingImpl::TOperatorBase<T, NLoggingImpl::TLoggerTraits<T>>  {
public:
    inline static TLog& Log() {
        Y_ASSERT(TLoggerOperator::Usage());
        return *TLoggerOperator::Get();
    }
};

namespace NLoggingImpl {

    TString GetLocalTimeSSimple();

    // Returns correct log type to use
    TString PrepareToOpenLog(TString logType, int logLevel, bool rotation, bool startAsDaemon);

    template <class TLoggerType>
    void InitLogImpl(TString logType, const int logLevel, const bool rotation, const bool startAsDaemon) {
        TLoggerOperator<TLoggerType>::Set(new TLoggerType(PrepareToOpenLog(logType, logLevel, rotation, startAsDaemon), (ELogPriority)logLevel));
    }
}

struct TLogRecordContext {
    constexpr TLogRecordContext(const TSourceLocation& sourceLocation, TStringBuf customMessage, ELogPriority priority)
        : SourceLocation(sourceLocation)
        , CustomMessage(customMessage)
        , Priority(priority)
    {}

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
