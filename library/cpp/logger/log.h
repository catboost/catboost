#pragma once

#include "backend.h"
#include "element.h"
#include "priority.h"
#include "record.h"
#include "thread.h"

#include <util/generic/fwd.h>
#include <util/generic/ptr.h>

#include <functional>
#include <cstdarg>

using TLogFormatter = std::function<TString(ELogPriority priority, TStringBuf)>;

// Logging facilities interface.
//
// ```cpp
// TLog base;
// ...
// auto log = base;
// log.SetFormatter([reqId](ELogPriority p, TStringBuf msg) {
//     return TStringBuilder() << "reqid=" << reqId << "; " << msg;
// });
//
// log.Write(TLOG_INFO, "begin");
// HandleRequest(...);
// log.Write(TLOG_INFO, "end");
// ```
//
// Users are encouraged to copy `TLog` instance.
class TLog {
public:
    // Construct empty logger all writes will be spilled.
    TLog();
    // Construct file logger.
    TLog(const TString& fname, ELogPriority priority = LOG_MAX_PRIORITY);
    // Construct any type of logger
    TLog(THolder<TLogBackend> backend);

    TLog(const TLog&);
    TLog(TLog&&);
    ~TLog();
    TLog& operator=(const TLog&);
    TLog& operator=(TLog&&);

    // Change underlying backend.
    // NOTE: not thread safe.
    void ResetBackend(THolder<TLogBackend> backend) noexcept;
    // Reset underlying backend, `IsNullLog()` will return `true` after this call.
    // NOTE: not thread safe.
    THolder<TLogBackend> ReleaseBackend() noexcept;
    // Check if underlying backend is defined and is not null.
    // NOTE: not thread safe with respect to `ResetBackend` and `ReleaseBackend`.
    bool IsNullLog() const noexcept;
    bool IsNotNullLog() const noexcept {
        return !IsNullLog();
    }

    // Write message to the log.
    //
    // @param[in] priority          Message priority to use.
    // @param[in] message           Message to write.
    // @param[in] metaFlags         Message meta flags.
    void Write(ELogPriority priority, TStringBuf message, TLogRecord::TMetaFlags metaFlags = {}) const;
    // Write message to the log using `DefaultPriority()`.
    void Write(const char* data, size_t len, TLogRecord::TMetaFlags metaFlags = {}) const;
    // Write message to the log, but pass the message in a c-style.
    void Write(ELogPriority priority, const char* data, size_t len, TLogRecord::TMetaFlags metaFlags = {}) const;

    // Write message to the log in a c-like printf style.
    void Y_PRINTF_FORMAT(3, 4) AddLog(ELogPriority priority, const char* format, ...) const;
    // Write message to the log in a c-like printf style with `DefaultPriority()` priority.
    void Y_PRINTF_FORMAT(2, 3) AddLog(const char* format, ...) const;

    // Call `ReopenLog()` of the underlying backend.
    void ReopenLog();
    // Call `ReopenLogNoFlush()` of the underlying backend.
    void ReopenLogNoFlush();
    // Call `QueueSize()` of the underlying backend.
    size_t BackEndQueueSize() const;

    // Set log default priority.
    // NOTE: not thread safe.
    void SetDefaultPriority(ELogPriority priority) noexcept;
    // Get default priority
    ELogPriority DefaultPriority() const noexcept;

    // Call `FiltrationLevel()` of the underlying backend.
    ELogPriority FiltrationLevel() const noexcept;

    // Set current log formatter.
    void SetFormatter(TLogFormatter formatter) noexcept;

    template <class T>
    inline TLogElement operator<<(const T& t) const {
        TLogElement ret(this);
        ret << t;
        return ret;
    }

public:
    // These methods are deprecated and present here only for compatibility reasons (for 13 years
    // already ...). Do not use them.
    bool OpenLog(const char* path, ELogPriority lp = LOG_MAX_PRIORITY);
    bool IsOpen() const noexcept;
    void AddLogVAList(const char* format, va_list lst);
    void CloseLog();

private:
    class TImpl;
    TIntrusivePtr<TImpl> Impl_;
    TLogFormatter Formatter_;
};

THolder<TLogBackend> CreateLogBackend(const TString& fname, ELogPriority priority = LOG_MAX_PRIORITY, bool threaded = false);
THolder<TLogBackend> CreateFilteredOwningThreadedLogBackend(const TString& fname, ELogPriority priority = LOG_MAX_PRIORITY, size_t queueLen = 0);
THolder<TOwningThreadedLogBackend> CreateOwningThreadedLogBackend(const TString& fname, size_t queueLen = 0);
