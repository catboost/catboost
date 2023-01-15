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

class TLog {
public:
    // Construct empty logger all data will be sent to `/dev/null`
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

    /*
     * NOT thread-safe
     */
    void ResetBackend(THolder<TLogBackend> backend) noexcept;
    THolder<TLogBackend> ReleaseBackend() noexcept;
    bool IsNullLog() const noexcept;

    void Write(const char* data, size_t len) const;
    void Write(ELogPriority priority, const char* data, size_t len) const;
    void Write(ELogPriority priority, TStringBuf data) const;
    void Y_PRINTF_FORMAT(2, 3) AddLog(const char* format, ...) const;
    void Y_PRINTF_FORMAT(3, 4) AddLog(ELogPriority priority, const char* format, ...) const;
    void ReopenLog();
    void ReopenLogNoFlush();
    size_t BackEndQueueSize() const;

    /*
     * This affects all write methods without priority argument
     */
    void SetDefaultPriority(ELogPriority priority) noexcept;
    ELogPriority DefaultPriority() const noexcept;

    ELogPriority FiltrationLevel() const noexcept;

    template <class T>
    inline TLogElement operator<<(const T& t) const {
        TLogElement ret(this);
        ret << t;
        return ret;
    }

    void SetFormatter(TLogFormatter formatter) noexcept;

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
