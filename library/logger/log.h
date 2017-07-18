#pragma once

#include "record.h"
#include "element.h"
#include "backend.h"
#include "priority.h"

#include <util/generic/ptr.h>
#include <functional>
#include <cstdarg>

class TString;

using TLogFormatter = std::function<TString(TLogPriority priority, TStringBuf)>;

class TLog {
public:
    /*
         * construct empty logger
         */
    TLog();

    /*
         * construct file logger
         */
    TLog(const TString& fname, TLogPriority priority = LOG_MAX_PRIORITY);

    /*
         * construct any type of logger :)
         */
    TLog(TAutoPtr<TLogBackend> backend);

    ~TLog();

    /*
         * NOT thread-safe
         */
    void ResetBackend(TAutoPtr<TLogBackend> backend) noexcept;
    TAutoPtr<TLogBackend> ReleaseBackend() noexcept;
    bool IsNullLog() const noexcept;

    void Write(const char* data, size_t len) const;
    void Write(TLogPriority priority, const char* data, size_t len) const;
    void Y_PRINTF_FORMAT(2, 3) AddLog(const char* format, ...) const;
    void Y_PRINTF_FORMAT(3, 4) AddLog(TLogPriority priority, const char* format, ...) const;
    void ReopenLog();

    /*
         * compat methods, remove in near future...
         */
    bool OpenLog(const char* path, TLogPriority lp = LOG_MAX_PRIORITY);
    bool IsOpen() const noexcept;
    void AddLogVAList(const char* format, va_list lst);
    void CloseLog();

    /*
         * This affects all write methods without priority argument
         */
    void SetDefaultPriority(TLogPriority priority) noexcept;
    TLogPriority DefaultPriority() const noexcept;

    TLogPriority FiltrationLevel() const noexcept;

    template <class T>
    inline TLogElement operator<<(const T& t) {
        TLogElement ret(this);
        ret << t;
        return ret;
    }

    void SetFormatter(TLogFormatter formatter) noexcept;

private:
    class TImpl;
    TSimpleIntrusivePtr<TImpl> Impl_;
    TLogFormatter Formatter;
};
