#include "log.h"
#include "file.h"
#include "stream.h"
#include "null.h"

#include <util/string/cast.h>
#include <util/stream/printf.h>
#include <util/system/yassert.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include "filter.h"

static inline TAutoPtr<TLogBackend> BackendFactory(const TString& logType, TLogPriority priority) {
    try {
        if (priority != LOG_MAX_PRIORITY) {
            if (logType == "console") {
                return new TFilteredLogBackend<TStreamLogBackend>(new TStreamLogBackend(&Cerr), priority);
            }
            if (logType == "cout") {
                return new TFilteredLogBackend<TStreamLogBackend>(new TStreamLogBackend(&Cout), priority);
            }
            if (logType == "cerr") {
                return new TFilteredLogBackend<TStreamLogBackend>(new TStreamLogBackend(&Cerr), priority);
            } else if (logType == "null" || !logType || logType == "/dev/null") {
                return new TFilteredLogBackend<TNullLogBackend>(new TNullLogBackend(), priority);
            } else {
                return new TFilteredLogBackend<TFileLogBackend>(new TFileLogBackend(logType), priority);
            }
        } else {
            if (logType == "console") {
                return new TStreamLogBackend(&Cerr);
            }
            if (logType == "cout") {
                return new TStreamLogBackend(&Cout);
            }
            if (logType == "cerr") {
                return new TStreamLogBackend(&Cerr);
            } else if (logType == "null" || !logType || logType == "/dev/null") {
                return new TNullLogBackend;
            } else {
                return new TFileLogBackend(logType);
            }
        }
    } catch (...) {
        Cdbg << "Warning: " << logType << ": " << CurrentExceptionMessage() << ". Use stderr instead." << Endl;
    }

    return new TStreamLogBackend(&Cerr);
}

class TLog::TImpl: public TAtomicRefCount<TImpl> {
    class TPriorityLogStream: public TOutputStream {
    public:
        inline TPriorityLogStream(TLogPriority p, const TImpl* parent)
            : Priority_(p)
            , Parent_(parent)
        {
        }

        void DoWrite(const void* buf, size_t len) override {
            Parent_->WriteData(Priority_, (const char*)buf, len);
        }

    private:
        TLogPriority Priority_;
        const TImpl* Parent_;
    };

public:
    inline TImpl(TAutoPtr<TLogBackend> backend)
        : BackEnd_(backend)
        , DefaultPriority_(LOG_DEF_PRIORITY)
    {
    }

    inline ~TImpl() {
    }

    inline void ReopenLog() {
        if (!IsOpen()) {
            return;
        }

        BackEnd_->ReopenLog();
    }

    inline void AddLog(TLogPriority priority, const char* format, va_list args) const {
        if (!IsOpen()) {
            return;
        }

        TPriorityLogStream ls(priority, this);

        Printf(ls, format, args);
    }

    inline void ResetBackend(TAutoPtr<TLogBackend> backend) noexcept {
        BackEnd_.Reset(backend.Release());
    }

    inline TAutoPtr<TLogBackend> ReleaseBackend() noexcept {
        return BackEnd_.Release();
    }

    inline bool IsNullLog() const noexcept {
        return !IsOpen() || (dynamic_cast<TNullLogBackend*>(BackEnd_.Get()) != nullptr);
    }

    inline bool IsOpen() const noexcept {
        return nullptr != BackEnd_.Get();
    }

    inline void CloseLog() noexcept {
        BackEnd_.Destroy();

        Y_ASSERT(!IsOpen());
    }

    inline void WriteData(TLogPriority priority, const char* data, size_t len) const {
        if (IsOpen()) {
            BackEnd_->WriteData(TLogRecord(priority, data, len));
        }
    }

    inline TLogPriority DefaultPriority() noexcept {
        return DefaultPriority_;
    }

    inline void SetDefaultPriority(TLogPriority priority) noexcept {
        DefaultPriority_ = priority;
    }

    inline TLogPriority FiltrationLevel() const noexcept {
        return BackEnd_->FiltrationLevel();
    }

private:
    THolder<TLogBackend> BackEnd_;
    TLogPriority DefaultPriority_;
};

TLog::TLog()
    : Impl_(new TImpl(nullptr))
{
}

TLog::TLog(const TString& fname, TLogPriority priority) {
    THolder<TLogBackend> backend(BackendFactory(fname, priority));

    Impl_ = new TImpl(backend);
}

TLog::TLog(TAutoPtr<TLogBackend> backend)
    : Impl_(new TImpl(backend))
{
}

TLog::~TLog() {
}

bool TLog::IsOpen() const noexcept {
    return Impl_->IsOpen();
}

void TLog::AddLog(const char* format, ...) const {
    va_list args;
    va_start(args, format);

    try {
        Impl_->AddLog(Impl_->DefaultPriority(), format, args);
    } catch (...) {
        va_end(args);

        throw;
    }

    va_end(args);
}

void TLog::AddLog(TLogPriority priority, const char* format, ...) const {
    va_list args;
    va_start(args, format);

    try {
        Impl_->AddLog(priority, format, args);
    } catch (...) {
        va_end(args);

        throw;
    }

    va_end(args);
}

void TLog::AddLogVAList(const char* format, va_list lst) {
    Impl_->AddLog(Impl_->DefaultPriority(), format, lst);
}

void TLog::ReopenLog() {
    Impl_->ReopenLog();
}

void TLog::CloseLog() {
    Impl_->CloseLog();
}

void TLog::SetDefaultPriority(TLogPriority priority) noexcept {
    Impl_->SetDefaultPriority(priority);
}

TLogPriority TLog::FiltrationLevel() const noexcept {
    return Impl_->FiltrationLevel();
}

TLogPriority TLog::DefaultPriority() const noexcept {
    return Impl_->DefaultPriority();
}

bool TLog::OpenLog(const char* path, TLogPriority lp) {
    if (path) {
        ResetBackend(BackendFactory(path, lp));
    } else {
        ResetBackend(new TStreamLogBackend(&Cerr));
    }

    return true;
}

void TLog::ResetBackend(TAutoPtr<TLogBackend> backend) noexcept {
    Impl_->ResetBackend(backend);
}

bool TLog::IsNullLog() const noexcept {
    return Impl_->IsNullLog();
}

TAutoPtr<TLogBackend> TLog::ReleaseBackend() noexcept {
    return Impl_->ReleaseBackend();
}

void TLog::Write(TLogPriority priority, const char* data, size_t len) const {
    if (Formatter) {
        auto formated = Formatter(priority, TStringBuf{data, len});
        Impl_->WriteData(priority, formated.data(), formated.length());
    } else {
        Impl_->WriteData(priority, data, len);
    }
}

void TLog::Write(const char* data, size_t len) const {
    Write(Impl_->DefaultPriority(), data, len);
}

void TLog::SetFormatter(TLogFormatter formatter) noexcept {
    Formatter = formatter;
}
