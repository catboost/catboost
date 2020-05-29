#include "file.h"
#include "filter.h"
#include "log.h"
#include "null.h"
#include "stream.h"
#include "thread.h"

#include <util/string/cast.h>
#include <util/stream/printf.h>
#include <util/system/yassert.h>
#include <util/generic/string.h>
#include <util/generic/scope.h>
#include <util/generic/yexception.h>

static inline THolder<TLogBackend> BackendFactory(const TString& logType, ELogPriority priority) {
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

    if (priority != LOG_MAX_PRIORITY) {
        return new TFilteredLogBackend<TStreamLogBackend>(new TStreamLogBackend(&Cerr), priority);
    }
    return new TStreamLogBackend(&Cerr);
}

THolder<TLogBackend> CreateLogBackend(const TString& fname, ELogPriority priority, bool threaded) {
    if (!threaded) {
        return BackendFactory(fname, priority);
    }
    return CreateFilteredOwningThreadedLogBackend(fname, priority);
}

THolder<TLogBackend> CreateFilteredOwningThreadedLogBackend(const TString& fname, ELogPriority priority, size_t queueLen) {
    return new TFilteredLogBackend<TOwningThreadedLogBackend>(CreateOwningThreadedLogBackend(fname, queueLen).Release(), priority);
}

THolder<TOwningThreadedLogBackend> CreateOwningThreadedLogBackend(const TString& fname, size_t queueLen) {
    return new TOwningThreadedLogBackend(BackendFactory(fname, LOG_MAX_PRIORITY).Release(), queueLen);
}

class TLog::TImpl: public TAtomicRefCount<TImpl> {
    class TPriorityLogStream: public IOutputStream {
    public:
        inline TPriorityLogStream(ELogPriority p, const TImpl* parent)
            : Priority_(p)
            , Parent_(parent)
        {
        }

        void DoWrite(const void* buf, size_t len) override {
            Parent_->WriteData(Priority_, (const char*)buf, len);
        }

    private:
        ELogPriority Priority_;
        const TImpl* Parent_;
    };

public:
    inline TImpl(THolder<TLogBackend> backend)
        : BackEnd_(std::move(backend))
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

    inline void ReopenLogNoFlush() {
        if (!IsOpen()) {
            return;
        }

        BackEnd_->ReopenLogNoFlush();
    }

    inline void AddLog(ELogPriority priority, const char* format, va_list args) const {
        if (!IsOpen()) {
            return;
        }

        TPriorityLogStream ls(priority, this);

        Printf(ls, format, args);
    }

    inline void ResetBackend(THolder<TLogBackend> backend) noexcept {
        BackEnd_.Reset(backend.Release());
    }

    inline THolder<TLogBackend> ReleaseBackend() noexcept {
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

    inline void WriteData(ELogPriority priority, const char* data, size_t len) const {
        if (IsOpen()) {
            BackEnd_->WriteData(TLogRecord(priority, data, len));
        }
    }

    inline ELogPriority DefaultPriority() noexcept {
        return DefaultPriority_;
    }

    inline void SetDefaultPriority(ELogPriority priority) noexcept {
        DefaultPriority_ = priority;
    }

    inline ELogPriority FiltrationLevel() const noexcept {
        return BackEnd_->FiltrationLevel();
    }

    inline size_t BackEndQueueSize() const {
        return BackEnd_->QueueSize();
    }

private:
    THolder<TLogBackend> BackEnd_;
    ELogPriority DefaultPriority_;
};

TLog::TLog()
    : Impl_(new TImpl(nullptr))
{
}

TLog::TLog(const TString& fname, ELogPriority priority) {
    THolder<TLogBackend> backend(BackendFactory(fname, priority));

    Impl_ = new TImpl(std::move(backend));
}

TLog::TLog(THolder<TLogBackend> backend)
    : Impl_(new TImpl(std::move(backend)))
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

    Y_DEFER {
        va_end(args);
    };

    Impl_->AddLog(Impl_->DefaultPriority(), format, args);
}

void TLog::AddLog(ELogPriority priority, const char* format, ...) const {
    va_list args;
    va_start(args, format);

    Y_DEFER {
        va_end(args);
    };

    Impl_->AddLog(priority, format, args);
}

void TLog::AddLogVAList(const char* format, va_list lst) {
    Impl_->AddLog(Impl_->DefaultPriority(), format, lst);
}

void TLog::ReopenLog() {
    TSimpleIntrusivePtr<TImpl> copy = Impl_;
    if (copy) {
        copy->ReopenLog();
    }
}

void TLog::ReopenLogNoFlush() {
    TSimpleIntrusivePtr<TImpl> copy = Impl_;
    if (copy) {
        copy->ReopenLogNoFlush();
    }
}

void TLog::CloseLog() {
    Impl_->CloseLog();
}

void TLog::SetDefaultPriority(ELogPriority priority) noexcept {
    Impl_->SetDefaultPriority(priority);
}

ELogPriority TLog::FiltrationLevel() const noexcept {
    return Impl_->FiltrationLevel();
}

ELogPriority TLog::DefaultPriority() const noexcept {
    return Impl_->DefaultPriority();
}

bool TLog::OpenLog(const char* path, ELogPriority lp) {
    if (path) {
        ResetBackend(BackendFactory(path, lp));
    } else {
        ResetBackend(new TStreamLogBackend(&Cerr));
    }

    return true;
}

void TLog::ResetBackend(THolder<TLogBackend> backend) noexcept {
    Impl_->ResetBackend(std::move(backend));
}

bool TLog::IsNullLog() const noexcept {
    return Impl_->IsNullLog();
}

THolder<TLogBackend> TLog::ReleaseBackend() noexcept {
    return Impl_->ReleaseBackend();
}

void TLog::Write(ELogPriority priority, const char* data, size_t len) const {
    if (Formatter) {
        auto formated = Formatter(priority, TStringBuf{data, len});
        Impl_->WriteData(priority, formated.data(), formated.length());
    } else {
        Impl_->WriteData(priority, data, len);
    }
}

void TLog::Write(ELogPriority priority, const TStringBuf data) const {
    Write(priority, data.data(), data.size());
}

void TLog::Write(const char* data, size_t len) const {
    Write(Impl_->DefaultPriority(), data, len);
}

void TLog::SetFormatter(TLogFormatter formatter) noexcept {
    Formatter = formatter;
}

size_t TLog::BackEndQueueSize() const {
    return Impl_->BackEndQueueSize();
}
