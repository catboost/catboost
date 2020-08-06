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
                return MakeHolder<TFilteredLogBackend<TStreamLogBackend>>(new TStreamLogBackend(&Cerr), priority);
            }
            if (logType == "cout") {
                return MakeHolder<TFilteredLogBackend<TStreamLogBackend>>(new TStreamLogBackend(&Cout), priority);
            }
            if (logType == "cerr") {
                return MakeHolder<TFilteredLogBackend<TStreamLogBackend>>(new TStreamLogBackend(&Cerr), priority);
            } else if (logType == "null" || !logType || logType == "/dev/null") {
                return MakeHolder<TFilteredLogBackend<TNullLogBackend>>(new TNullLogBackend(), priority);
            } else {
                return MakeHolder<TFilteredLogBackend<TFileLogBackend>>(new TFileLogBackend(logType), priority);
            }
        } else {
            if (logType == "console") {
                return MakeHolder<TStreamLogBackend>(&Cerr);
            }
            if (logType == "cout") {
                return MakeHolder<TStreamLogBackend>(&Cout);
            }
            if (logType == "cerr") {
                return MakeHolder<TStreamLogBackend>(&Cerr);
            } else if (logType == "null" || !logType || logType == "/dev/null") {
                return MakeHolder<TNullLogBackend>();
            } else {
                return MakeHolder<TFileLogBackend>(logType);
            }
        }
    } catch (...) {
        Cdbg << "Warning: " << logType << ": " << CurrentExceptionMessage() << ". Use stderr instead." << Endl;
    }

    if (priority != LOG_MAX_PRIORITY) {
        return MakeHolder<TFilteredLogBackend<TStreamLogBackend>>(new TStreamLogBackend(&Cerr), priority);
    }
    return MakeHolder<TStreamLogBackend>(&Cerr);
}

THolder<TLogBackend> CreateLogBackend(const TString& fname, ELogPriority priority, bool threaded) {
    if (!threaded) {
        return BackendFactory(fname, priority);
    }
    return CreateFilteredOwningThreadedLogBackend(fname, priority);
}

THolder<TLogBackend> CreateFilteredOwningThreadedLogBackend(const TString& fname, ELogPriority priority, size_t queueLen) {
    return MakeHolder<TFilteredLogBackend<TOwningThreadedLogBackend>>(CreateOwningThreadedLogBackend(fname, queueLen).Release(), priority);
}

THolder<TOwningThreadedLogBackend> CreateOwningThreadedLogBackend(const TString& fname, size_t queueLen) {
    return MakeHolder<TOwningThreadedLogBackend>(BackendFactory(fname, LOG_MAX_PRIORITY).Release(), queueLen);
}

class TLog::TImpl: public TAtomicRefCount<TImpl> {
    class TPriorityLogStream final: public IOutputStream {
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
        ELogPriority Priority_ = LOG_DEF_PRIORITY;
        const TImpl* Parent_ = nullptr;
    };

public:
    inline TImpl(THolder<TLogBackend> backend)
        : Backend_(std::move(backend))
    {
    }

    inline void ReopenLog() {
        if (!IsOpen()) {
            return;
        }

        Backend_->ReopenLog();
    }

    inline void ReopenLogNoFlush() {
        if (!IsOpen()) {
            return;
        }

        Backend_->ReopenLogNoFlush();
    }

    inline void AddLog(ELogPriority priority, const char* format, va_list args) const {
        if (!IsOpen()) {
            return;
        }

        TPriorityLogStream ls(priority, this);

        Printf(ls, format, args);
    }

    inline void ResetBackend(THolder<TLogBackend> backend) noexcept {
        Backend_ = std::move(backend);
    }

    inline THolder<TLogBackend> ReleaseBackend() noexcept {
        return std::move(Backend_);
    }

    inline bool IsNullLog() const noexcept {
        return !IsOpen() || (dynamic_cast<TNullLogBackend*>(Backend_.Get()) != nullptr);
    }

    inline bool IsOpen() const noexcept {
        return nullptr != Backend_.Get();
    }

    inline void CloseLog() noexcept {
        Backend_.Destroy();

        Y_ASSERT(!IsOpen());
    }

    inline void WriteData(ELogPriority priority, const char* data, size_t len) const {
        if (IsOpen()) {
            Backend_->WriteData(TLogRecord(priority, data, len));
        }
    }

    inline ELogPriority DefaultPriority() noexcept {
        return DefaultPriority_;
    }

    inline void SetDefaultPriority(ELogPriority priority) noexcept {
        DefaultPriority_ = priority;
    }

    inline ELogPriority FiltrationLevel() const noexcept {
        return Backend_->FiltrationLevel();
    }

    inline size_t BackEndQueueSize() const {
        return Backend_->QueueSize();
    }

private:
    THolder<TLogBackend> Backend_;
    ELogPriority DefaultPriority_ = LOG_DEF_PRIORITY;
};

TLog::TLog()
    : Impl_(MakeIntrusive<TImpl>(nullptr))
{
}

TLog::TLog(const TString& fname, ELogPriority priority)
    : TLog(BackendFactory(fname, priority))
{
}

TLog::TLog(THolder<TLogBackend> backend)
    : Impl_(MakeIntrusive<TImpl>(std::move(backend)))
{
}

TLog::TLog(const TLog&) = default;
TLog::TLog(TLog&&) = default;
TLog::~TLog() = default;
TLog& TLog::operator=(const TLog&) = default;
TLog& TLog::operator=(TLog&&) = default;

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
    if (const auto copy = Impl_) {
        copy->ReopenLog();
    }
}

void TLog::ReopenLogNoFlush() {
    if (const auto copy = Impl_) {
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
        ResetBackend(MakeHolder<TStreamLogBackend>(&Cerr));
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
    if (Formatter_) {
        const auto formated = Formatter_(priority, TStringBuf{data, len});
        Impl_->WriteData(priority, formated.data(), formated.size());
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
    Formatter_ = std::move(formatter);
}

size_t TLog::BackEndQueueSize() const {
    return Impl_->BackEndQueueSize();
}
