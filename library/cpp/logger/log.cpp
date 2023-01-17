#include "log.h"
#include "uninitialized_creator.h"
#include "filter.h"
#include "null.h"
#include "stream.h"
#include "thread.h"

#include <util/stream/printf.h>
#include <util/system/yassert.h>
#include <util/generic/scope.h>

THolder<TLogBackend> CreateLogBackend(const TString& fname, ELogPriority priority, bool threaded) {
    TLogBackendCreatorUninitialized creator;
    creator.InitCustom(fname, priority, threaded);
    return creator.CreateLogBackend();
}

THolder<TLogBackend> CreateFilteredOwningThreadedLogBackend(const TString& fname, ELogPriority priority, size_t queueLen) {
    return MakeHolder<TFilteredLogBackend>(CreateOwningThreadedLogBackend(fname, queueLen), priority);
}

THolder<TOwningThreadedLogBackend> CreateOwningThreadedLogBackend(const TString& fname, size_t queueLen) {
    return MakeHolder<TOwningThreadedLogBackend>(CreateLogBackend(fname, LOG_MAX_PRIORITY, false).Release(), queueLen);
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

    inline void WriteData(ELogPriority priority, const char* data, size_t len, TLogRecord::TMetaFlags metaFlags = {}) const {
        if (IsOpen()) {
            Backend_->WriteData(TLogRecord(priority, data, len, std::move(metaFlags)));
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
    : TLog(CreateLogBackend(fname, priority, false))
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
        ResetBackend(CreateLogBackend(path, lp));
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

void TLog::Write(ELogPriority priority, const char* data, size_t len, TLogRecord::TMetaFlags metaFlags) const {
    if (Formatter_) {
        const auto formated = Formatter_(priority, TStringBuf{data, len});
        Impl_->WriteData(priority, formated.data(), formated.size(), std::move(metaFlags));
    } else {
        Impl_->WriteData(priority, data, len, std::move(metaFlags));
    }
}

void TLog::Write(ELogPriority priority, const TStringBuf data, TLogRecord::TMetaFlags metaFlags) const {
    Write(priority, data.data(), data.size(), std::move(metaFlags));
}

void TLog::Write(const char* data, size_t len, TLogRecord::TMetaFlags metaFlags) const {
    Write(Impl_->DefaultPriority(), data, len, std::move(metaFlags));
}

void TLog::SetFormatter(TLogFormatter formatter) noexcept {
    Formatter_ = std::move(formatter);
}

size_t TLog::BackEndQueueSize() const {
    return Impl_->BackEndQueueSize();
}
