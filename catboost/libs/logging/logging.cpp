#include "logging.h"

#include <library/logger/filter.h>
#include <library/logger/global/rty_formater.h>
#include <library/logger/log.h>


#include <util/system/mem_info.h>
#include <util/stream/printf.h>
#include <util/datetime/base.h>


namespace NMatrixnetLoggingImpl {
    TStringBuf StripFileName(TStringBuf string) {
        return string.RNextTok(LOCSLASH_C);
    }
}

class TCustomFuncLogger : public TLogBackend {
public:
    TCustomFuncLogger(TCustomLoggingFunction func)
        : LoggerFunc(func)
    {
    }
    void WriteData(const TLogRecord& rec) override {
        LoggerFunc(rec.Data, rec.Len);
    }
    void ReopenLog() override {
    }

private:
    TCustomLoggingFunction LoggerFunc = nullptr;
};

void SetCustomLoggingFunction(TCustomLoggingFunction func) {
    TMatrixnetLogSettings::GetRef().Log.ResetBackend(new TCustomFuncLogger(func));
}

void RestoreOriginalLogger() {
    TMatrixnetLogSettings::GetRef().Log.RestoreDefaultBackend();
}


TCatboostLogEntry::TCatboostLogEntry(TCatboostLog* parent, const TSourceLocation& sourceLocation, TStringBuf customMessage, ELogPriority priority) : Parent(parent)
, SourceLocation(sourceLocation)
, CustomMessage(customMessage)
, Priority(priority)
{
    if (TMatrixnetLogSettings::GetRef().OutputExtendedInfo) {
        (*this) << CustomMessage << ": " << NLoggingImpl::GetLocalTimeS() << " " << NMatrixnetLoggingImpl::StripFileName(SourceLocation.File) << ":" << SourceLocation.Line;
        if (Priority > TLOG_RESOURCES && !ExitStarted()) {
            *this << NLoggingImpl::GetSystemResources();
        }
        (*this) << " ";
    }
}

void TCatboostLogEntry::DoFlush()
{
    if (IsNull()) {
        return;
    }
    Parent->Output(*this);
    Reset();
}


TCatboostLogEntry::~TCatboostLogEntry()
{
    try {
        Finish();
    }
    catch (...) {
    }
}

class TCatboostLog::TImpl : public TLog {
public:
    TImpl(TAutoPtr<TLogBackend> backend)
        : TLog(backend)
    {}
};

TCatboostLog::TCatboostLog()
    : ImplHolder(new TCatboostLog::TImpl(CreateLogBackend("cout")))
{}

TCatboostLog::~TCatboostLog() {
}

void TCatboostLog::Output(const TCatboostLogEntry& entry) {
    const size_t filled = entry.Filled();

    if (filled) {
        ImplHolder->Write(entry.Data(), entry.Filled());
    }
}

void TCatboostLog::ResetBackend(THolder<TLogBackend>&& backend) {
    ImplHolder->ResetBackend(backend);
}

void TCatboostLog::RestoreDefaultBackend() {
    ImplHolder->ResetBackend(CreateLogBackend("cout"));
}
