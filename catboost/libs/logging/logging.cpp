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

void SetCustomLoggingFunction(TCustomLoggingFunction lowPriorityFunc, TCustomLoggingFunction highPriorityFunc) {
    TMatrixnetLogSettings::GetRef().Log.ResetBackend(new TCustomFuncLogger(lowPriorityFunc), new TCustomFuncLogger(highPriorityFunc));
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
    TImpl(TAutoPtr<TLogBackend> lowPriorityBackend, TAutoPtr<TLogBackend> highPriorityBackend)
        : LowPriorityLog(lowPriorityBackend)
        , HighPriorityLog(highPriorityBackend)
    {}
    void ResetBackend(THolder<TLogBackend>&& lowPriorityBackend, THolder<TLogBackend>&& highPriorityBackend) {
        LowPriorityLog.ResetBackend(lowPriorityBackend);
        HighPriorityLog.ResetBackend(highPriorityBackend);
    }
    void Write(const TCatboostLogEntry& entry) {
        if (entry.Priority <= TLOG_WARNING) {
            HighPriorityLog.Write(entry.Data(), entry.Filled());
        } else {
            LowPriorityLog.Write(entry.Data(), entry.Filled());
        }
    }
private:
    TLog LowPriorityLog;
    TLog HighPriorityLog;
};

TCatboostLog::TCatboostLog()
    : ImplHolder(new TCatboostLog::TImpl(CreateLogBackend("cout"), CreateLogBackend("cerr")))
{}

TCatboostLog::~TCatboostLog() {
}

void TCatboostLog::Output(const TCatboostLogEntry& entry) {
    const size_t filled = entry.Filled();

    if (filled) {
        ImplHolder->Write(entry);
    }
}

void TCatboostLog::ResetBackend(THolder<TLogBackend>&& lowPriorityBackend, THolder<TLogBackend>&& highPriorityBackend) {
    ImplHolder->ResetBackend(std::move(lowPriorityBackend), std::move(highPriorityBackend));
}

void TCatboostLog::RestoreDefaultBackend() {
    ImplHolder->ResetBackend(CreateLogBackend("cout"), CreateLogBackend("cerr"));
}
