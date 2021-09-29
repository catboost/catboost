#include "logging.h"

#include <library/cpp/logger/filter.h>
#include <library/cpp/logger/global/rty_formater.h>
#include <library/cpp/logger/log.h>


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
    TCustomFuncLogger(TCustomLoggingFunction func, TCustomLoggingObject obj)
        : LoggerFunc(func), LoggingObject(obj)
    {
    }

    void WriteData(const TLogRecord& rec) override {
        LoggerFunc(rec.Data, rec.Len, LoggingObject);
    }
    void ReopenLog() override {
    }

private:
    TCustomLoggingFunction LoggerFunc = nullptr;
    TCustomLoggingObject LoggingObject = nullptr;
};

void SetCustomLoggingFunction(
    TCustomLoggingFunction normalPriorityFunc,
    TCustomLoggingFunction errorPriorityFunc,
    TCustomLoggingObject normalPriorityObj,
    TCustomLoggingObject errorPriorityObj)
{
    TCatBoostLogSettings::GetRef().Log.ResetBackend(
            MakeHolder<TCustomFuncLogger>(normalPriorityFunc, normalPriorityObj),
            MakeHolder<TCustomFuncLogger>(errorPriorityFunc, errorPriorityObj));
}

void RestoreOriginalLogger() {
    TCatBoostLogSettings::GetRef().Log.RestoreDefaultBackend();
}


TCatboostLogEntry::TCatboostLogEntry(TCatboostLog* parent, const TSourceLocation& sourceLocation, TStringBuf customMessage, ELogPriority priority) : Parent(parent)
, SourceLocation(sourceLocation)
, CustomMessage(customMessage)
, Priority(priority)
{
    if (parent->NeedExtendedInfo()) {
        (*this) << CustomMessage << ": " << NLoggingImpl::GetLocalTimeS() << " " << NMatrixnetLoggingImpl::StripFileName(SourceLocation.File) << ":" << SourceLocation.Line << " ";
        RegularMessageStartOffset = this->Filled();
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
        LowPriorityLog.ResetBackend(std::move(lowPriorityBackend));
        HighPriorityLog.ResetBackend(std::move(highPriorityBackend));
    }
    void ResetTraceBackend(THolder<TLogBackend>&& traceBackend) {
        TraceLog.ResetBackend(std::move(traceBackend));
    }
    void WriteRegularLog(const TCatboostLogEntry& entry, bool outputExtendedInfo) {
        const size_t regularOffset = outputExtendedInfo ? 0 : entry.GetRegularMessageStartOffset();
        if (entry.Priority <= TLOG_WARNING) {
            HighPriorityLog.Write(entry.Data() + regularOffset, entry.Filled() - regularOffset);
        } else {
            LowPriorityLog.Write(entry.Data() + regularOffset, entry.Filled() - regularOffset);
        }
    }
    void WriteTraceLog(const TCatboostLogEntry& entry) {
        TraceLog.Write(entry.Data(), entry.Filled());
    }
private:
    TLog LowPriorityLog;
    TLog HighPriorityLog;
    TLog TraceLog;
};

TCatboostLog::TCatboostLog()
    : ImplHolder(new TCatboostLog::TImpl(CreateLogBackend("cout"), CreateLogBackend("cerr")))
    , IsCustomBackendSpecified(false)
{}

TCatboostLog::~TCatboostLog() {
}

void TCatboostLog::Output(const TCatboostLogEntry& entry) {
    if (entry.Filled() != 0) {
        if (LogPriority >= entry.Priority) {
            ImplHolder->WriteRegularLog(entry, OutputExtendedInfo);
        }
        if (HaveTraceLog) {
            ImplHolder->WriteTraceLog(entry);
        }
    }
}

void TCatboostLog::ResetBackend(THolder<TLogBackend>&& lowPriorityBackend, THolder<TLogBackend>&& highPriorityBackend) {
    if (IsCustomBackendSpecified.exchange(true)) {
        CATBOOST_WARNING_LOG << "Custom logger is already specified. Specify more than one logger at same time is not thread safe.";
    }
    ImplHolder->ResetBackend(std::move(lowPriorityBackend), std::move(highPriorityBackend));
}

void TCatboostLog::ResetTraceBackend(THolder<TLogBackend>&& traceBackend /*= THolder<TLogBackend>()*/) {
    HaveTraceLog = (bool)traceBackend;
    ImplHolder->ResetTraceBackend(std::move(traceBackend));
}

void TCatboostLog::RestoreDefaultBackend() {
    ImplHolder->ResetBackend(CreateLogBackend("cout"), CreateLogBackend("cerr"));
    IsCustomBackendSpecified.store(false);
}

void ResetTraceBackend(const TString& name) {
    TCatBoostLogSettings::GetRef().Log.ResetTraceBackend(CreateLogBackend(name));
}
