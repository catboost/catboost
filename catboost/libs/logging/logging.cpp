#include "logging.h"

#include <library/logger/filter.h>

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
    auto logger = TLoggerOperator<TGlobalLog>::Get();
    logger->ResetBackend(new TCustomFuncLogger(func));
}

void RestoreOriginalLogger() {
    TLoggerOperator<TGlobalLog>::Set(CreateDefaultLogger<TGlobalLog>());
}

bool TMatrixnetMessageFormater::CheckLoggingContext(TLog&, const TLogRecordContext& context) {
    return context.Priority <= TMatrixnetLogSettings::GetRef().LogPriority;
}

TSimpleSharedPtr<TLogElement> TMatrixnetMessageFormater::StartRecord(TLog& logger, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier) {
    if (!earlier) {
        earlier.Reset(new TLogElement(&logger));
    }
    if (TMatrixnetLogSettings::GetRef().OutputExtendedInfo) {
        (*earlier) << context.CustomMessage << ": " << NLoggingImpl::GetLocalTimeS() << " " << NMatrixnetLoggingImpl::StripFileName(context.SourceLocation.File) << ":" << context.SourceLocation.Line;
        if (context.Priority > TLOG_RESOURCES && !ExitStarted()) {
            (*earlier) << NLoggingImpl::GetSystemResources();
        }
        (*earlier) << " ";
    }
    return earlier;
}
