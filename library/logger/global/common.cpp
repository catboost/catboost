#include "common.h"

namespace NLoggingImpl {
    TString GetLocalTimeSSimple() {
        struct tm tm;
        return Strftime("%b%d_%H%M%S", Now().LocalTime(&tm));
    }
}

TLogRecordContext::TLogRecordContext(const TSourceLocation& sourceLocation, const char* customMessage, ELogPriority priority)
    : SourceLocation(sourceLocation)
    , CustomMessage(customMessage)
    , Priority(priority)
{
}

bool TLogFilter::CheckLoggingContext(TLog& log, const TLogRecordContext& context) {
    return context.Priority <= log.FiltrationLevel();
}

TSimpleSharedPtr<TLogElement> TLogFilter::StartRecord(TLog& logger, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier) {
    if (earlier)
        return earlier;
    TSimpleSharedPtr<TLogElement> result(new TLogElement(&logger));
    *result << context.Priority;
    return result;
}
