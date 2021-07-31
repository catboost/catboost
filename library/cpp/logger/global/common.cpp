#include "common.h"

#include <util/generic/yexception.h>

namespace NLoggingImpl {
    TString GetLocalTimeSSimple() {
        struct tm tm;
        return Strftime("%b%d_%H%M%S", Now().LocalTime(&tm));
    }

    TString PrepareToOpenLog(TString logType, const int logLevel, const bool rotation, const bool startAsDaemon) {
        Y_ENSURE(logLevel >= 0 && logLevel <= (int)LOG_MAX_PRIORITY, "Incorrect log level");

        if (rotation && TFsPath(logType).Exists()) {
            TString newPath = Sprintf("%s_%s_%" PRIu64, logType.data(), NLoggingImpl::GetLocalTimeSSimple().data(), static_cast<ui64>(Now().MicroSeconds()));
            TFsPath(logType).RenameTo(newPath);
        }
        if (startAsDaemon && (logType == "console"sv || logType == "cout"sv || logType == "cerr"sv)) {
            logType = "null";
        }

        return logType;
    }
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
