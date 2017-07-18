#include "rty_formater.h"
#include <util/datetime/base.h>
#include <util/stream/str.h>
#include <util/stream/printf.h>
#include <util/system/mem_info.h>
#include <inttypes.h>

namespace NLoggingImpl {
    TString GetLocalTimeS() {
        const TInstant now = Now();
        struct tm tm;
        TString time(Strftime("%Y-%m-%d %H:%M:%S.", now.LocalTime(&tm)));
        TStringOutput stream(time);
        Printf(stream, "%03" PRIu32, now.MilliSecondsOfSecond());
        stream << Strftime(" %z", &tm);
        return time;
    }

    TStringBuf StripFileName(TStringBuf string) {
        return string.RNextTok(LOCSLASH_C);
    }

    TString GetSystemResources() {
        NMemInfo::TMemInfo mi = NMemInfo::GetMemInfo();
        return PrintSystemResources(mi);
    }

    TString PrintSystemResources(const NMemInfo::TMemInfo& mi) {
        return Sprintf(" rss=%0.3fMb, vms=%0.3fMb", mi.RSS * 1.0 / (1024 * 1024), mi.VMS * 1.0 / (1024 * 1024));
    }
}

bool TRTYMessageFormater::CheckLoggingContext(TLog& /*logger*/, const TLogRecordContext& /*context*/) {
    return true;
}

TSimpleSharedPtr<TLogElement> TRTYMessageFormater::StartRecord(TLog& logger, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier) {
    if (!earlier)
        earlier.Reset(new TLogElement(&logger));
    (*earlier) << context.CustomMessage << ": " << NLoggingImpl::GetLocalTimeS() << " " << NLoggingImpl::StripFileName(context.SourceLocation.File) << ":" << context.SourceLocation.Line;
    if (context.Priority > TLOG_RESOURCES && !ExitStarted())
        (*earlier) << NLoggingImpl::GetSystemResources();
    (*earlier) << " ";
    return earlier;
}
