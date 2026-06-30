#include "rty_formater.h"
#include <util/datetime/base.h>
#include <util/datetime/systime.h>
#include <util/stream/str.h>
#include <util/stream/printf.h>
#include <util/system/mem_info.h>
#include <util/system/yassert.h>
#include <inttypes.h>
#include <cstdio>

namespace {
    constexpr size_t LocalTimeSBufferSize = sizeof("2017-07-24 12:20:34.313 +0300");

    size_t PrintLocalTimeS(const TInstant instant, char* const begin, const char* const end) {
        Y_ABORT_UNLESS(static_cast<size_t>(end - begin) >= LocalTimeSBufferSize);

        struct tm tm;
        instant.LocalTime(&tm);

        // both stftime and snprintf exclude the terminating null byte from the return value
        char* pos = begin;
        pos += strftime(pos, end - pos, "%Y-%m-%d %H:%M:%S.", &tm);
        pos += snprintf(pos, end - pos, "%03" PRIu32, instant.MilliSecondsOfSecond());
        pos += strftime(pos, end - pos, " %z", &tm);
        Y_ABORT_UNLESS(LocalTimeSBufferSize - 1 == pos - begin); // together with Y_ABORT_UNLESS above this also implies pos<=end
        return (pos - begin);
    }
}

namespace NLoggingImpl {
    IOutputStream& operator<<(IOutputStream& out, TLocalTimeS localTimeS) {
        char buffer[LocalTimeSBufferSize];
        size_t len = PrintLocalTimeS(localTimeS.GetInstant(), buffer, buffer + sizeof(buffer));
        out.Write(buffer, len);
        return out;
    }

    TLocalTimeS::operator TString() const {
        TString res;
        res.reserve(LocalTimeSBufferSize);
        res.ReserveAndResize(PrintLocalTimeS(Instant, res.begin(), res.begin() + res.capacity()));
        return res;
    }

    TString TLocalTimeS::operator+(const TStringBuf right) const {
        TString res(*this);
        res += right;
        return res;
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

namespace {
    class TRtyLoggerFormatter : public ILoggerFormatter {
    public:
        void Format(const TLogRecordContext& context, TLogElement& elem) const override {
            elem << context.CustomMessage << ": " << NLoggingImpl::GetLocalTimeS() << " "
                 << NLoggingImpl::StripFileName(context.SourceLocation.File) << ":" << context.SourceLocation.Line;
            if (context.Priority > TLOG_RESOURCES && !ExitStarted()) {
                elem << NLoggingImpl::GetSystemResources();
            }
            elem << " ";
        }
    };
}

ILoggerFormatter* CreateRtyLoggerFormatter() {
    return new TRtyLoggerFormatter();
}

bool TRTYMessageFormater::CheckLoggingContext(TLog& /*logger*/, const TLogRecordContext& /*context*/) {
    return true;
}

TSimpleSharedPtr<TLogElement> TRTYMessageFormater::StartRecord(TLog& logger, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier) {
    if (!earlier) {
        earlier.Reset(new TLogElement(&logger));
    }

    TLoggerFormatterOperator::Get()->Format(context, *earlier);
    return earlier;
}
