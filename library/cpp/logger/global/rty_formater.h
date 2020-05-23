#pragma once

#include "common.h"

namespace NMemInfo {
    struct TMemInfo;
}

namespace NLoggingImpl {
    class TLocalTimeS {
    public:
        TLocalTimeS(TInstant instant = TInstant::Now())
            : Instant(instant)
        {
        }

        TInstant GetInstant() const {
            return Instant;
        }

        operator TString() const;
        TString operator+(TStringBuf right) const;

    private:
        TInstant Instant;
    };

    IOutputStream& operator<<(IOutputStream& out, TLocalTimeS localTimeS);

    inline TLocalTimeS GetLocalTimeS() {
        return TLocalTimeS();
    }

    TString GetSystemResources();
    TString PrintSystemResources(const NMemInfo::TMemInfo& info);
}

struct TRTYMessageFormater {
    static bool CheckLoggingContext(TLog& logger, const TLogRecordContext& context);
    static TSimpleSharedPtr<TLogElement> StartRecord(TLog& logger, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier);
};

using TRTYLogPreprocessor = TLogRecordPreprocessor<TLogFilter, TRTYMessageFormater>;
