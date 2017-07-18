#pragma once

#include "common.h"

namespace NMemInfo {
    struct TMemInfo;
}

namespace NLoggingImpl {
    TString GetLocalTimeS();
    TString GetSystemResources();
    TString PrintSystemResources(const NMemInfo::TMemInfo& info);
}

struct TRTYMessageFormater {
    static bool CheckLoggingContext(TLog& logger, const TLogRecordContext& context);
    static TSimpleSharedPtr<TLogElement> StartRecord(TLog& logger, const TLogRecordContext& context, TSimpleSharedPtr<TLogElement> earlier);
};

using TRTYLogPreprocessor = TLogRecordPreprocessor<TLogFilter, TRTYMessageFormater>;
