#pragma once

#include "defaults.h"
#include <util/generic/utility.h>
#include <util/datetime/base.h>

/// portable getrusage

struct TRusage {
    // some fields may be zero if unsupported

    ui64 MaxRss = 0;
    ui64 MajorPageFaults = 0;
    TDuration Utime;
    TDuration Stime;

    void Fill();

    static size_t GetCurrentRSS();

    static TRusage Get() {
        TRusage r;
        r.Fill();
        return r;
    }
};
