#pragma once

#include "defaults.h"
#include <util/generic/utility.h>
#include <util/datetime/base.h>

/// portable getrusage

struct TRusage {
    // some fields may be zero if unsupported

    // RSS in bytes
    // returned value may be not accurate, see discussion
    // http://www.mail-archive.com/freebsd-stable@freebsd.org/msg77102.html
    ui64 Rss = 0;
    ui64 MajorPageFaults = 0;
    TDuration Utime;
    TDuration Stime;

    void Fill();

    static TRusage Get() {
        TRusage r;
        r.Fill();
        return r;
    }
};
