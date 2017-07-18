#include "platform.h"

#ifdef _win_

#include "winint.h"
#include <psapi.h>

#else

#include <sys/time.h>
#include <sys/resource.h>

#endif

#include <util/generic/yexception.h>

#include "info.h"

#include "rusage.h"

#ifdef _win_
TDuration FiletimeToDuration(const FILETIME& ft) {
    union {
        ui64 ft_scalar;
        FILETIME ft_struct;
    } nt_time;
    nt_time.ft_struct = ft;
    return TDuration::MicroSeconds(nt_time.ft_scalar / 10);
}
#endif

void TRusage::Fill() {
    *this = TRusage();

#ifdef _win_
    // copy-paste from PostgreSQL getrusage.c

    FILETIME starttime;
    FILETIME exittime;
    FILETIME kerneltime;
    FILETIME usertime;

    if (GetProcessTimes(GetCurrentProcess(), &starttime, &exittime, &kerneltime, &usertime) == 0) {
        ythrow TSystemError() << "GetProcessTimes failed";
    }

    Utime = FiletimeToDuration(usertime);
    Stime = FiletimeToDuration(kerneltime);

    PROCESS_MEMORY_COUNTERS pmc;

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)) == 0) {
        ythrow TSystemError() << "GetProcessMemoryInfo failed";
    }

    Rss = pmc.PeakWorkingSetSize;
    MajorPageFaults = pmc.PageFaultCount;

#else
    struct rusage ru;
    int r = getrusage(RUSAGE_SELF, &ru);
    if (r < 0) {
        ythrow TSystemError() << "rusage failed";
    }

    Rss = ru.ru_maxrss * 1024LL;
    MajorPageFaults = ru.ru_majflt;
    Utime = ru.ru_utime;
    Stime = ru.ru_stime;
#endif
}
