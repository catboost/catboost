#if defined(__APPLE__) && defined(__MACH__)

    #include <mach/mach.h>

#endif

#ifdef _win_

    #include "winint.h"
    #include <psapi.h>

#else

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

size_t TRusage::GetCurrentRSS() {
/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L; /* Can't access? */
    return (size_t)info.resident_size;
#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE* fp = nullptr;
    if ((fp = fopen("/proc/self/statm", "r")) == nullptr) {
        return (size_t)0L; /* Can't open? */
    }
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t)0L; /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L; /* Unsupported. */
#endif
}

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

    MaxRss = pmc.PeakWorkingSetSize;
    MajorPageFaults = pmc.PageFaultCount;

#else
    struct rusage ru;
    int r = getrusage(RUSAGE_SELF, &ru);
    if (r < 0) {
        ythrow TSystemError() << "rusage failed";
    }

    #if defined(_darwin_)
    // see https://lists.apple.com/archives/darwin-kernel/2009/Mar/msg00005.html
    MaxRss = ru.ru_maxrss;
    #else
    MaxRss = ru.ru_maxrss * 1024LL;
    #endif
    MajorPageFaults = ru.ru_majflt;
    Utime = ru.ru_utime;
    Stime = ru.ru_stime;
#endif
}
