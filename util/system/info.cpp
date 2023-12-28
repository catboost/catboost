#include "info.h"

#include "error.h"
#include "fs.h"

#include <cmath>
#include <cstdlib>

#if defined(_linux_) || defined(_cygwin_)
    #include <fcntl.h>
    #include <sys/sysinfo.h>
#endif

#if defined(_win_)
    #include "winint.h"
    #include <stdio.h>
#else
    #include <unistd.h>
#endif

#if defined(_bionic_)
//TODO
#elif defined(_cygwin_)
static int getloadavg(double* loadavg, int nelem) {
    for (int i = 0; i < nelem; ++i) {
        loadavg[i] = 0.0;
    }

    return nelem;
}
#elif defined(_unix_) || defined(_darwin_)
    #include <sys/types.h>
#endif

#if defined(_freebsd_) || defined(_darwin_)
    #include <sys/sysctl.h>
#endif

#include <util/string/ascii.h>
#include <util/string/cast.h>
#include <util/string/strip.h>
#include <util/string/split.h>
#include <util/stream/file.h>
#include <util/generic/yexception.h>

#if defined(_linux_)
/*
This function olny works properly if you apply correct setting to your nanny/deploy project

In nanny - Runtime -> Instance spec -> Advanced settings -> Cgroupfs settings: Mount mode = Read only

In deploy - Stage - Edit stage - Box - Cgroupfs settings: Mount mode = Read only
*/
static inline double CgroupV1Cpus(const TString& cpuCfsQuotaUsPath, const TString& cfsPeriodUsPath) {
    try {
        double q = FromString<int32_t>(StripString(TFileInput(cpuCfsQuotaUsPath).ReadAll()));

        if (q <= 0) {
            return 0;
        }

        double p = FromString<int32_t>(StripString(TFileInput(cfsPeriodUsPath).ReadAll()));

        if (p <= 0) {
            return 0;
        }

        return q / p;
    } catch (...) {
        return 0;
    }
}

/*
In cgroups v2 there isn't a dedicated "cpu" directory under /sys/fs/cgroup,
so the approximation of the number of CPUs may use the cpu.max file.
The format is the following:

$MAX $PERIOD
Which indicates that the group may consume up to $MAX in each $PERIOD duration.

The "max" value could be either the string "max" or a number. In the first case
our approximation doesn't work so we can bail out earlier.
*/
static inline double CgroupV2Cpus(const TString& cpuMaxPath) {
    try {
        TVector<TString> cgroupCpuMax = StringSplitter(TFileInput(cpuMaxPath).ReadAll()).Split(' ').Take(2);
        double max = FromString<int32_t>(StripString(cgroupCpuMax[0]));
        double period = FromString<int32_t>(StripString(cgroupCpuMax[1]));

        if (max <= 0 || period <= 0) {
            return 0;
        }

        return max / period;
    } catch (...) {
        return 0;
    }
}

static inline double CgroupCpus() {
    static const TString cpuMaxPath("/sys/fs/cgroup/cpu.max");
    static const TString cpuCfsQuotaUsPath("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
    static const TString cfsPeriodUsPath("/sys/fs/cgroup/cpu/cpu.cfs_period_us");

    if (NFs::Exists(cpuMaxPath)) {
        auto cgroup2Cpus = CgroupV2Cpus(cpuMaxPath);
        if (cgroup2Cpus > 0) {
            return cgroup2Cpus;
        }
    }

    if (NFs::Exists(cpuCfsQuotaUsPath) && NFs::Exists(cfsPeriodUsPath)) {
        auto cgroups1Cpus = CgroupV1Cpus(cpuCfsQuotaUsPath, cfsPeriodUsPath);
        if (cgroups1Cpus > 0) {
            return cgroups1Cpus;
        }
    }
    return 0;
}
#endif

size_t NSystemInfo::NumberOfMillicores() {
#if defined(_linux_)
    return CgroupCpus() * 1000;
#else
    // fallback behaviour if cgroupfs is not available
    // returns number of millicores which is a multiple of an integer number of cpus
    return NSystemInfo::NumberOfCpus() * 1000;
#endif
}

size_t NSystemInfo::NumberOfCpus() {
#if defined(_linux_)
    if (auto res = CgroupCpus(); res) {
        return Max<ssize_t>(1, std::llround(res));
    }
#endif

#if defined(_win_)
    SYSTEM_INFO info;

    GetSystemInfo(&info);

    return info.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    return sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(_linux_)
    unsigned ret;
    int fd, nread, column;
    char buf[512];
    static const char matchstr[] = "processor\t:";

    fd = open("/proc/cpuinfo", O_RDONLY);

    if (fd == -1) {
        abort();
    }

    column = 0;
    ret = 0;

    while (true) {
        nread = read(fd, buf, sizeof(buf));

        if (nread <= 0) {
            break;
        }

        for (int i = 0; i < nread; ++i) {
            const char ch = buf[i];

            if (ch == '\n') {
                column = 0;
            } else if (column != -1) {
                if (AsciiToLower(ch) == matchstr[column]) {
                    ++column;

                    if (column == sizeof(matchstr) - 1) {
                        column = -1;
                        ++ret;
                    }
                } else {
                    column = -1;
                }
            }
        }
    }

    if (ret == 0) {
        abort();
    }

    close(fd);

    return ret;
#elif defined(_freebsd_) || defined(_darwin_)
    int mib[2];
    size_t len;
    unsigned ncpus = 1;

    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(ncpus);
    if (sysctl(mib, 2, &ncpus, &len, nullptr, 0) == -1) {
        abort();
    }

    return ncpus;
#else
    #error todo
#endif
}

size_t NSystemInfo::LoadAverage(double* la, size_t len) {
#if defined(_win_) || defined(_musl_) || defined(_bionic_)
    int ret = -1;
#else
    for (size_t i = 0; i < len; ++i) {
        la[i] = 0;
    }

    int ret = getloadavg(la, len);
#endif

    if (ret < 0) {
        for (size_t i = 0; i < len; ++i) {
            la[i] = 0;
        }

        ret = len;
    }

    return (size_t)ret;
}

static size_t NCpus;
static size_t NMillicores;

size_t NSystemInfo::CachedNumberOfCpus() {
    if (!NCpus) {
        NCpus = NumberOfCpus();
    }

    return NCpus;
}

size_t NSystemInfo::CachedNumberOfMillicores() {
    if (!NMillicores) {
        NMillicores = NumberOfMillicores();
    }

    return NMillicores;
}

size_t NSystemInfo::GetPageSize() noexcept {
#if defined(_win_)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    return sysInfo.dwPageSize;
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

size_t NSystemInfo::TotalMemorySize() {
#if defined(_linux_) && defined(_64_)
    try {
        auto q = FromString<size_t>(StripString(TFileInput("/sys/fs/cgroup/memory/memory.limit_in_bytes").ReadAll()));

        if (q < (((size_t)1) << 60)) {
            return q;
        }
    } catch (...) {
    }
#endif

#if defined(_linux_) || defined(_cygwin_)
    struct sysinfo info;
    sysinfo(&info);
    return info.totalram;
#elif defined(_darwin_)
    int mib[2];
    int64_t memSize;
    size_t length;

    // Get the Physical memory size
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    length = sizeof(int64_t);
    if (sysctl(mib, 2, &memSize, &length, NULL, 0) != 0) {
        ythrow yexception() << "sysctl failed: " << LastSystemErrorText();
    }
    return (size_t)memSize;
#elif defined(_win_)
    MEMORYSTATUSEX memoryStatusEx;
    memoryStatusEx.dwLength = sizeof(memoryStatusEx);
    if (!GlobalMemoryStatusEx(&memoryStatusEx)) {
        ythrow yexception() << "GlobalMemoryStatusEx failed: " << LastSystemErrorText();
    }
    return (size_t)memoryStatusEx.ullTotalPhys;
#else
    return 0;
#endif
}

size_t NSystemInfo::MaxOpenFiles() {
#if defined(ANDROID) || defined(__ANDROID__)
    return sysconf(_SC_OPEN_MAX);
#elif defined(_win_)
    return _getmaxstdio();
#else
    return getdtablesize();
#endif
}
