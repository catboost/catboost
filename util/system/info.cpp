#include "info.h"

#include "error.h"

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

#include <util/generic/yexception.h>
#include <util/string/ascii.h>


size_t NSystemInfo::NumberOfCpus() {
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

size_t NSystemInfo::CachedNumberOfCpus() {
    if (!NCpus) {
        NCpus = NumberOfCpus();
    }

    return NCpus;
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
#if defined(_linux_) || defined(_cygwin_)
    struct sysinfo info;
    sysinfo(&info);
    return info.totalram;
#elif defined (_darwin_)
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
#elif defined (_win_)
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
