#include <util/system/platform.h>

#include <library/python/symbols/registry/syms.h>

#if !defined(_MSC_VER)
#include <aio.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <pthread.h>
#include <pwd.h>
#include <sched.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/ipc.h>
#include <dlfcn.h>

#if defined(_linux_)
#include <sys/prctl.h>
#include <sys/sendfile.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>
#endif

#if defined(_darwin_)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if defined(_linux_)
#include <sys/inotify.h>
#endif

namespace {
    static inline void* ErrnoLocation() {
        return &errno;
    }

    static int ClockGetres(clockid_t clk_id, struct timespec* res) {
#if defined(_darwin_)
        using TFunc = decltype(&ClockGetres);
        static TFunc func = (TFunc)dlsym(RTLD_SELF, "_clock_getres");

        if (func) {
            return func(clk_id, res);
        }

        memset(res, 0, sizeof(*res));

        res->tv_sec = 0;
        res->tv_nsec = 50000000;

        return 0;
#else
        return clock_getres(clk_id, res);
#endif
    }
}

BEGIN_SYMS("c")
#if defined(_linux_)
SYM(prctl)
SYM(sched_getaffinity)
SYM(sched_setaffinity)
SYM(inotify_init)
SYM(inotify_add_watch)
SYM(inotify_rm_watch)
#endif
#if defined(_darwin_)
SYM(sysctlbyname)
#endif
SYM(ftok)
SYM(aio_error)
SYM(aio_read)
SYM(aio_return)
SYM(aio_suspend)
SYM(calloc)
SYM(clock_gettime)
SYM_2("clock_getres", ClockGetres)
SYM(closedir)
SYM(freeifaddrs)
SYM(getifaddrs)
SYM(getnameinfo)
SYM(getpwnam)
SYM(inet_ntop)
SYM(opendir)
SYM(pthread_kill)
SYM(pthread_self)
SYM(readdir_r)
SYM(siginterrupt)
SYM(strdup)
SYM(sendfile)
SYM(strtod)
SYM_2("__errno_location", ErrnoLocation)
END_SYMS()
#endif
