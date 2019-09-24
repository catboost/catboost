#include <util/system/platform.h>

#include <library/python/symbols/registry/syms.h>

#if !defined(_MSC_VER)
#if __has_include(<aio.h>)
#include <aio.h>
#endif
#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <pthread.h>
#include <pwd.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/ipc.h>
#include <dlfcn.h>

#if defined(_linux_)
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/sendfile.h>
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>
#endif

#if defined(_darwin_)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach_error.h> // Y_IGNORE
#include <mach/mach_time.h> // Y_IGNORE
#endif

#if defined(_linux_)
#include <sys/inotify.h>
#include <sys/mman.h>
#endif

namespace {
    static inline void* ErrnoLocation() {
        return &errno;
    }

    static int ClockGetres(clockid_t clk_id, struct timespec* res) {
#if defined(_darwin_)
        static auto func = (decltype(&ClockGetres))dlsym(RTLD_SELF, "_clock_getres");

        if (func) {
            return func(clk_id, res);
        }

        // https://opensource.apple.com/source/Libc/Libc-1158.1.2/gen/clock_gettime.c.auto.html

        switch (clk_id){
            case CLOCK_REALTIME:
            case CLOCK_MONOTONIC:
            case CLOCK_PROCESS_CPUTIME_ID:
                res->tv_nsec = NSEC_PER_USEC;
                res->tv_sec = 0;

                return 0;

            case CLOCK_MONOTONIC_RAW:
            case CLOCK_MONOTONIC_RAW_APPROX:
            case CLOCK_UPTIME_RAW:
            case CLOCK_UPTIME_RAW_APPROX:
            case CLOCK_THREAD_CPUTIME_ID: {
                mach_timebase_info_data_t tb_info;

                if (mach_timebase_info(&tb_info)) {
                    return -1;
                }

                res->tv_nsec = tb_info.numer / tb_info.denom + (tb_info.numer % tb_info.denom != 0);
                res->tv_sec = 0;

                return 0;
            }

            default:
                errno = EINVAL;
                return -1;
        }
#else
        return clock_getres(clk_id, res);
#endif
    }
}

BEGIN_SYMS("c")

SYM(calloc)
SYM(clock_gettime)
SYM_2("clock_getres", ClockGetres)
SYM(closedir)
SYM(fdopen)
SYM(fflush)
SYM(freeifaddrs)
SYM(ftok)
SYM(getifaddrs)
SYM(getnameinfo)
SYM(getpwnam)
SYM(inet_ntop)
SYM(opendir)
SYM(printf)
SYM(pthread_kill)
SYM(pthread_self)
SYM(readdir_r)
SYM(sem_close)
SYM(sem_getvalue)
SYM(sem_open)
SYM(sem_post)
SYM(sem_trywait)
SYM(sem_unlink)
SYM(sem_wait)
SYM(siginterrupt)
SYM(strdup)
SYM(sendfile)
SYM(strtod)
SYM_2("__errno_location", ErrnoLocation)

#if defined(_linux_)
SYM(prctl)
SYM(ptrace)
SYM(sched_getaffinity)
SYM(sched_setaffinity)
SYM(sem_timedwait)
SYM(inotify_init)
SYM(inotify_add_watch)
SYM(inotify_rm_watch)
SYM(mlockall)
#endif

#if defined(_darwin_)
SYM(mach_absolute_time)
SYM(mach_timebase_info)
SYM(sysctlbyname)
#endif

#if __has_include(<aio.h>)
SYM(aio_error)
SYM(aio_read)
SYM(aio_return)
SYM(aio_suspend)
#endif

END_SYMS()
#endif
