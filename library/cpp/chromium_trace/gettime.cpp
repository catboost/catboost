#include "gettime.h"

#include <util/generic/utility.h>
#include <util/system/datetime.h>

#ifdef _win_
#include <windows.h>
#endif

namespace NChromiumTrace {
    TInstant GetThreadCPUTime() {
#ifdef _linux_
        struct timespec ts;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
        struct timeval tv;
        tv.tv_sec = ts.tv_sec;
        tv.tv_usec = ts.tv_nsec / 1000;
        return TInstant(tv);
#else
        // TODO: add implementations for win32/darwin, move to util
        return TInstant();
#endif
    }

    TInstant GetWallTime() {
#ifdef _win_
        LARGE_INTEGER counter;
        LARGE_INTEGER frequency;
        if (QueryPerformanceCounter(&counter) && QueryPerformanceFrequency(&frequency)) {
            return TInstant::MicroSeconds(counter.QuadPart * 1'000'000 / frequency.QuadPart);
        }
        return {};
#else
        // With "complete" events, trace viewer relies only on timestamps to
        // restore sequential ordering of events.  In case of two events too
        // close in time, it fails to disambiguate between them.  It is
        // acceptable to adjust time a bit, just to get a unique timestamp in
        // each call.
        // Tracing is not pretty much useful at sub-microsecond scale anyway.
        static thread_local ui64 LastTimeStamp = 0;
        LastTimeStamp = Max(LastTimeStamp + 1, MicroSeconds());
        return TInstant::MicroSeconds(LastTimeStamp);
#endif        
    }
}
