#include "process_uptime.h"
#include "systime.h"
#include "uptime.h"

#if defined(_linux_)
    #include <util/string/builder.h>
    #include <util/string/cast.h>
    #include <util/stream/file.h>
    #include <util/string/split.h>

    #include <unistd.h>
#elif defined(_win_)
    #include <processthreadsapi.h>
#endif

TDuration ProcessUptime() {
#if defined(_win_)
    FILETIME createProcessTime;
    FILETIME dummy1;
    FILETIME dummy2;
    FILETIME dummy3;
    if (GetProcessTimes(GetCurrentProcess(), &createProcessTime, &dummy1, &dummy2, &dummy3)) {
        timeval processCreationTimeval{.tv_sec = 0, .tv_usec = 0};
        FileTimeToTimeval(&createProcessTime, &processCreationTimeval);
        const TInstant processCreationInstant = TInstant::Seconds(processCreationTimeval.tv_sec) + TDuration::MicroSeconds(processCreationTimeval.tv_usec);
        return TInstant::Now() - processCreationInstant;
    }
    ythrow TSystemError() << "Failed to obtain process starttime";
#elif defined(_linux_)
    static const auto statPath = "/proc/self/stat";
    // /proc/<pid>/stat format: #21 (0-based) item - starttime %llu - The time the process started after system boot
    TUnbufferedFileInput statFile(statPath);
    auto statStr = statFile.ReadAll();
    const auto completeStatsSize = 20; // First two fields skipped to ignore variations of parentheses
    const TVector<TStringBuf> stats = StringSplitter(TStringBuf{statStr}.RAfter(')').After(' ')).Split(' ').Take(completeStatsSize);
    ui64 startTimeTicks = 0;
    Y_THROW_UNLESS(stats.size() == completeStatsSize, "Broken format of " << statPath);
    if (!TryFromString<ui64>(stats.back(), startTimeTicks)) {
        ythrow yexception() << "Failed to extract process starttime value from " << statPath;
    }
    long ticksPerSecond = sysconf(_SC_CLK_TCK);
    Y_THROW_UNLESS_EX(ticksPerSecond != -1, TSystemError() << "Failed to get _SC_CLK_TCK");
    Y_THROW_UNLESS(ticksPerSecond > 0, "Invalid value of the _SC_CLK_TCK variable: " << ticksPerSecond);
    const ui64 startTimeSeconds = startTimeTicks / ticksPerSecond;
    const ui64 fractionTicks = startTimeTicks % ticksPerSecond;
    const TDuration startTimeFractionSeconds = TDuration::MicroSeconds(1'000'000u * fractionTicks / ticksPerSecond);
    const TDuration startTime = TDuration::Seconds(startTimeSeconds) + startTimeFractionSeconds;
    return Uptime() - startTime;
#else
    ythrow yexception() << "unimplemented";
#endif
}
