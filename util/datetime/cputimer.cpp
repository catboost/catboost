#include "cputimer.h"

#include <util/system/defaults.h>
#include <util/system/hp_timer.h>
#include <util/string/printf.h>
#include <util/stream/output.h>
#include <util/generic/singleton.h>

#if defined(_unix_)
    #include <unistd.h>
    #include <sched.h>
#elif defined(_win_)
    #include <util/system/winint.h>
#endif

TTimer::TTimer(const TStringBuf message) {
    static const int SMALL_DURATION_CHAR_LENGTH = 9;                     // strlen("0.123456s")
    Message_.Reserve(message.length() + SMALL_DURATION_CHAR_LENGTH + 1); // +"\n"
    Message_ << message;
    // Do not measure the allocations above.
    Start_ = TInstant::Now();
}

TTimer::~TTimer() {
    const TDuration duration = TInstant::Now() - Start_;
    Message_ << duration << "\n";
    Cerr << Message_.Str();
}

static ui64 ManuallySetCyclesPerSecond = 0;

static ui64 GetCyclesPerSecond() {
    if (ManuallySetCyclesPerSecond != 0) {
        return ManuallySetCyclesPerSecond;
    } else {
        return NHPTimer::GetCyclesPerSecond();
    }
}

void SetCyclesPerSecond(ui64 cycles) {
    ManuallySetCyclesPerSecond = cycles;
}

ui64 GetCyclesPerMillisecond() {
    return GetCyclesPerSecond() / 1000;
}

TDuration CyclesToDuration(ui64 cycles) {
    return TDuration::MicroSeconds(cycles * 1000000 / GetCyclesPerSecond());
}

TDuration CyclesToDurationSafe(ui64 cycles)
{
    constexpr ui64 cyclesLimit = std::numeric_limits<ui64>::max() / 1000000;
    if (cycles <= cyclesLimit) {
        return CyclesToDuration(cycles);
    }
    return TDuration::MicroSeconds(cycles / GetCyclesPerSecond() * 1000000);
}

ui64 DurationToCycles(TDuration duration) {
    return duration.MicroSeconds() * GetCyclesPerSecond() / 1000000;
}

ui64 DurationToCyclesSafe(TDuration duration)
{
    if (duration.MicroSeconds() <= std::numeric_limits<ui64>::max() / GetCyclesPerSecond()) {
        return DurationToCycles(duration);
    }
    return duration.MicroSeconds() / 1000000 * GetCyclesPerSecond();
}

TPrecisionTimer::TPrecisionTimer()
    : Start(::GetCycleCount())
{
}

ui64 TPrecisionTimer::GetCycleCount() const {
    return ::GetCycleCount() - Start;
}

TString FormatCycles(ui64 cycles) {
    ui64 milliseconds = cycles / GetCyclesPerMillisecond();
    ui32 ms = ui32(milliseconds % 1000);
    milliseconds /= 1000;
    ui32 secs = ui32(milliseconds % 60);
    milliseconds /= 60;
    ui32 mins = ui32(milliseconds);
    TString result;
    sprintf(result, "%" PRIu32 " m %.2" PRIu32 " s %.3" PRIu32 " ms", mins, secs, ms);
    return result;
}

TFormattedPrecisionTimer::TFormattedPrecisionTimer(const char* message, IOutputStream* out)
    : Message(message)
    , Out(out)
{
    Start = GetCycleCount();
}

TFormattedPrecisionTimer::~TFormattedPrecisionTimer() {
    const ui64 end = GetCycleCount();
    const ui64 diff = end - Start;

    *Out << Message << ": " << diff << " ticks " << FormatCycles(diff) << Endl;
}

TFuncTimer::TFuncTimer(const char* func)
    : Start_(TInstant::Now())
    , Func_(func)
{
    Cerr << "enter " << Func_ << Endl;
}

TFuncTimer::~TFuncTimer() {
    Cerr << "leave " << Func_ << " -> " << (TInstant::Now() - Start_) << Endl;
}

TTimeLogger::TTimeLogger(const TString& message, bool verbose)
    : Message(message)
    , Verbose(verbose)
    , OK(false)
    , Begin(time(nullptr))
    , BeginCycles(GetCycleCount())
{
    if (Verbose) {
        fprintf(stderr, "=========================================================\n");
        fprintf(stderr, "%s started: %.24s (%lu) (%d)\n", Message.data(), ctime(&Begin), (unsigned long)Begin, (int)getpid());
    }
}

double TTimeLogger::ElapsedTime() const {
    return time(nullptr) - Begin;
}

void TTimeLogger::SetOK() {
    OK = true;
}

TTimeLogger::~TTimeLogger() {
    time_t tim = time(nullptr);
    ui64 endCycles = GetCycleCount();
    if (Verbose) {
        const char* prefix = (OK) ? "" : "!";
        fprintf(stderr, "%s%s ended: %.24s (%lu) (%d) (took %lus = %s)\n",
                prefix, Message.data(), ctime(&tim), (unsigned long)tim, (int)getpid(),
                (unsigned long)tim - (unsigned long)Begin, FormatCycles(endCycles - BeginCycles).data());
        fprintf(stderr, "%s=========================================================\n", prefix);
    }
}
