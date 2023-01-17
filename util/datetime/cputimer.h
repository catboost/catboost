#pragma once

#include "base.h"

#include <util/system/rusage.h>
#include <util/generic/string.h>
#include <util/stream/str.h>

class TTimer {
private:
    TInstant Start_;
    TStringStream Message_;

public:
    TTimer(const TStringBuf message = TStringBuf(" took: "));
    ~TTimer();
};

class TSimpleTimer {
    TInstant T;

public:
    TSimpleTimer() {
        Reset();
    }
    TDuration Get() const {
        return TInstant::Now() - T;
    }
    void Reset() {
        T = TInstant::Now();
    }
};

class TProfileTimer {
    TDuration T;

public:
    TProfileTimer() {
        Reset();
    }
    TDuration Get() const {
        return TRusage::Get().Utime - T;
    }
    TDuration Step() {
        TRusage r;
        r.Fill();
        TDuration d = r.Utime - T;
        T = r.Utime;
        return d;
    }
    void Reset() {
        T = TRusage::Get().Utime;
    }
};

/// Return cached processor cycle count per second. Method takes 1 second at first invocation.
/// Note, on older systems cycle rate may change during program lifetime,
/// so returned value may be incorrect. Modern Intel and AMD processors keep constant TSC rate.
ui64 GetCyclesPerMillisecond();
void SetCyclesPerSecond(ui64 cycles);

TDuration CyclesToDuration(ui64 cycles);
ui64 DurationToCycles(TDuration duration);

// NBS-3400 - CyclesToDuration and DurationToCycles may overflow for long running events
TDuration CyclesToDurationSafe(ui64 cycles);
ui64 DurationToCyclesSafe(TDuration duration);

class TPrecisionTimer {
private:
    ui64 Start = 0;

public:
    TPrecisionTimer();

    ui64 GetCycleCount() const;
};

TString FormatCycles(ui64 cycles);

class TFormattedPrecisionTimer {
private:
    ui64 Start;
    const char* Message;
    IOutputStream* Out;

public:
    TFormattedPrecisionTimer(const char* message = "took ", IOutputStream* out = &Cout);
    ~TFormattedPrecisionTimer();
};

class TFuncTimer {
public:
    TFuncTimer(const char* func);
    ~TFuncTimer();

private:
    const TInstant Start_;
    const char* Func_;
};

class TFakeTimer {
public:
    inline TFakeTimer(const char* = nullptr) noexcept {
    }
};

#if defined(WITH_DEBUG)
    #define TDebugTimer TFuncTimer
#else
    #define TDebugTimer TFakeTimer
#endif

class TTimeLogger {
private:
    TString Message;
    bool Verbose;
    bool OK;
    time_t Begin;
    ui64 BeginCycles;

public:
    TTimeLogger(const TString& message, bool verbose = true);
    ~TTimeLogger();

    void SetOK();
    double ElapsedTime() const;
};
