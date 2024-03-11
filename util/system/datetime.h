#pragma once

#include "defaults.h"
#include "platform.h"

#if defined(_win_)
    #include <intrin.h>
    #pragma intrinsic(__rdtsc)
#endif // _win_

#if defined(_darwin_) && !defined(_x86_)
    #include <mach/mach_time.h>
#endif

/// util/system/datetime.h contains only system time providers
/// for handy datetime utilities include util/datetime/base.h

/// Current time in microseconds since epoch
ui64 MicroSeconds() noexcept;
/// Current time in milliseconds since epoch
inline ui64 MilliSeconds() {
    return MicroSeconds() / ui64(1000);
}
/// Current time in milliseconds since epoch (deprecated, use MilliSeconds instead)
inline ui64 millisec() {
    return MilliSeconds();
}
/// Current time in seconds since epoch
ui32 Seconds() noexcept;
///Current thread time in microseconds
ui64 ThreadCPUUserTime() noexcept;
ui64 ThreadCPUSystemTime() noexcept;
ui64 ThreadCPUTime() noexcept;

void NanoSleep(ui64 ns) noexcept;

#if defined(_x86_)
namespace NPrivate {
    bool HaveRdtscpImpl();
}
#endif

// GetCycleCount guarantees to return synchronous values on different cores
// and provide constant rate only on modern Intel and AMD processors
// NOTE: rdtscp is used to prevent out of order execution
// rdtsc can be reordered, while rdtscp cannot be reordered
// with preceding instructions
// PERFORMANCE: rdtsc - 15 cycles per call , rdtscp - 19 cycles per call
// WARNING: following instruction can be executed out-of-order
Y_FORCE_INLINE ui64 GetCycleCount() noexcept {
#if defined(_MSC_VER)
    // Generates the rdtscp instruction, which returns the processor time stamp.
    // The processor time stamp records the number of clock cycles since the last reset.
    static const bool haveRdtscp = ::NPrivate::HaveRdtscpImpl();

    if (haveRdtscp) {
        unsigned int aux;
        return __rdtscp(&aux);
    } else {
        return __rdtsc();
    }
#elif defined(_x86_64_)
    static const bool haveRdtscp = ::NPrivate::HaveRdtscpImpl();

    unsigned hi, lo;

    if (haveRdtscp) {
        __asm__ __volatile__("rdtscp"
                             : "=a"(lo), "=d"(hi)::"%rcx");
    } else {
        __asm__ __volatile__("rdtsc"
                             : "=a"(lo), "=d"(hi));
    }

    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
#elif defined(_i386_)
    static const bool haveRdtscp = ::NPrivate::HaveRdtscpImpl();

    ui64 x;
    if (haveRdtscp) {
        __asm__ volatile("rdtscp\n\t"
                         : "=A"(x)::"%ecx");
    } else {
        __asm__ volatile("rdtsc\n\t"
                         : "=A"(x));
    }
    return x;
#elif defined(_darwin_)
    return mach_absolute_time();
#elif defined(__clang__) && !defined(_arm_)
    return __builtin_readcyclecounter();
#elif defined(_arm32_)
    return MicroSeconds();
#elif defined(_arm64_)
    ui64 x;

    __asm__ __volatile__("isb; mrs %0, cntvct_el0"
                         : "=r"(x));

    return x;
#else
    #error "unsupported arch"
#endif
}
