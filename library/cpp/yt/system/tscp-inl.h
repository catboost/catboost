#ifndef TSCP_INL_H_
#error "Direct inclusion of this file is not allowed, include tscp.h"
// For the sake of sane code completion.
#include "tscp.h"
#endif

#include "cpu_id.h"

#include <util/system/cpu_id.h>

namespace NYT::NProfiling {

////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////

#if defined(__x86_64__)

const bool SupportsRdtscp = NX86::HaveRDTSCP();

#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace

inline TTscp TTscp::Get()
{
#if defined(__x86_64__)
    ui64 t;
    ui64 c;
    if (SupportsRdtscp) {
        ui64 rax, rcx, rdx;
        asm volatile ( "rdtscp\n" : "=a" (rax), "=c" (rcx), "=d" (rdx) : : );
        t = (rdx << 32) + rax;
        c = rcx;
    } else {
        ui64 rax, rdx;
        asm volatile ( "rdtsc\n" : "=a" (rax), "=d" (rdx) : : );
        t = (rdx << 32) + rax;

        // cpuId[1] >> 24 is an APIC id.
        ui32 cpuId[4];
        NX86::CpuId(1, 0, cpuId);
        c = cpuId[1] >> 24;
    }
#elif defined(__arm64__) || defined(__aarch64__)
    ui64 c;
    __asm__ volatile("mrs %x0, tpidrro_el0" : "=r"(c));
    c = c & 0x07u;
    TCpuInstant t = GetCpuInstant();
#endif
    return TTscp{
        .Instant = static_cast<TCpuInstant>(t),
        .ProcessorId = static_cast<int>(c) & (MaxProcessorId - 1)
    };
}

inline TTscp TTscp::GetApproximate()
{
    return TTscp{
        .Instant = GetApproximateCpuInstant(),
        .ProcessorId = NYT::GetCurrentCpuId() & (MaxProcessorId - 1),
    };
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NProfiling
