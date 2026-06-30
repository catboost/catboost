#pragma once

#include <library/cpp/yt/cpu_clock/clock.h>

#include <util/generic/bitops.h>

namespace NYT::NProfiling {

////////////////////////////////////////////////////////////////////////////////

//! Timestamp counter + processor id.
struct TTscp
{
    //! Defines the range for TTscp::ProcessorId.
    //! Always a power of 2.
    static constexpr int MaxProcessorId = 64;
    static_assert(IsPowerOf2(MaxProcessorId), "MaxProcessorId must be a power of 2.");

    //! The moment this TTscp instance was created.
    TCpuInstant Instant;

    //! "Processor id", always taken modulo #MaxProcessorId.
    //! There's no guarantee that same value indicates the same phisycal core.
    int ProcessorId;

    //! Returns the current timestamp counter and processor id, both obtained from a single serializing rdtscp.
    static TTscp Get();

    //! A cheaper, lower-precision counterpart of #Get: the processor id comes from the
    //! rseq fast path (GetCurrentCpuId) and the instant from a non-serializing rdtsc
    //! (GetApproximateCpuInstant), instead of a single serializing rdtscp.
    //!
    //! WARNING (fiber TLS): inlined and reads the thread pointer (via GetCurrentCpuId),
    //! so it must be reached only through a non-inlinable, fiber-switch-free frame (a
    //! virtual call or YT_PREVENT_TLS_CACHING; see library/cpp/yt/misc/tls.h).
    static TTscp GetApproximate();
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NProfiling

#define TSCP_INL_H_
#include "tscp-inl.h"
#undef TSCP_INL_H_
