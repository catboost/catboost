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
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NProfiling

#define TSCP_INL_H_
#include "tscp-inl.h"
#undef TSCP_INL_H_
