#pragma once

#include <util/system/types.h>

#include <cstddef>

namespace NYT::NRseq::NDetail {

////////////////////////////////////////////////////////////////////////////////

//! The kernel-managed struct rseq (original ABI).
struct alignas(32) TRseqArea
{
    ui32 CpuIdStart;
    ui32 CpuId;
    ui64 RseqCs;
    ui32 Flags;
    ui32 Padding[3];
};

static_assert(sizeof(TRseqArea) == 32);
static_assert(offsetof(TRseqArea, CpuId) == 4);
static_assert(offsetof(TRseqArea, RseqCs) == 8);

//! Byte offset from the thread pointer to the rseq area (glibc's area when glibc registers
//! rseq, otherwise our own). A fixed offset across threads only when the area is glibc-owned
//! or in the static TLS block; #IsPerCpuFastPathSupported probes this and gates the fast path.
//! NB: 0 until our startup initializer runs, but nothing reads rseq before then.
extern std::ptrdiff_t RseqAreaOffset;

//! Reads the field at |fieldOffset| within the calling thread's rseq area.
//! Branch-free and force-inlined.
//!
//! WARNING (fiber TLS): reads the thread pointer, so it must be reached only through a
//! non-inlinable, fiber-switch-free frame (YT_PREVENT_TLS_CACHING, see library/cpp/yt/misc/tls.h).
template <class T>
T ReadRseqAreaField(std::ptrdiff_t fieldOffset);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq::NDetail

#define RSEQ_AREA_INL_H_
#include "rseq_area-inl.h"
#undef RSEQ_AREA_INL_H_
