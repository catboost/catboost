#pragma once

// Rseq is available on Linux for the arches where we can read the thread pointer.
// Consumers must guard rseq usage with this macro.
#if defined(__linux__) && (defined(__x86_64__) || defined(__aarch64__))
    #define YT_RSEQ_AVAILABLE
#endif

#ifdef YT_RSEQ_AVAILABLE

#include <cstddef>

namespace NYT::NRseq {

////////////////////////////////////////////////////////////////////////////////

//! Byte offset from the thread pointer to the rseq area's cpu_id field (glibc's area when
//! glibc registers rseq, otherwise our own). A fixed offset across threads only when the area
//! is glibc-owned or in the static TLS block; #IsPerCpuFastPathSafe probes this and gates the
//! fast path. NB: 0 until our startup initializer runs, but nothing reads rseq before then.
extern std::ptrdiff_t CpuIdFieldOffset;

//! Reads a field of the calling thread's rseq area at the given offset from the thread
//! pointer. Branch-free and force-inlined.
//!
//! WARNING (fiber TLS): reads the thread pointer, so it must be reached only through a
//! non-inlinable, fiber-switch-free frame (YT_PREVENT_TLS_CACHING, see library/cpp/yt/misc/tls.h).
template <class T>
T ReadField(std::ptrdiff_t fieldOffset);

//! Registers the calling thread with rseq if we own the registration and it is not
//! registered yet. Returns false if rseq is unavailable (no kernel support). Cheap
//! once registered.
bool EnsureCurrentThreadRegistered();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq

#define RSEQ_INL_H_
#include "rseq-inl.h"
#undef RSEQ_INL_H_

#endif // YT_RSEQ_AVAILABLE
