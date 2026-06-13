#pragma once

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

/*!
 * Index of the CPU the calling thread currently runs on, in [0, NumberOfCpus).
 * On Linux/x86-64 and Linux/arm64 the fast path is a single thread-local read of
 * the kernel's rseq area; the first call per thread lazily registers rseq, and
 * reads fall back to sched_getcpu() when rseq is unavailable. On other platforms
 * it is sched_getcpu() (Linux) or 0.
 * It is a hint (the thread may migrate right after) -- for sharding, not correctness.
 *
 * The rseq backend lives in library/cpp/yt/rseq, with no third-party dependency.
 *
 * Fiber-TLS: the fast path is inlined and reads the thread pointer, so it must be
 * reached only through a non-inlinable, fiber-switch-free frame (a virtual call or
 * YT_PREVENT_TLS_CACHING; see library/cpp/yt/misc/tls.h). Otherwise the thread base
 * may be cached and reused after the fiber resumes on another OS thread, reading a
 * different -- possibly already-freed -- thread's area.
 */
int GetCurrentCpuId();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define CPU_ID_INL_H_
#include "cpu_id-inl.h"
#undef CPU_ID_INL_H_
