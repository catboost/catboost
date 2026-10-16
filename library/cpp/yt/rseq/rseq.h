#pragma once

namespace NYT::NRseq {

////////////////////////////////////////////////////////////////////////////////

//! Returns whether the per-CPU rseq fast path is safe to use in this process.
/*!
 *  The fast path reads the rseq area at a thread-pointer offset cached at startup, which is
 *  sound only when __rseq_abi sits at a fixed offset from the thread pointer (a glibc-owned
 *  area or the static TLS block, incl. tcmalloc) -- not when it lands in a dlopen'd module's
 *  dynamically allocated TLS. Returns false there so callers fall back to atomics. Decided
 *  once on a spawned thread and cached: one spawn at first use.
 */
bool IsPerCpuFastPathSupported();

//! Registers the calling thread with rseq if we own the registration and it is not
//! registered yet. Returns false if rseq is unavailable (no kernel support). Cheap
//! once registered.
bool EnsureCurrentThreadRegistered();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq
