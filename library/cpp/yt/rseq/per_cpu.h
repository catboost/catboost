#pragma once

#include <util/system/types.h>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace NYT::NRseq {

// The library is Linux-only: rseq is a Linux kernel feature and the build enforces OS_LINUX
// (see ya.make); off the rseq fast path the primitives fall back to plain atomics.

////////////////////////////////////////////////////////////////////////////////

//! Number of shards a per-CPU array must provide -- one per CPU.
/*!
 *  Equals nr_cpu_ids (highest possible CPU id + 1), from /sys/devices/system/cpu/possible so
 *  it covers offlined and hot-pluggable CPUs. The fast path indexes by the raw rseq cpu_id, so
 *  a plain count (e.g. _SC_NPROCESSORS_CONF) would undersize the array on sparse topologies --
 *  it is only the fallback when the bitmap is unreadable. Always >= 1; cached.
 */
int GetCpuCount();

//! Adds |delta| to the calling CPU's slot of a per-CPU array of shards, lock-free.
/*!
 *  |shards| is an array of GetCpuCount() |TShard| slots (typically cache-line padded); |field|
 *  selects the |TValue| to update. The stride is sizeof(TShard), which must be a multiple of 8
 *  (checked at compile time) so the field stays 8-byte aligned (for a tear-free RMW).
 *
 *  Fast path (x86-64 Linux): a non-atomic read-modify-write committed by an rseq critical
 *  section -- no atomic, no lock; safe against preemption/migration (the kernel restarts it)
 *  and other threads (one thread per CPU). Otherwise (non-x86-64 Linux, or no kernel rseq): an
 *  atomic fetch_add. A process uses one path consistently, so the two never mix on a slot
 *  (except on exotic sparse topologies; see per_cpu.cpp).
 *
 *  WARNING (fiber TLS): the fast path reads the thread pointer, so reach #AddPerCpu only via a
 *  non-inlinable, fiber-switch-free frame (a virtual call or YT_PREVENT_TLS_CACHING; see
 *  library/cpp/yt/misc/tls.h).
 */
template <class TShard, class TValue>
    requires std::integral<TValue> && (sizeof(TValue) == 8)
void AddPerCpu(TShard* shards, TValue TShard::* field, TValue delta);

//! Stores |value| (8 or 16 bytes) into the calling CPU's slot, lock-free.
/*!
 *  |shards| / |field| as in #AddPerCpu; |TValue| is an 8- or 16-byte trivially-copyable type.
 *
 *  Fast path (x86-64 Linux): an rseq-committed store (movq for 8 bytes, movdqu for 16);
 *  otherwise relaxed atomic store(s). An 8-byte store is single-copy atomic, never torn; a
 *  16-byte store is not atomic on either path, so a reader on another CPU may see the halves
 *  split -- fine for a last-writer-wins gauge.
 *
 *  WARNING (fiber TLS): same contract as #AddPerCpu.
 */
template <class TShard, class TValue>
    requires (sizeof(TValue) == 8 || sizeof(TValue) == 16) && std::is_trivially_copyable_v<TValue>
void StorePerCpu(TShard* shards, TValue TShard::* field, TValue value);

//! Relaxed atomic load of slot |index| -- the reader counterpart of #AddPerCpu.
/*!
 *  |shards| / |field| as in #AddPerCpu; reads shards[index].*field for |index| in
 *  [0, GetCpuCount()). Not tied to the calling CPU (no rseq fast path); aggregate a counter by
 *  loading every slot and summing.
 */
template <class TShard, class TValue>
    requires std::integral<TValue> && (sizeof(TValue) == 8)
TValue LoadPerCpu(const TShard* shards, TValue TShard::* field, int index);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq

#define PER_CPU_INL_H_
#include "per_cpu-inl.h"
#undef PER_CPU_INL_H_
