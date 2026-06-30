#ifndef PER_CPU_INL_H_
#error "Direct inclusion of this file is not allowed, include per_cpu.h"
// For the sake of sane code completion.
#include "per_cpu.h"
#endif

#include <library/cpp/yt/assert/assert.h>

#include <util/system/compiler.h>

#if defined(__x86_64__)
#include <emmintrin.h>  // __m128i -- used only by the x86 rseq fast-path 16-byte store
#endif

#include <array>
#include <string_view>

// The full rseq fast path (a per-CPU non-atomic add committed by an rseq critical
// section) is implemented for x86-64 Linux only. Everywhere else AddPerCpu uses the
// atomic fallback.
#if defined(__x86_64__)
    #include "rseq.h"
    #define YT_RSEQ_PERCPU_FAST
#endif

namespace NYT::NRseq {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

////////////////////////////////////////////////////////////////////////////////

//! Returns a CPU id in [0, GetCpuCount()) for the atomic fallback path.
int GetFallbackCpuId();

//! Parses a kernel CPU range list (e.g. "0-3,8-11") and returns the highest CPU id plus one,
//! or -1 if the list contains no id. Exposed for testing; #GetCpuCount feeds it the
//! /sys/devices/system/cpu/possible bitmap.
int ParsePossibleCpuCount(std::string_view list);

//! Cached #GetCpuCount value used by the fast path's bounds check (see per_cpu.cpp).
/*!
 *  The fast path indexes the slot array by the raw rseq cpu_id and bounds-checks it against
 *  this value with a single unsigned compare (which also rejects a negative, unregistered
 *  cpu_id); an out-of-range cpu_id takes the clamped atomic fallback instead. Set by the
 *  first call to #GetCpuCount; defaults to 0 so every update falls back until the size is
 *  known -- callers must therefore size the array via #GetCpuCount before any update.
 */
extern constinit int CpuCount;

//! Returns a pointer to slot |index| of a per-CPU array -- |base| advanced by
//! |index * stride| bytes. The size_t cast keeps the byte offset from overflowing.
template <class T>
Y_FORCE_INLINE T* GetSlot(T* base, size_t stride, int index)
{
    using TByte = std::conditional_t<std::is_const_v<T>, const char, char>;
    return reinterpret_cast<T*>(
        reinterpret_cast<TByte*>(base) + static_cast<size_t>(index) * stride);
}

#ifdef YT_RSEQ_PERCPU_FAST

//! *reinterpret_cast<i64*>(slot) += value, committed by an rseq critical section
//! validated against |cpuId|.
/*!
 *  Returns true on commit, false if the kernel aborted the sequence (caller retries).
 */
Y_FORCE_INLINE bool RseqCommitAdd8(void* slot, i64 value, int cpuId)
{
    // The kernel-managed struct rseq: cpu_id_start@0, cpu_id@4, rseq_cs@8. CpuIdFieldOffset
    // is TP -> cpu_id, so the area starts 4 bytes earlier.
    char* area = static_cast<char*>(__builtin_thread_pointer()) + CpuIdFieldOffset - 4;
    __asm__ __volatile__ goto(
        ".pushsection __rseq_cs, \"aw\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"                      // version, flags
        ".quad 2f, (3f - 2f), 4f\n\t"         // start_ip, post_commit_offset, abort_ip
        ".popsection\n\t"
        "leaq 1b(%%rip), %%rax\n\t"
        "movq %%rax, 8(%[area])\n\t"          // area->rseq_cs = &descriptor
        "2:\n\t"                              // start_ip
        "cmpl %[cpuId], 4(%[area])\n\t"       // if (area->cpu_id != cpuId) abort
        "jnz 4f\n\t"
        "addq %[value], (%[slot])\n\t"        // commit: *slot += value (non-atomic)
        "3:\n\t"                              // post_commit_ip
        ".pushsection __rseq_failure, \"ax\"\n\t"
        ".byte 0x0f, 0xb9, 0x3d\n\t"          // ud1: makes the signature a valid instruction
        ".long 0x53053053\n\t"                // rseq signature (precedes abort_ip)
        "4:\n\t"                              // abort_ip
        "jmp %l[abort]\n\t"
        ".popsection\n\t"
        :
        : [area] "r" (area), [slot] "r" (slot), [value] "r" (value), [cpuId] "r" (cpuId)
        : "rax", "memory"
        : abort);
    return true;
abort:
    return false;
}

//! *reinterpret_cast<i64*>(slot) = value (non-atomic movq), committed by an rseq critical
//! section validated against |cpuId|.
/*!
 *  Returns true on commit, false if the kernel aborted the sequence (caller retries).
 */
Y_FORCE_INLINE bool RseqCommitStore8(void* slot, i64 value, int cpuId)
{
    char* area = static_cast<char*>(__builtin_thread_pointer()) + CpuIdFieldOffset - 4;
    __asm__ __volatile__ goto(
        ".pushsection __rseq_cs, \"aw\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "leaq 1b(%%rip), %%rax\n\t"
        "movq %%rax, 8(%[area])\n\t"
        "2:\n\t"
        "cmpl %[cpuId], 4(%[area])\n\t"
        "jnz 4f\n\t"
        "movq %[value], (%[slot])\n\t"        // commit: 8-byte store (non-atomic)
        "3:\n\t"
        ".pushsection __rseq_failure, \"ax\"\n\t"
        ".byte 0x0f, 0xb9, 0x3d\n\t"
        ".long 0x53053053\n\t"
        "4:\n\t"
        "jmp %l[abort]\n\t"
        ".popsection\n\t"
        :
        : [area] "r" (area), [slot] "r" (slot), [value] "r" (value), [cpuId] "r" (cpuId)
        : "rax", "memory"
        : abort);
    return true;
abort:
    return false;
}

//! *reinterpret_cast<__m128i*>(slot) = value (single non-atomic movdqu), committed by an rseq
//! critical section validated against |cpuId|.
/*!
 *  Returns true on commit, false if the kernel aborted the sequence (caller retries). A single
 *  16-byte instruction means an abort never leaves the slot half-written; a reader on another
 *  CPU may still observe the store torn mid-flight -- acceptable for a last-writer-wins gauge.
 *  This helper is x86-only (compiled under YT_RSEQ_PERCPU_FAST), so __m128i in the signature
 *  costs nothing off x86 and keeps the value in an xmm register for the movdqu.
 */
Y_FORCE_INLINE bool RseqCommitStore16(void* slot, __m128i value, int cpuId)
{
    char* area = static_cast<char*>(__builtin_thread_pointer()) + CpuIdFieldOffset - 4;
    __asm__ __volatile__ goto(
        ".pushsection __rseq_cs, \"aw\"\n\t"
        ".balign 32\n\t"
        "1:\n\t"
        ".long 0, 0\n\t"
        ".quad 2f, (3f - 2f), 4f\n\t"
        ".popsection\n\t"
        "leaq 1b(%%rip), %%rax\n\t"
        "movq %%rax, 8(%[area])\n\t"
        "2:\n\t"
        "cmpl %[cpuId], 4(%[area])\n\t"
        "jnz 4f\n\t"
        "movdqu %[value], (%[slot])\n\t"      // commit: 16-byte store (non-atomic)
        "3:\n\t"
        ".pushsection __rseq_failure, \"ax\"\n\t"
        ".byte 0x0f, 0xb9, 0x3d\n\t"
        ".long 0x53053053\n\t"
        "4:\n\t"
        "jmp %l[abort]\n\t"
        ".popsection\n\t"
        :
        : [area] "r" (area), [slot] "r" (slot), [value] "x" (value), [cpuId] "r" (cpuId)
        : "rax", "memory"
        : abort);
    return true;
abort:
    return false;
}

//! Runs |commit(slot, cpuId)| for the calling CPU under rseq, retrying on abort.
/*!
 *  |commit| runs one rseq critical section (see the RseqCommit* helpers above) and returns
 *  true on commit, false if the kernel aborted it. Returns false when the fast path is
 *  unavailable -- the rseq cpu_id is not within [0, CpuCount) -- in which case nothing is
 *  written and the caller must use the fallback. The cpu_id is read unsigned, so the single
 *  |cpuId >= CpuCount| test also rejects an unregistered thread (whose cpu_id sentinel reads
 *  as ~0u) and a cpu_id beyond the slot array (reachable only when #GetCpuCount could not
 *  read an exact bound).
 *
 *  Must be reached only after #GetCpuCount has run (see NDetail::CpuCount); callers satisfy
 *  this by sizing the slot array with #GetCpuCount. CpuCount defaults to 0, so every update
 *  falls back until the size is known.
 */
template <class TCommit>
Y_FORCE_INLINE bool RunRseqPerCpu(void* base, size_t stride, TCommit commit)
{
    ui32 cpuId = ReadField<ui32>(CpuIdFieldOffset);
    ui32 cpuCount = CpuCount;
    if (cpuId >= cpuCount) [[unlikely]] {
        // Fresh thread not yet rseq-registered (e.g. a build without tcmalloc): register once
        // and re-read. If it stays out of range, fall back.
        EnsureCurrentThreadRegistered();
        cpuId = ReadField<ui32>(CpuIdFieldOffset);
        if (cpuId >= cpuCount) [[unlikely]] {
            return false;
        }
    }
    for (;;) {
        void* slot = GetSlot(base, stride, static_cast<int>(cpuId));
        if (commit(slot, cpuId)) [[likely]] {
            return true;
        }
        // Aborted (migration/preemption): re-read the CPU and re-validate before reusing it,
        // since after a migration it may name an out-of-range CPU.
        cpuId = ReadField<ui32>(CpuIdFieldOffset);
        if (cpuId >= cpuCount) [[unlikely]] {
            return false;
        }
    }
}

#endif // YT_RSEQ_PERCPU_FAST

//! Relaxed atomic load of |slot| (the read side of #AddPerCpu / #StorePerCpu).
template <class T>
Y_FORCE_INLINE T AtomicLoad(const T* slot)
{
    return __atomic_load_n(slot, __ATOMIC_RELAXED);
}

template <class T>
Y_FORCE_INLINE void AtomicAddPerCpu(T* base, size_t stride, T value)
{
    auto* slot = GetSlot(base, stride, GetFallbackCpuId());
    __atomic_fetch_add(slot, value, __ATOMIC_RELAXED);
}

//! Stores |value| into the calling CPU's slot with relaxed atomic stores: one 8-byte store,
//! or two for a 16-byte value (the CPU is resolved once). Each 8-byte store is single-copy
//! atomic, but the two halves of a 16-byte value may be observed split -- a torn value
//! matching the fast path, which the last-writer-wins gauge tolerates. |T| is bit-cast to
//! ui64 halves, so any 8- or 16-byte trivially-copyable type (incl. __m128i) works on any
//! arch.
template <class T>
    requires (sizeof(T) == 8 || sizeof(T) == 16)
Y_FORCE_INLINE void AtomicStorePerCpu(T* base, size_t stride, T value)
{
    auto* slot = reinterpret_cast<ui64*>(GetSlot(base, stride, GetFallbackCpuId()));
    if constexpr (sizeof(T) == 8) {
        __atomic_store_n(slot, __builtin_bit_cast(ui64, value), __ATOMIC_RELAXED);
    } else {
        auto parts = __builtin_bit_cast(std::array<ui64, 2>, value);
        __atomic_store_n(slot, parts[0], __ATOMIC_RELAXED);
        __atomic_store_n(slot + 1, parts[1], __ATOMIC_RELAXED);
    }
}

// base + stride implementations behind the public pointer-to-member API below.

template <class T>
    requires std::integral<T> && (sizeof(T) == 8)
Y_FORCE_INLINE void AddPerCpuImpl(T* base, size_t stride, T value)
{
#ifdef YT_RSEQ_PERCPU_FAST
    i64 delta = static_cast<i64>(value);
    if (RunRseqPerCpu(base, stride, [&] (void* slot, int cpuId) {
            return RseqCommitAdd8(slot, delta, cpuId);
        })) [[likely]]
    {
        return;
    }
#endif
    AtomicAddPerCpu(base, stride, value);
}

template <class T>
    requires (sizeof(T) == 8 || sizeof(T) == 16) && std::is_trivially_copyable_v<T>
Y_FORCE_INLINE void StorePerCpuImpl(T* base, size_t stride, T value)
{
#ifdef YT_RSEQ_PERCPU_FAST
    if constexpr (sizeof(T) == 16) {
        auto packed = __builtin_bit_cast(__m128i, value);
        if (RunRseqPerCpu(base, stride, [&] (void* slot, int cpuId) {
                return RseqCommitStore16(slot, packed, cpuId);
            })) [[likely]]
        {
            return;
        }
    } else {
        auto packed = __builtin_bit_cast(i64, value);
        if (RunRseqPerCpu(base, stride, [&] (void* slot, int cpuId) {
                return RseqCommitStore8(slot, packed, cpuId);
            })) [[likely]]
        {
            return;
        }
    }
#endif
    AtomicStorePerCpu(base, stride, value);
}

template <class T>
    requires std::integral<T> && (sizeof(T) == 8)
Y_FORCE_INLINE T LoadPerCpuImpl(const T* base, size_t stride, int index)
{
    return AtomicLoad(GetSlot(base, stride, index));
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <class TShard, class TValue>
    requires std::integral<TValue> && (sizeof(TValue) == 8)
Y_FORCE_INLINE void AddPerCpu(TShard* shards, TValue TShard::* field, TValue delta)
{
    static_assert(sizeof(TShard) % 8 == 0, "Shard size must be a multiple of 8");
    NDetail::AddPerCpuImpl(&(shards[0].*field), sizeof(TShard), delta);
}

template <class TShard, class TValue>
    requires (sizeof(TValue) == 8 || sizeof(TValue) == 16) && std::is_trivially_copyable_v<TValue>
Y_FORCE_INLINE void StorePerCpu(TShard* shards, TValue TShard::* field, TValue value)
{
    static_assert(sizeof(TShard) % 8 == 0, "Shard size must be a multiple of 8");
    NDetail::StorePerCpuImpl(&(shards[0].*field), sizeof(TShard), value);
}

template <class TShard, class TValue>
    requires std::integral<TValue> && (sizeof(TValue) == 8)
Y_FORCE_INLINE TValue LoadPerCpu(const TShard* shards, TValue TShard::* field, int index)
{
    static_assert(sizeof(TShard) % 8 == 0, "Shard size must be a multiple of 8");
    return NDetail::LoadPerCpuImpl(&(shards[0].*field), sizeof(TShard), index);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq
