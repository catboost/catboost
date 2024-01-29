#pragma once

#include <stddef.h>

#include <library/cpp/yt/misc/enum.h>

#include <library/cpp/yt/containers/enum_indexed_array.h>

#include <util/system/types.h>

#include <util/generic/size_literals.h>

#include <util/datetime/base.h>

namespace NYT::NYTAlloc {

////////////////////////////////////////////////////////////////////////////////
// Macros

#if defined(_linux_) && \
    !defined(_asan_enabled_) && \
    !defined(_msan_enabled_) && \
    !defined(_tsan_enabled_)
    #define YT_ALLOC_ENABLED
#endif

////////////////////////////////////////////////////////////////////////////////
// Constants

constexpr int SmallRankCount = 23;
constexpr int MinLargeRank = 15;
constexpr int LargeRankCount = 30;
constexpr size_t LargeAllocationSizeThreshold = 32_KB;
constexpr size_t HugeAllocationSizeThreshold = 1ULL << (LargeRankCount - 1);
constexpr size_t MaxAllocationSize = 1_TB;
constexpr size_t PageSize = 4_KB;
constexpr size_t RightReadableAreaSize = 16;

////////////////////////////////////////////////////////////////////////////////
// Allocation API

// Allocates a chunk of memory of (at least) #size bytes.
// The returned pointer is guaranteed to be 16-byte aligned.
// Moreover, it is guaranteeed that #RightReadableAreaSize bytes immediately following
// the allocated chunk are readable (but may belong to another allocated chunk).
// This enables eliminating some nasty corner cases in SIMD memory manipulations.
void* Allocate(size_t size);

// Allocates a chunk of memory of (at least) #size bytes.
// The returned pointer is guaranteed to be 4K-byte aligned.
// #size, however, need not be divisible by page size (but internally it will be rounded up).
void* AllocatePageAligned(size_t size);

// An optimized version of #Allocate with #Size being known at compile-time.
template <size_t Size>
void* AllocateConstSize();

// Frees a chunk of memory previously allocated via Allocate functions.
// Does nothing if #ptr is null.
void Free(void* ptr);

// Similar to #Free but assumes that #ptr is not null.
void FreeNonNull(void* ptr);

// Returns the size of the chunk pointed to by #ptr.
// This size is not guaranteed to be exactly equal to #size passed to allocation functions
// due to rounding; the returned size, however, is never less than the latter size.
// If #ptr is null or we are unable to determine the allocation size, then 0 is returned.
size_t GetAllocationSize(const void* ptr);

// Returns the size of the chunk that will actually be allocated
// when requesting an allocation of given #size. This is never less than #size.
size_t GetAllocationSize(size_t size);

////////////////////////////////////////////////////////////////////////////////
// Memory zone API
//
// Each allocation is either in the "normal zone" or "undumpable zone".
// The latter indicates that this memory region will be excluded from a coredump
// should it happen.
//
// The current zone used for allocations is stored in TLS.

// Memory zone is used to pass hint to the allocator.
DEFINE_ENUM(EMemoryZone,
    ((Unknown)    (-1)) // not a valid zone
    ((Normal)     ( 0)) // default memory type
    ((Undumpable) ( 1)) // memory is omitted from the core dump
);

// Updates the current zone in TLS.
void SetCurrentMemoryZone(EMemoryZone zone);

// Returns the current zone from TLS.
EMemoryZone GetCurrentMemoryZone();

// Returns the zone where #ptr resides;
// EMemoryZone::Invalid indicates that #ptr is outside of any recognized memory zone.
EMemoryZone GetAllocationMemoryZone(const void* ptr);

////////////////////////////////////////////////////////////////////////////////
// When a "timing event" (hiccup) occurs during an allocation,
// YTAlloc records this event and captures the current fiber id.
// The latter is provided externally by calling SetCurrentFiberId.
//
// This may be helpful to correlate various application-level timings
// with internal events in YTAlloc.
//
// The current fiber id is stored in TLS.

using TFiberId = ui64;

// Updates the current fiber id in TLS.
void SetCurrentFiberId(TFiberId id);

// Returns the currently assinged fiber id from TLS.
TFiberId GetCurrentFiberId();

////////////////////////////////////////////////////////////////////////////////
// Logging

DEFINE_ENUM(ELogEventSeverity,
    (Debug)
    (Info)
    (Warning)
    (Error)
);

struct TLogEvent
{
    ELogEventSeverity Severity;
    TStringBuf Message;
};

using TLogHandler = void(*)(const TLogEvent& event);

// Sets the handler to be invoked for each log event produced by YTAlloc.
// Can be called multiple times (but calls to the previous incarnations of the handler
// are racy).
void EnableLogging(TLogHandler logHandler);

////////////////////////////////////////////////////////////////////////////////
// Backtraces

using TBacktraceProvider = int(*)(void** frames, int maxFrames, int skipFrames);

// Sets the provider used for collecting backtraces when allocation profiling
// is turned ON. Can be called multiple times (but calls to the previous
// incarnations of the provider are racy).
void SetBacktraceProvider(TBacktraceProvider provider);

using TBacktraceFormatter = TString(*)(const void* const* frames, int frameCount);

// Sets the callback used for formatting backtraces during large arena mmap calls
// to help detect memory leaks. Can be called multiple times (but calls to the
// previous incarnations of the provider are racy).
void SetBacktraceFormatter(TBacktraceFormatter provider);

////////////////////////////////////////////////////////////////////////////////
// Misc

//! Tries to mlock all opened file mappings of the current process.
//! Typically invoked on application startup to lock all binaries in memory
//! and prevent executable code and static data to be paged out
//! causing latency spikes.
void MlockFileMappings(bool populate = true);

////////////////////////////////////////////////////////////////////////////////
// Configuration API

// Calling this function enables periodic calls to madvise(ADV_STOCKPILE);
// cf. https://st.yandex-team.ru/KERNEL-186
void EnableStockpile();

// Sets the interval between madvise(ADV_STOCKPILE) calls.
// Only makes sense if stockpile was enabled.
void SetStockpileInterval(TDuration value);

// Sets the number of threads to be invoking madvise(ADV_STOCKPILE).
// This call should be made before calling #EnableStockpile.
void SetStockpileThreadCount(int value);

// Sets the size passsed to madvise(ADV_STOCKPILE) calls.
// Only makes sense if stockpile was enabled.
void SetStockpileSize(size_t value);

// For large blobs, YTAlloc keeps at least
// LargeUnreclaimableCoeff * TotalLargeBytesUsed clamped to range
// [MinLargeUnreclaimableBytes, MaxLargeUnreclaimableBytes]
// bytes of pooled (unreclaimable) memory.
void SetLargeUnreclaimableCoeff(double value);
void SetMinLargeUnreclaimableBytes(size_t value);
void SetMaxLargeUnreclaimableBytes(size_t value);

// When a syscall (mmap, munmap, or madvise) or an internal lock acquisition
// takes longer then the configured time, a "timing event" is recorded.
void SetTimingEventThreshold(TDuration value);

// Toggles the global allocation profiling knob (OFF by default).
// For profiled allocations, YTAlloc collects (see #SetBacktraceProvider) and aggregates their
// backtraces.
void SetAllocationProfilingEnabled(bool value);

// Determines the fraction of allocations to be sampled for profiling.
void SetAllocationProfilingSamplingRate(double rate);

// Controls if small allocations of a given rank are profiled (OFF by default).
void SetSmallArenaAllocationProfilingEnabled(size_t rank, bool value);

// Controls if large allocations of a given rank are profiled (OFF by default).
void SetLargeArenaAllocationProfilingEnabled(size_t rank, bool value);

// Controls the depth of the backtraces to collect. Deeper backtraces
// take more time and affect the program performance.
void SetProfilingBacktraceDepth(int depth);

// Controls the minimum number of bytes a certain backtrace must
// allocate to appear in profiling reports.
void SetMinProfilingBytesUsedToReport(size_t size);

// If set to true (default), YTAlloc uses madvise with MADV_DONTNEED to release unused large blob pages
// (slower but leads to more predicable RSS values);
// if false then MADV_FREE is used instead, if available
// (faster but RSS may get stuck arbitrary higher than the actual usage as long
// as no memory pressure is applied).
void SetEnableEagerMemoryRelease(bool value);

// If set to true, YTAlloc uses madvise with MADV_POPULATE to prefault freshly acclaimed pages.
// Otherwise (this is the default), these pages are prefaulted with linear memory access.
// See https://st.yandex-team.ru/KERNEL-185.
void SetEnableMadvisePopulate(bool value);

////////////////////////////////////////////////////////////////////////////////
// Statistics API

DEFINE_ENUM(EBasicCounter,
    (BytesAllocated)
    (BytesFreed)
    (BytesUsed)
);

using ESystemCounter = EBasicCounter;
using ESmallCounter = EBasicCounter;
using ELargeCounter = EBasicCounter;
using EUndumpableCounter = EBasicCounter;

DEFINE_ENUM(ESmallArenaCounter,
    (PagesMapped)
    (BytesMapped)
    (PagesCommitted)
    (BytesCommitted)
);

DEFINE_ENUM(ELargeArenaCounter,
    (BytesSpare)
    (BytesOverhead)
    (BlobsAllocated)
    (BlobsFreed)
    (BlobsUsed)
    (BytesAllocated)
    (BytesFreed)
    (BytesUsed)
    (ExtentsAllocated)
    (PagesMapped)
    (BytesMapped)
    (PagesPopulated)
    (BytesPopulated)
    (PagesReleased)
    (BytesReleased)
    (PagesCommitted)
    (BytesCommitted)
    (OverheadBytesReclaimed)
    (SpareBytesReclaimed)
);

DEFINE_ENUM(EHugeCounter,
    (BytesAllocated)
    (BytesFreed)
    (BytesUsed)
    (BlobsAllocated)
    (BlobsFreed)
    (BlobsUsed)
);

DEFINE_ENUM(ETotalCounter,
    (BytesAllocated)
    (BytesFreed)
    (BytesUsed)
    (BytesCommitted)
    (BytesUnaccounted)
);

// Returns statistics for all user allocations.
TEnumIndexedArray<ETotalCounter, ssize_t> GetTotalAllocationCounters();

// Returns statistics for small allocations; these are included into total statistics.
TEnumIndexedArray<ESmallCounter, ssize_t> GetSmallAllocationCounters();

// Returns statistics for large allocations; these are included into total statistics.
TEnumIndexedArray<ELargeCounter, ssize_t> GetLargeAllocationCounters();

// Returns per-arena statistics for small allocations; these are included into total statistics.
std::array<TEnumIndexedArray<ESmallArenaCounter, ssize_t>, SmallRankCount> GetSmallArenaAllocationCounters();

// Returns per-arena statistics for large allocations; these are included into total statistics.
std::array<TEnumIndexedArray<ELargeArenaCounter, ssize_t>, LargeRankCount> GetLargeArenaAllocationCounters();

// Returns statistics for huge allocations; these are included into total statistics.
TEnumIndexedArray<EHugeCounter, ssize_t> GetHugeAllocationCounters();

// Returns statistics for all system allocations; these are not included into total statistics.
TEnumIndexedArray<ESystemCounter, ssize_t> GetSystemAllocationCounters();

// Returns statistics for undumpable allocations.
TEnumIndexedArray<EUndumpableCounter, ssize_t> GetUndumpableAllocationCounters();

DEFINE_ENUM(ETimingEventType,
    (Mmap)
    (Munmap)
    (MadvisePopulate)
    (MadviseFree)
    (MadviseDontNeed)
    (Locking)
    (Prefault)
    (FilePrefault)
);

struct TTimingEventCounters
{
    // Number of events happened since start.
    size_t Count = 0;
    // Total size of memory blocks involved in these events (if applicable).
    size_t Size = 0;
};

// Returns statistics for timing events happened since start.
// See SetTimingEventThreshold.
TEnumIndexedArray<ETimingEventType, TTimingEventCounters> GetTimingEventCounters();

////////////////////////////////////////////////////////////////////////////////

// We never collect backtraces deeper than this limit.
constexpr int MaxAllocationProfilingBacktraceDepth = 16;

struct TBacktrace
{
    int FrameCount;
    std::array<void*, MaxAllocationProfilingBacktraceDepth> Frames;
};

struct TProfiledAllocation
{
    TBacktrace Backtrace;
    TEnumIndexedArray<EBasicCounter, ssize_t> Counters;
};

// Returns statistics for profiled allocations (available when allocation
// profiling is ON). Allocations are grouped by backtrace; for each backtrace
// we provide the counters indicating the number of allocated, freed, and used bytes.
// To appear here, used bytes counter must be at least the value configured
// via SetMinProfilingBytesUsedToReport.
std::vector<TProfiledAllocation> GetProfiledAllocationStatistics();

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYTAlloc

#define YT_ALLOC_INL_H_
#include "ytalloc-inl.h"
#undef YT_ALLOC_INL_H_
