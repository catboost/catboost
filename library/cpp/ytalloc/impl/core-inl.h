#pragma once

// This file contains the core parts of YTAlloc but no malloc/free-bridge.
// The latter bridge is placed into alloc.cpp, which includes (sic!) core-inl.h.
// This ensures that AllocateInline/FreeInline calls are properly inlined into malloc/free.
// Also core-inl.h can be directly included in, e.g., benchmarks.

#include <library/cpp/yt/containers/intrusive_linked_list.h>

#include <library/cpp/yt/memory/memory_tag.h>

#include <library/cpp/yt/threading/at_fork.h>
#include <library/cpp/yt/threading/fork_aware_spin_lock.h>

#include <library/cpp/yt/memory/free_list.h>

#include <util/system/tls.h>
#include <util/system/align.h>
#include <util/system/thread.h>

#include <util/string/printf.h>

#include <util/generic/singleton.h>
#include <util/generic/size_literals.h>
#include <util/generic/utility.h>

#include <util/digest/numeric.h>

#include <library/cpp/ytalloc/api/ytalloc.h>

#include <atomic>
#include <array>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <cstdio>
#include <optional>

#include <sys/mman.h>

#ifdef _linux_
    #include <sys/utsname.h>
#endif

#include <errno.h>
#include <pthread.h>
#include <time.h>

#ifndef MAP_POPULATE
    #define MAP_POPULATE 0x08000
#endif

// MAP_FIXED which doesn't unmap underlying mapping.
// Linux kernels older than 4.17 silently ignore this flag.
#ifndef MAP_FIXED_NOREPLACE
    #ifdef _linux_
        #define MAP_FIXED_NOREPLACE 0x100000
    #else
        #define MAP_FIXED_NOREPLACE 0
    #endif
#endif

#ifndef MADV_POPULATE
    #define MADV_POPULATE 0x59410003
#endif

#ifndef MADV_STOCKPILE
    #define MADV_STOCKPILE 0x59410004
#endif

#ifndef MADV_FREE
    #define MADV_FREE 8
#endif

#ifndef MADV_DONTDUMP
    #define MADV_DONTDUMP 16
#endif

#ifndef NDEBUG
    #define YTALLOC_PARANOID
#endif

#ifdef YTALLOC_PARANOID
    #define YTALLOC_NERVOUS
#endif

#define YTALLOC_VERIFY(condition)                                                             \
    do {                                                                                      \
        if (Y_UNLIKELY(!(condition))) {                                                       \
            ::NYT::NYTAlloc::AssertTrap("Assertion failed: " #condition, __FILE__, __LINE__); \
        }                                                                                     \
    } while (false)

#ifdef NDEBUG
    #define YTALLOC_ASSERT(condition) YTALLOC_VERIFY(condition)
#else
    #define YTALLOC_ASSERT(condition) (void)(0)
#endif

#ifdef YTALLOC_PARANOID
    #define YTALLOC_PARANOID_ASSERT(condition) YTALLOC_VERIFY(condition)
#else
    #define YTALLOC_PARANOID_ASSERT(condition) (true || (condition))
#endif

#define YTALLOC_TRAP(message) ::NYT::NYTAlloc::AssertTrap(message, __FILE__, __LINE__)

namespace NYT::NYTAlloc {

////////////////////////////////////////////////////////////////////////////////
// Allocations are classified into three types:
//
// a) Small chunks (less than LargeAllocationSizeThreshold)
// These are the fastest and are extensively cached (both per-thread and globally).
// Memory claimed for these allocations is never reclaimed back.
// Code dealing with such allocations is heavy optimized with all hot paths
// as streamlined as possible. The implementation is mostly inspired by LFAlloc.
//
// b) Large blobs (from LargeAllocationSizeThreshold to HugeAllocationSizeThreshold)
// These are cached as well. We expect such allocations to be less frequent
// than small ones but still do our best to provide good scalability.
// In particular, thread-sharded concurrent data structures as used to provide access to
// cached blobs. Memory is claimed via madvise(MADV_POPULATE) and reclaimed back
// via madvise(MADV_FREE).
//
// c) Huge blobs (from HugeAllocationSizeThreshold)
// These should be rare; we delegate directly to mmap and munmap for each allocation.
//
// We also provide a separate allocator for all system allocations (that are needed by YTAlloc itself).
// These are rare and also delegate to mmap/unmap.

// Periods between background activities.
constexpr auto BackgroundInterval = TDuration::Seconds(1);

static_assert(LargeRankCount - MinLargeRank <= 16, "Too many large ranks");
static_assert(SmallRankCount <= 32, "Too many small ranks");

constexpr size_t SmallZoneSize = 1_TB;
constexpr size_t LargeZoneSize = 16_TB;
constexpr size_t HugeZoneSize = 1_TB;
constexpr size_t SystemZoneSize = 1_TB;

constexpr size_t MaxCachedChunksPerRank = 256;

constexpr uintptr_t UntaggedSmallZonesStart = 0;
constexpr uintptr_t UntaggedSmallZonesEnd = UntaggedSmallZonesStart + 32 * SmallZoneSize;
constexpr uintptr_t MinUntaggedSmallPtr = UntaggedSmallZonesStart + SmallZoneSize * 1;
constexpr uintptr_t MaxUntaggedSmallPtr = UntaggedSmallZonesStart + SmallZoneSize * SmallRankCount;

constexpr uintptr_t TaggedSmallZonesStart = UntaggedSmallZonesEnd;
constexpr uintptr_t TaggedSmallZonesEnd = TaggedSmallZonesStart + 32 * SmallZoneSize;
constexpr uintptr_t MinTaggedSmallPtr = TaggedSmallZonesStart + SmallZoneSize * 1;
constexpr uintptr_t MaxTaggedSmallPtr = TaggedSmallZonesStart + SmallZoneSize * SmallRankCount;

constexpr uintptr_t DumpableLargeZoneStart = TaggedSmallZonesEnd;
constexpr uintptr_t DumpableLargeZoneEnd = DumpableLargeZoneStart + LargeZoneSize;

constexpr uintptr_t UndumpableLargeZoneStart = DumpableLargeZoneEnd;
constexpr uintptr_t UndumpableLargeZoneEnd = UndumpableLargeZoneStart + LargeZoneSize;

constexpr uintptr_t LargeZoneStart(bool dumpable)
{
    return dumpable ? DumpableLargeZoneStart : UndumpableLargeZoneStart;
}
constexpr uintptr_t LargeZoneEnd(bool dumpable)
{
    return dumpable ? DumpableLargeZoneEnd : UndumpableLargeZoneEnd;
}

constexpr uintptr_t HugeZoneStart = UndumpableLargeZoneEnd;
constexpr uintptr_t HugeZoneEnd = HugeZoneStart + HugeZoneSize;

constexpr uintptr_t SystemZoneStart = HugeZoneEnd;
constexpr uintptr_t SystemZoneEnd = SystemZoneStart + SystemZoneSize;

// We leave 64_KB at the end of 256_MB block and never use it.
// That serves two purposes:
//   1. SmallExtentSize % SmallSegmentSize == 0
//   2. Every small object satisfies RightReadableArea requirement.
constexpr size_t SmallExtentAllocSize = 256_MB;
constexpr size_t SmallExtentSize = SmallExtentAllocSize - 64_KB;
constexpr size_t SmallSegmentSize = 96_KB; // LCM(SmallRankToSize)

constexpr ui16 SmallRankBatchSize[SmallRankCount] = {
    0, 256, 256, 256, 256, 256, 256, 256, 256, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3
};

constexpr bool CheckSmallSizes()
{
    for (size_t rank = 0; rank < SmallRankCount; rank++) {
        auto size = SmallRankToSize[rank];
        if (size == 0) {
            continue;
        }

        if (SmallSegmentSize % size != 0) {
            return false;
        }

        if (SmallRankBatchSize[rank] > MaxCachedChunksPerRank) {
            return false;
        }
    }

    return true;
}

static_assert(CheckSmallSizes());
static_assert(SmallExtentSize % SmallSegmentSize == 0);
static_assert(SmallSegmentSize % PageSize == 0);

constexpr size_t LargeExtentSize = 1_GB;
static_assert(LargeExtentSize >= LargeAllocationSizeThreshold, "LargeExtentSize < LargeAllocationSizeThreshold");

constexpr const char* BackgroundThreadName = "YTAllocBack";
constexpr const char* StockpileThreadName = "YTAllocStock";

DEFINE_ENUM(EAllocationKind,
    (Untagged)
    (Tagged)
);

// Forward declarations.
struct TThreadState;
struct TLargeArena;
struct TLargeBlobExtent;

////////////////////////////////////////////////////////////////////////////////
// Traps and assertions

[[noreturn]]
void OomTrap()
{
    _exit(9);
}

[[noreturn]]
void AssertTrap(const char* message, const char* file, int line)
{
    ::fprintf(stderr, "*** YTAlloc has detected an internal trap at %s:%d\n*** %s\n",
        file,
        line,
        message);
    __builtin_trap();
}

template <class T, class E>
void AssertBlobState(T* header, E expectedState)
{
    auto actualState = header->State;
    if (Y_UNLIKELY(actualState != expectedState)) {
        char message[256];
        snprintf(message, sizeof(message), "Invalid blob header state at %p: expected %" PRIx64 ", actual %" PRIx64,
            header,
            static_cast<ui64>(expectedState),
            static_cast<ui64>(actualState));
        YTALLOC_TRAP(message);
    }
}

////////////////////////////////////////////////////////////////////////////////

// Provides a never-dying singleton with explicit construction.
template <class T>
class TExplicitlyConstructableSingleton
{
public:
    TExplicitlyConstructableSingleton()
    { }

    ~TExplicitlyConstructableSingleton()
    { }

    template <class... Ts>
    void Construct(Ts&&... args)
    {
        new (&Storage_) T(std::forward<Ts>(args)...);
#ifndef NDEBUG
        Constructed_ = true;
#endif
    }

    Y_FORCE_INLINE T* Get()
    {
#ifndef NDEBUG
        YTALLOC_PARANOID_ASSERT(Constructed_);
#endif
        return &Storage_;
    }

    Y_FORCE_INLINE const T* Get() const
    {
#ifndef NDEBUG
        YTALLOC_PARANOID_ASSERT(Constructed_);
#endif
        return &Storage_;
    }

    Y_FORCE_INLINE T* operator->()
    {
        return Get();
    }

    Y_FORCE_INLINE const T* operator->() const
    {
        return Get();
    }

    Y_FORCE_INLINE T& operator*()
    {
        return *Get();
    }

    Y_FORCE_INLINE const T& operator*() const
    {
        return *Get();
    }

private:
    union {
        T Storage_;
    };

#ifndef NDEBUG
    bool Constructed_;
#endif
};

////////////////////////////////////////////////////////////////////////////////

// Initializes all singletons.
// Safe to call multiple times.
// Guaranteed to not allocate.
void InitializeGlobals();

// Spawns the background thread, if it's time.
// Safe to call multiple times.
// Must be called on allocation slow path.
void StartBackgroundThread();

////////////////////////////////////////////////////////////////////////////////

class TLogManager
{
public:
    // Sets the handler to be invoked for each log event produced by YTAlloc.
    void EnableLogging(TLogHandler logHandler)
    {
        LogHandler_.store(logHandler);
    }

    // Checks (in a racy way) that logging is enabled.
    bool IsLoggingEnabled()
    {
        return LogHandler_.load() != nullptr;
    }

    // Logs the message via log handler (if any).
    template <class... Ts>
    void LogMessage(ELogEventSeverity severity, const char* format, Ts&&... args)
    {
        auto logHandler = LogHandler_.load();
        if (!logHandler) {
            return;
        }

        std::array<char, 16_KB> buffer;
        auto len = ::snprintf(buffer.data(), buffer.size(), format, std::forward<Ts>(args)...);

        TLogEvent event;
        event.Severity = severity;
        event.Message = TStringBuf(buffer.data(), len);
        logHandler(event);
    }

    // A special case of zero args.
    void LogMessage(ELogEventSeverity severity, const char* message)
    {
        LogMessage(severity, "%s", message);
    }

private:
    std::atomic<TLogHandler> LogHandler_= nullptr;

};

TExplicitlyConstructableSingleton<TLogManager> LogManager;

#define YTALLOC_LOG_EVENT(...)   LogManager->LogMessage(__VA_ARGS__)
#define YTALLOC_LOG_DEBUG(...)   YTALLOC_LOG_EVENT(ELogEventSeverity::Debug, __VA_ARGS__)
#define YTALLOC_LOG_INFO(...)    YTALLOC_LOG_EVENT(ELogEventSeverity::Info, __VA_ARGS__)
#define YTALLOC_LOG_WARNING(...) YTALLOC_LOG_EVENT(ELogEventSeverity::Warning, __VA_ARGS__)
#define YTALLOC_LOG_ERROR(...)   YTALLOC_LOG_EVENT(ELogEventSeverity::Error, __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE size_t GetUsed(ssize_t allocated, ssize_t freed)
{
    return allocated >= freed ? static_cast<size_t>(allocated - freed) : 0;
}

template <class T>
Y_FORCE_INLINE void* HeaderToPtr(T* header)
{
    return header + 1;
}

template <class T>
Y_FORCE_INLINE T* PtrToHeader(void* ptr)
{
    return static_cast<T*>(ptr) - 1;
}

template <class T>
Y_FORCE_INLINE const T* PtrToHeader(const void* ptr)
{
    return static_cast<const T*>(ptr) - 1;
}

Y_FORCE_INLINE size_t PtrToSmallRank(const void* ptr)
{
    return (reinterpret_cast<uintptr_t>(ptr) >> 40) & 0x1f;
}

Y_FORCE_INLINE char* AlignDownToSmallSegment(char* extent, char* ptr)
{
    auto offset = static_cast<uintptr_t>(ptr - extent);
    // NB: This modulo operation is always performed using multiplication.
    offset -= (offset % SmallSegmentSize);
    return extent + offset;
}

Y_FORCE_INLINE char* AlignUpToSmallSegment(char* extent, char* ptr)
{
    return AlignDownToSmallSegment(extent, ptr + SmallSegmentSize - 1);
}

template <class T>
static Y_FORCE_INLINE void UnalignPtr(void*& ptr)
{
    if (reinterpret_cast<uintptr_t>(ptr) % PageSize == 0) {
        reinterpret_cast<char*&>(ptr) -= PageSize - sizeof (T);
    }
    YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) % PageSize == sizeof (T));
}

template <class T>
static Y_FORCE_INLINE void UnalignPtr(const void*& ptr)
{
    if (reinterpret_cast<uintptr_t>(ptr) % PageSize == 0) {
        reinterpret_cast<const char*&>(ptr) -= PageSize - sizeof (T);
    }
    YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) % PageSize == sizeof (T));
}

template <class T>
Y_FORCE_INLINE size_t GetRawBlobSize(size_t size)
{
    return AlignUp(size + sizeof (T) + RightReadableAreaSize, PageSize);
}

template <class T>
Y_FORCE_INLINE size_t GetBlobAllocationSize(size_t size)
{
    size += sizeof(T);
    size += RightReadableAreaSize;
    size = AlignUp(size, PageSize);
    size -= sizeof(T);
    size -= RightReadableAreaSize;
    return size;
}

Y_FORCE_INLINE size_t GetLargeRank(size_t size)
{
    size_t rank = 64 - __builtin_clzl(size);
    if (size == (1ULL << (rank - 1))) {
        --rank;
    }
    return rank;
}

Y_FORCE_INLINE void PoisonRange(void* ptr, size_t size, ui32 magic)
{
#ifdef YTALLOC_PARANOID
    size = ::AlignUp<size_t>(size, 4);
    std::fill(static_cast<ui32*>(ptr), static_cast<ui32*>(ptr) + size / 4, magic);
#else
    Y_UNUSED(ptr);
    Y_UNUSED(size);
    Y_UNUSED(magic);
#endif
}

Y_FORCE_INLINE void PoisonFreedRange(void* ptr, size_t size)
{
    PoisonRange(ptr, size, 0xdeadbeef);
}

Y_FORCE_INLINE void PoisonUninitializedRange(void* ptr, size_t size)
{
    PoisonRange(ptr, size, 0xcafebabe);
}

// Checks that the header size is divisible by 16 (as needed due to alignment restrictions).
#define CHECK_HEADER_ALIGNMENT(T) static_assert(sizeof(T) % 16 == 0, "sizeof(" #T ") % 16 != 0");

////////////////////////////////////////////////////////////////////////////////

static_assert(sizeof(TFreeList<void>) == CacheLineSize, "sizeof(TFreeList) != CacheLineSize");

////////////////////////////////////////////////////////////////////////////////

constexpr size_t ShardCount = 16;
std::atomic<size_t> GlobalCurrentShardIndex;

// Provides a context for working with sharded data structures.
// Captures the initial shard index upon construction (indicating the shard
// where all insertions go). Maintains the current shard index (round-robin,
// indicating the shard currently used for extraction).
// Can be or be not thread-safe depending on TCounter.
template <class TCounter>
class TShardedState
{
public:
    TShardedState()
        : InitialShardIndex_(GlobalCurrentShardIndex++ % ShardCount)
        , CurrentShardIndex_(InitialShardIndex_)
    { }

    Y_FORCE_INLINE size_t GetInitialShardIndex() const
    {
        return InitialShardIndex_;
    }

    Y_FORCE_INLINE size_t GetNextShardIndex()
    {
        return ++CurrentShardIndex_ % ShardCount;
    }

private:
    const size_t InitialShardIndex_;
    TCounter CurrentShardIndex_;
};

using TLocalShardedState = TShardedState<size_t>;
using TGlobalShardedState = TShardedState<std::atomic<size_t>>;

// Implemented as a collection of free lists (each called a shard).
// One needs TShardedState to access the sharded data structure.
template <class T>
class TShardedFreeList
{
public:
    // First tries to extract an item from the initial shard;
    // if failed then proceeds to all shards in round-robin fashion.
    template <class TState>
    T* Extract(TState* state)
    {
        if (auto* item = Shards_[state->GetInitialShardIndex()].Extract()) {
            return item;
        }
        return ExtractRoundRobin(state);
    }

    // Attempts to extract an item from all shards in round-robin fashion.
    template <class TState>
    T* ExtractRoundRobin(TState* state)
    {
       for (size_t index = 0; index < ShardCount; ++index) {
            if (auto* item = Shards_[state->GetNextShardIndex()].Extract()) {
                return item;
            }
        }
        return nullptr;
    }

    // Extracts items from all shards linking them together.
    T* ExtractAll()
    {
        T* head = nullptr;
        T* tail = nullptr;
        for (auto& shard : Shards_) {
            auto* item = shard.ExtractAll();
            if (!head) {
                head = item;
            }
            if (tail) {
                YTALLOC_PARANOID_ASSERT(!tail->Next);
                tail->Next = item;
            } else {
                tail = item;
            }
            while (tail && tail->Next) {
                tail = tail->Next;
            }
        }
        return head;
    }

    template <class TState>
    void Put(TState* state, T* item)
    {
        Shards_[state->GetInitialShardIndex()].Put(item);
    }

private:
    std::array<TFreeList<T>, ShardCount> Shards_;
};

////////////////////////////////////////////////////////////////////////////////

// Holds YTAlloc control knobs.
// Thread safe.
class TConfigurationManager
{
public:
    void SetLargeUnreclaimableCoeff(double value)
    {
        LargeUnreclaimableCoeff_.store(value);
    }

    double GetLargeUnreclaimableCoeff() const
    {
        return LargeUnreclaimableCoeff_.load(std::memory_order_relaxed);
    }


    void SetMinLargeUnreclaimableBytes(size_t value)
    {
        MinLargeUnreclaimableBytes_.store(value);
    }

    void SetMaxLargeUnreclaimableBytes(size_t value)
    {
        MaxLargeUnreclaimableBytes_.store(value);
    }

    size_t GetMinLargeUnreclaimableBytes() const
    {
        return MinLargeUnreclaimableBytes_.load(std::memory_order_relaxed);
    }

    size_t GetMaxLargeUnreclaimableBytes() const
    {
        return MaxLargeUnreclaimableBytes_.load(std::memory_order_relaxed);
    }


    void SetTimingEventThreshold(TDuration value)
    {
        TimingEventThresholdNs_.store(value.MicroSeconds() * 1000);
    }

    i64 GetTimingEventThresholdNs() const
    {
        return TimingEventThresholdNs_.load(std::memory_order_relaxed);
    }


    void SetAllocationProfilingEnabled(bool value);

    bool IsAllocationProfilingEnabled() const
    {
        return AllocationProfilingEnabled_.load();
    }


    Y_FORCE_INLINE bool GetAllocationProfilingSamplingRate()
    {
        return AllocationProfilingSamplingRate_.load();
    }

    void SetAllocationProfilingSamplingRate(double rate)
    {
        if (rate < 0) {
            rate = 0;
        }
        if (rate > 1) {
            rate = 1;
        }
        i64 rateX64K = static_cast<i64>(rate * (1ULL << 16));
        AllocationProfilingSamplingRateX64K_.store(ClampVal<ui32>(rateX64K, 0, std::numeric_limits<ui16>::max() + 1));
        AllocationProfilingSamplingRate_.store(rate);
    }


    Y_FORCE_INLINE bool IsSmallArenaAllocationProfilingEnabled(size_t rank)
    {
        return SmallArenaAllocationProfilingEnabled_[rank].load(std::memory_order_relaxed);
    }

    Y_FORCE_INLINE bool IsSmallArenaAllocationProfiled(size_t rank)
    {
        return IsSmallArenaAllocationProfilingEnabled(rank) && IsAllocationSampled();
    }

    void SetSmallArenaAllocationProfilingEnabled(size_t rank, bool value)
    {
        if (rank >= SmallRankCount) {
            return;
        }
        SmallArenaAllocationProfilingEnabled_[rank].store(value);
    }


    Y_FORCE_INLINE bool IsLargeArenaAllocationProfilingEnabled(size_t rank)
    {
        return LargeArenaAllocationProfilingEnabled_[rank].load(std::memory_order_relaxed);
    }

    Y_FORCE_INLINE bool IsLargeArenaAllocationProfiled(size_t rank)
    {
        return IsLargeArenaAllocationProfilingEnabled(rank) && IsAllocationSampled();
    }

    void SetLargeArenaAllocationProfilingEnabled(size_t rank, bool value)
    {
        if (rank >= LargeRankCount) {
            return;
        }
        LargeArenaAllocationProfilingEnabled_[rank].store(value);
    }


    Y_FORCE_INLINE int GetProfilingBacktraceDepth()
    {
        return ProfilingBacktraceDepth_.load();
    }

    void SetProfilingBacktraceDepth(int depth)
    {
        if (depth < 1) {
            return;
        }
        if (depth > MaxAllocationProfilingBacktraceDepth) {
            depth = MaxAllocationProfilingBacktraceDepth;
        }
        ProfilingBacktraceDepth_.store(depth);
    }


    Y_FORCE_INLINE size_t GetMinProfilingBytesUsedToReport()
    {
        return MinProfilingBytesUsedToReport_.load();
    }

    void SetMinProfilingBytesUsedToReport(size_t size)
    {
        MinProfilingBytesUsedToReport_.store(size);
    }

    void SetEnableEagerMemoryRelease(bool value)
    {
        EnableEagerMemoryRelease_.store(value);
    }

    bool GetEnableEagerMemoryRelease()
    {
        return EnableEagerMemoryRelease_.load(std::memory_order_relaxed);
    }

    void SetEnableMadvisePopulate(bool value)
    {
        EnableMadvisePopulate_.store(value);
    }

    bool GetEnableMadvisePopulate()
    {
        return EnableMadvisePopulate_.load(std::memory_order_relaxed);
    }

    void EnableStockpile()
    {
        StockpileEnabled_.store(true);
    }

    bool IsStockpileEnabled()
    {
        return StockpileEnabled_.load();
    }

    void SetStockpileInterval(TDuration value)
    {
        StockpileInterval_.store(value);
    }

    TDuration GetStockpileInterval()
    {
        return StockpileInterval_.load();
    }

    void SetStockpileThreadCount(int count)
    {
        StockpileThreadCount_.store(count);
    }

    int GetStockpileThreadCount()
    {
        return ClampVal(StockpileThreadCount_.load(), 0, MaxStockpileThreadCount);
    }

    void SetStockpileSize(size_t value)
    {
        StockpileSize_.store(value);
    }

    size_t GetStockpileSize()
    {
        return StockpileSize_.load();
    }

private:
    std::atomic<double> LargeUnreclaimableCoeff_ = 0.05;
    std::atomic<size_t> MinLargeUnreclaimableBytes_ = 128_MB;
    std::atomic<size_t> MaxLargeUnreclaimableBytes_ = 10_GB;
    std::atomic<i64> TimingEventThresholdNs_ = 10000000; // in ns, 10 ms by default

    std::atomic<bool> AllocationProfilingEnabled_ = false;
    std::atomic<double> AllocationProfilingSamplingRate_ = 1.0;
    std::atomic<ui32> AllocationProfilingSamplingRateX64K_ = std::numeric_limits<ui32>::max();
    std::array<std::atomic<bool>, SmallRankCount> SmallArenaAllocationProfilingEnabled_ = {};
    std::array<std::atomic<bool>, LargeRankCount> LargeArenaAllocationProfilingEnabled_ = {};
    std::atomic<int> ProfilingBacktraceDepth_ = 10;
    std::atomic<size_t> MinProfilingBytesUsedToReport_ = 1_MB;

    std::atomic<bool> EnableEagerMemoryRelease_ = true;
    std::atomic<bool> EnableMadvisePopulate_ = false;

    std::atomic<bool> StockpileEnabled_ = false;
    std::atomic<TDuration> StockpileInterval_ = TDuration::MilliSeconds(10);
    static constexpr int MaxStockpileThreadCount = 8;
    std::atomic<int> StockpileThreadCount_ = 4;
    std::atomic<size_t> StockpileSize_ = 1_GB;

private:
    bool IsAllocationSampled()
    {
        Y_POD_STATIC_THREAD(ui16) Counter;
        return Counter++ < AllocationProfilingSamplingRateX64K_.load();
    }
};

TExplicitlyConstructableSingleton<TConfigurationManager> ConfigurationManager;

////////////////////////////////////////////////////////////////////////////////

template <class TEvent, class TManager>
class TEventLogManagerBase
{
public:
    void DisableForCurrentThread()
    {
        TManager::DisabledForCurrentThread_ = true;
    }

    template <class... TArgs>
    void EnqueueEvent(TArgs&&... args)
    {
        if (TManager::DisabledForCurrentThread_) {
            return;
        }

        auto timestamp = TInstant::Now();
        auto fiberId = NYTAlloc::GetCurrentFiberId();
        auto guard = Guard(EventLock_);

        auto event = TEvent(args...);
        OnEvent(event);

        if (EventCount_ >= EventBufferSize) {
            return;
        }

        auto& enqueuedEvent = Events_[EventCount_++];
        enqueuedEvent = std::move(event);
        enqueuedEvent.Timestamp = timestamp;
        enqueuedEvent.FiberId = fiberId;
    }

    void RunBackgroundTasks()
    {
        if (LogManager->IsLoggingEnabled()) {
            for (const auto& event : PullEvents()) {
                ProcessEvent(event);
            }
        }
    }

protected:
    NThreading::TForkAwareSpinLock EventLock_;

    virtual void OnEvent(const TEvent& event) = 0;

    virtual void ProcessEvent(const TEvent& event) = 0;

private:
    static constexpr size_t EventBufferSize = 1000;
    size_t EventCount_ = 0;
    std::array<TEvent, EventBufferSize> Events_;

    std::vector<TEvent> PullEvents()
    {
        std::vector<TEvent> events;
        events.reserve(EventBufferSize);

        auto guard = Guard(EventLock_);
        for (size_t index = 0; index < EventCount_; ++index) {
            events.push_back(Events_[index]);
        }
        EventCount_ = 0;
        return events;
    }
};

////////////////////////////////////////////////////////////////////////////////

struct TTimingEvent
{
    ETimingEventType Type;
    TDuration Duration;
    size_t Size;
    TInstant Timestamp;
    TFiberId FiberId;

    TTimingEvent()
    { }

    TTimingEvent(
        ETimingEventType type,
        TDuration duration,
        size_t size)
        : Type(type)
        , Duration(duration)
        , Size(size)
    { }
};

class TTimingManager
    : public TEventLogManagerBase<TTimingEvent, TTimingManager>
{
public:
    TEnumIndexedArray<ETimingEventType, TTimingEventCounters> GetTimingEventCounters()
    {
        auto guard = Guard(EventLock_);
        return EventCounters_;
    }

private:
    TEnumIndexedArray<ETimingEventType, TTimingEventCounters> EventCounters_;

    Y_POD_STATIC_THREAD(bool) DisabledForCurrentThread_;

    friend class TEventLogManagerBase<TTimingEvent, TTimingManager>;

    virtual void OnEvent(const TTimingEvent& event) override
    {
        auto& counters = EventCounters_[event.Type];
        counters.Count += 1;
        counters.Size += event.Size;
    }

    virtual void ProcessEvent(const TTimingEvent& event) override
    {
        YTALLOC_LOG_DEBUG("Timing event logged (Type: %s, Duration: %s, Size: %zu, Timestamp: %s, FiberId: %" PRIu64 ")",
            ToString(event.Type).c_str(),
            ToString(event.Duration).c_str(),
            event.Size,
            ToString(event.Timestamp).c_str(),
            event.FiberId);
    }
};

Y_POD_THREAD(bool) TTimingManager::DisabledForCurrentThread_;

TExplicitlyConstructableSingleton<TTimingManager> TimingManager;

////////////////////////////////////////////////////////////////////////////////

i64 GetElapsedNs(const struct timespec& startTime, const struct timespec& endTime)
{
    if (Y_LIKELY(startTime.tv_sec == endTime.tv_sec)) {
        return static_cast<i64>(endTime.tv_nsec) - static_cast<i64>(startTime.tv_nsec);
    }

    return
        static_cast<i64>(endTime.tv_nsec) - static_cast<i64>(startTime.tv_nsec) +
        (static_cast<i64>(endTime.tv_sec) - static_cast<i64>(startTime.tv_sec)) * 1000000000;
}

// Used to log statistics about long-running syscalls and lock acquisitions.
class TTimingGuard
    : public TNonCopyable
{
public:
    explicit TTimingGuard(ETimingEventType eventType, size_t size = 0)
        : EventType_(eventType)
        , Size_(size)
    {
        ::clock_gettime(CLOCK_MONOTONIC, &StartTime_);
    }

    ~TTimingGuard()
    {
        auto elapsedNs = GetElapsedNs();
        if (elapsedNs > ConfigurationManager->GetTimingEventThresholdNs()) {
            TimingManager->EnqueueEvent(EventType_, TDuration::MicroSeconds(elapsedNs / 1000), Size_);
        }
    }

private:
    const ETimingEventType EventType_;
    const size_t Size_;
    struct timespec StartTime_;

    i64 GetElapsedNs() const
    {
        struct timespec endTime;
        ::clock_gettime(CLOCK_MONOTONIC, &endTime);
        return NYTAlloc::GetElapsedNs(StartTime_, endTime);
    }
};

template <class T>
Y_FORCE_INLINE TGuard<T> GuardWithTiming(const T& lock)
{
    TTimingGuard timingGuard(ETimingEventType::Locking);
    TGuard<T> lockingGuard(lock);
    return lockingGuard;
}

////////////////////////////////////////////////////////////////////////////////

// A wrapper for mmap, mumap, and madvise calls.
// The latter are invoked with MADV_POPULATE (if enabled) and MADV_FREE flags
// and may fail if the OS support is missing. These failures are logged (once) and
// handled as follows:
// * if MADV_POPULATE fails then we fallback to manual per-page prefault
// for all subsequent attempts;
// * if MADV_FREE fails then it (and all subsequent attempts) is replaced with MADV_DONTNEED
// (which is non-lazy and is less efficient but will somehow do).
// Also this class mlocks all VMAs on startup to prevent pagefaults in our heavy binaries
// from disturbing latency tails.
class TMappedMemoryManager
{
public:
    void* Map(uintptr_t hint, size_t size, int flags)
    {
        TTimingGuard timingGuard(ETimingEventType::Mmap, size);
        auto* result = ::mmap(
            reinterpret_cast<void*>(hint),
            size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | flags,
            -1,
            0);
        if (result == MAP_FAILED) {
            auto error = errno;
            if (error == EEXIST && (flags & MAP_FIXED_NOREPLACE)) {
                // Caller must retry with different hint address.
                return result;
            }
            YTALLOC_VERIFY(error == ENOMEM);
            ::fprintf(stderr, "*** YTAlloc has received ENOMEM error while trying to mmap %zu bytes\n",
                size);
            OomTrap();
        }
        return result;
    }

    void Unmap(void* ptr, size_t size)
    {
        TTimingGuard timingGuard(ETimingEventType::Munmap, size);
        auto result = ::munmap(ptr, size);
        YTALLOC_VERIFY(result == 0);
    }

    void DontDump(void* ptr, size_t size)
    {
        auto result = ::madvise(ptr, size, MADV_DONTDUMP);
        // Must not fail.
        YTALLOC_VERIFY(result == 0);
    }

    void PopulateFile(void* ptr, size_t size)
    {
        TTimingGuard timingGuard(ETimingEventType::FilePrefault, size);

        auto* begin = static_cast<volatile char*>(ptr);
        for (auto* current = begin; current < begin + size; current += PageSize) {
            *current;
        }
    }

    void PopulateReadOnly(void* ptr, size_t size)
    {
        if (!MadvisePopulateUnavailable_.load(std::memory_order_relaxed) &&
            ConfigurationManager->GetEnableMadvisePopulate())
        {
            if (!TryMadvisePopulate(ptr, size)) {
                MadvisePopulateUnavailable_.store(true);
            }
        }
    }

    void Populate(void* ptr, size_t size)
    {
        if (MadvisePopulateUnavailable_.load(std::memory_order_relaxed) ||
            !ConfigurationManager->GetEnableMadvisePopulate())
        {
            DoPrefault(ptr, size);
        } else if (!TryMadvisePopulate(ptr, size)) {
            MadvisePopulateUnavailable_.store(true);
            DoPrefault(ptr, size);
        }
    }

    void Release(void* ptr, size_t size)
    {
        if (CanUseMadviseFree() && !ConfigurationManager->GetEnableEagerMemoryRelease()) {
            DoMadviseFree(ptr, size);
        } else {
            DoMadviseDontNeed(ptr, size);
        }
    }

    bool Stockpile(size_t size)
    {
        if (MadviseStockpileUnavailable_.load(std::memory_order_relaxed)) {
            return false;
        }
        if (!TryMadviseStockpile(size)) {
            MadviseStockpileUnavailable_.store(true);
            return false;
        }
        return true;
    }

    void RunBackgroundTasks()
    {
        if (!LogManager->IsLoggingEnabled()) {
            return;
        }
        if (IsBuggyKernel() && !BuggyKernelLogged_) {
            YTALLOC_LOG_WARNING("Kernel is buggy; see KERNEL-118");
            BuggyKernelLogged_ = true;
        }
        if (MadviseFreeSupported_ && !MadviseFreeSupportedLogged_) {
            YTALLOC_LOG_INFO("MADV_FREE is supported");
            MadviseFreeSupportedLogged_ = true;
        }
        if (MadviseFreeNotSupported_ && !MadviseFreeNotSupportedLogged_) {
            YTALLOC_LOG_WARNING("MADV_FREE is not supported");
            MadviseFreeNotSupportedLogged_ = true;
        }
        if (MadvisePopulateUnavailable_.load() && !MadvisePopulateUnavailableLogged_) {
            YTALLOC_LOG_WARNING("MADV_POPULATE is not supported");
            MadvisePopulateUnavailableLogged_ = true;
        }
        if (MadviseStockpileUnavailable_.load() && !MadviseStockpileUnavailableLogged_) {
            YTALLOC_LOG_WARNING("MADV_STOCKPILE is not supported");
            MadviseStockpileUnavailableLogged_ = true;
        }
    }

private:
    bool BuggyKernelLogged_ = false;

    std::atomic<bool> MadviseFreeSupported_ = false;
    bool MadviseFreeSupportedLogged_ = false;

    std::atomic<bool> MadviseFreeNotSupported_ = false;
    bool MadviseFreeNotSupportedLogged_ = false;

    std::atomic<bool> MadvisePopulateUnavailable_ = false;
    bool MadvisePopulateUnavailableLogged_ = false;

    std::atomic<bool> MadviseStockpileUnavailable_ = false;
    bool MadviseStockpileUnavailableLogged_ = false;

private:
    bool TryMadvisePopulate(void* ptr, size_t size)
    {
        TTimingGuard timingGuard(ETimingEventType::MadvisePopulate, size);
        auto result = ::madvise(ptr, size, MADV_POPULATE);
        if (result != 0) {
            auto error = errno;
            YTALLOC_VERIFY(error == EINVAL || error == ENOMEM);
            if (error == ENOMEM) {
                ::fprintf(stderr, "*** YTAlloc has received ENOMEM error while trying to madvise(MADV_POPULATE) %zu bytes\n",
                    size);
                OomTrap();
            }
            return false;
        }
        return true;
    }

    void DoPrefault(void* ptr, size_t size)
    {
        TTimingGuard timingGuard(ETimingEventType::Prefault, size);
        auto* begin = static_cast<char*>(ptr);
        for (auto* current = begin; current < begin + size; current += PageSize) {
            *current = 0;
        }
    }

    bool CanUseMadviseFree()
    {
        if (MadviseFreeSupported_.load()) {
            return true;
        }
        if (MadviseFreeNotSupported_.load()) {
            return false;
        }

        if (IsBuggyKernel()) {
            MadviseFreeNotSupported_.store(true);
        } else {
            auto* ptr = Map(0, PageSize, 0);
            if (::madvise(ptr, PageSize, MADV_FREE) == 0) {
                MadviseFreeSupported_.store(true);
            } else {
                MadviseFreeNotSupported_.store(true);
            }
            Unmap(ptr, PageSize);
        }

        // Will not recurse.
        return CanUseMadviseFree();
    }

    void DoMadviseDontNeed(void* ptr, size_t size)
    {
        TTimingGuard timingGuard(ETimingEventType::MadviseDontNeed, size);
        auto result = ::madvise(ptr, size, MADV_DONTNEED);
        if (result != 0) {
            auto error = errno;
            // Failure is possible for locked pages.
            Y_ABORT_UNLESS(error == EINVAL);
        }
    }

    void DoMadviseFree(void* ptr, size_t size)
    {
        TTimingGuard timingGuard(ETimingEventType::MadviseFree, size);
        auto result = ::madvise(ptr, size, MADV_FREE);
        if (result != 0) {
            auto error = errno;
            // Failure is possible for locked pages.
            YTALLOC_VERIFY(error == EINVAL);
        }
    }

    bool TryMadviseStockpile(size_t size)
    {
        auto result = ::madvise(nullptr, size, MADV_STOCKPILE);
        if (result != 0) {
            auto error = errno;
            if (error == ENOMEM || error == EAGAIN || error == EINTR) {
                // The call is advisory, ignore ENOMEM, EAGAIN, and EINTR.
                return true;
            }
            YTALLOC_VERIFY(error == EINVAL);
            return false;
        }
        return true;
    }

    // Some kernels are known to contain bugs in MADV_FREE; see https://st.yandex-team.ru/KERNEL-118.
    bool IsBuggyKernel()
    {
#ifdef _linux_
        static const bool result = [] () {
            struct utsname buf;
            YTALLOC_VERIFY(uname(&buf) == 0);
            if (strverscmp(buf.release, "4.4.1-1") >= 0 &&
                strverscmp(buf.release, "4.4.96-44") < 0)
            {
                return true;
            }
            if (strverscmp(buf.release, "4.14.1-1") >= 0 &&
                strverscmp(buf.release, "4.14.79-33") < 0)
            {
                return true;
            }
            return false;
        }();
        return result;
#else
        return false;
#endif
    }
};

TExplicitlyConstructableSingleton<TMappedMemoryManager> MappedMemoryManager;

////////////////////////////////////////////////////////////////////////////////
// System allocator

// Each system allocation is prepended with such a header.
struct TSystemBlobHeader
{
    explicit TSystemBlobHeader(size_t size)
        : Size(size)
    { }

    size_t Size;
    char Padding[8];
};

CHECK_HEADER_ALIGNMENT(TSystemBlobHeader)

// Used for some internal allocations.
// Delgates directly to TMappedMemoryManager.
class TSystemAllocator
{
public:
    void* Allocate(size_t size);
    void Free(void* ptr);

private:
    std::atomic<uintptr_t> CurrentPtr_ = SystemZoneStart;
};

TExplicitlyConstructableSingleton<TSystemAllocator> SystemAllocator;

////////////////////////////////////////////////////////////////////////////////

// Deriving from this class makes instances bound to TSystemAllocator.
struct TSystemAllocatable
{
    void* operator new(size_t size) noexcept
    {
        return SystemAllocator->Allocate(size);
    }

    void* operator new[](size_t size) noexcept
    {
        return SystemAllocator->Allocate(size);
    }

    void operator delete(void* ptr) noexcept
    {
        SystemAllocator->Free(ptr);
    }

    void operator delete[](void* ptr) noexcept
    {
        SystemAllocator->Free(ptr);
    }
};

////////////////////////////////////////////////////////////////////////////////

// Maintains a pool of objects.
// Objects are allocated in groups each containing BatchSize instances.
// The actual allocation is carried out by TSystemAllocator.
// Memory is never actually reclaimed; freed instances are put into TFreeList.
template <class T, size_t BatchSize>
class TSystemPool
{
public:
    T* Allocate()
    {
        while (true) {
            auto* obj = FreeList_.Extract();
            if (Y_LIKELY(obj)) {
                new (obj) T();
                return obj;
            }
            AllocateMore();
        }
    }

    void Free(T* obj)
    {
        obj->T::~T();
        PoisonFreedRange(obj, sizeof(T));
        FreeList_.Put(obj);
    }

private:
    TFreeList<T> FreeList_;

private:
    void AllocateMore()
    {
        auto* objs = static_cast<T*>(SystemAllocator->Allocate(sizeof(T) * BatchSize));
        for (size_t index = 0; index < BatchSize; ++index) {
            auto* obj = objs + index;
            FreeList_.Put(obj);
        }
    }
};

// A sharded analogue TSystemPool.
template <class T, size_t BatchSize>
class TShardedSystemPool
{
public:
    template <class TState>
    T* Allocate(TState* state)
    {
        if (auto* obj = FreeLists_[state->GetInitialShardIndex()].Extract()) {
            new (obj) T();
            return obj;
        }

        while (true) {
            for (size_t index = 0; index < ShardCount; ++index) {
                if (auto* obj = FreeLists_[state->GetNextShardIndex()].Extract()) {
                    new (obj) T();
                    return obj;
                }
            }
            AllocateMore();
        }
    }

    template <class TState>
    void Free(TState* state, T* obj)
    {
        obj->T::~T();
        PoisonFreedRange(obj, sizeof(T));
        FreeLists_[state->GetInitialShardIndex()].Put(obj);
    }

private:
    std::array<TFreeList<T>, ShardCount> FreeLists_;

private:
    void AllocateMore()
    {
        auto* objs = static_cast<T*>(SystemAllocator->Allocate(sizeof(T) * BatchSize));
        for (size_t index = 0; index < BatchSize; ++index) {
            auto* obj = objs + index;
            FreeLists_[index % ShardCount].Put(obj);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

// Handles allocations inside a zone of memory given by its start and end pointers.
// Each allocation is a separate mapped region of memory.
// A special care is taken to guarantee that all allocated regions fall inside the zone.
class TZoneAllocator
{
public:
    TZoneAllocator(uintptr_t zoneStart, uintptr_t zoneEnd)
        : ZoneStart_(zoneStart)
        , ZoneEnd_(zoneEnd)
        , Current_(zoneStart)
    {
        YTALLOC_VERIFY(ZoneStart_ % PageSize == 0);
    }

    void* Allocate(size_t size, int flags)
    {
        YTALLOC_VERIFY(size % PageSize == 0);
        bool restarted = false;
        while (true) {
            auto hint = (Current_ += size) - size;
            if (reinterpret_cast<uintptr_t>(hint) + size > ZoneEnd_) {
                if (restarted) {
                    ::fprintf(stderr, "*** YTAlloc was unable to mmap %zu bytes in zone %" PRIx64 "--%" PRIx64 "\n",
                        size,
                        ZoneStart_,
                        ZoneEnd_);
                    OomTrap();
                }
                restarted = true;
                Current_ = ZoneStart_;
            } else {
                char* ptr = static_cast<char*>(MappedMemoryManager->Map(
                    hint,
                    size,
                    MAP_FIXED_NOREPLACE | flags));
                if (reinterpret_cast<uintptr_t>(ptr) == hint) {
                    return ptr;
                }
                if (ptr != MAP_FAILED) {
                    MappedMemoryManager->Unmap(ptr, size);
                }
            }
        }
    }

    void Free(void* ptr, size_t size)
    {
        MappedMemoryManager->Unmap(ptr, size);
    }

private:
    const uintptr_t ZoneStart_;
    const uintptr_t ZoneEnd_;

    std::atomic<uintptr_t> Current_;
};

////////////////////////////////////////////////////////////////////////////////

// YTAlloc supports tagged allocations.
// Since the total number of tags can be huge, a two-level scheme is employed.
// Possible tags are arranged into sets each containing TaggedCounterSetSize tags.
// There are up to MaxTaggedCounterSets in total.
// Upper 4 sets are reserved for profiled allocations.
constexpr size_t TaggedCounterSetSize = 16384;
constexpr size_t AllocationProfilingTaggedCounterSets = 4;
constexpr size_t MaxTaggedCounterSets = 256 + AllocationProfilingTaggedCounterSets;

constexpr size_t MaxCapturedAllocationBacktraces = 65000;
static_assert(
    MaxCapturedAllocationBacktraces < AllocationProfilingTaggedCounterSets * TaggedCounterSetSize,
    "MaxCapturedAllocationBacktraces is too big");

constexpr TMemoryTag AllocationProfilingMemoryTagBase = TaggedCounterSetSize * (MaxTaggedCounterSets - AllocationProfilingTaggedCounterSets);
constexpr TMemoryTag AllocationProfilingUnknownMemoryTag = AllocationProfilingMemoryTagBase + MaxCapturedAllocationBacktraces;

static_assert(
    MaxMemoryTag == TaggedCounterSetSize * (MaxTaggedCounterSets - AllocationProfilingTaggedCounterSets) - 1,
    "Wrong MaxMemoryTag");

template <class TCounter>
using TUntaggedTotalCounters = TEnumIndexedArray<EBasicCounter, TCounter>;

template <class TCounter>
struct TTaggedTotalCounterSet
    : public TSystemAllocatable
{
    std::array<TEnumIndexedArray<EBasicCounter, TCounter>, TaggedCounterSetSize> Counters;
};

using TLocalTaggedBasicCounterSet = TTaggedTotalCounterSet<ssize_t>;
using TGlobalTaggedBasicCounterSet = TTaggedTotalCounterSet<std::atomic<ssize_t>>;

template <class TCounter>
struct TTotalCounters
{
    // The sum of counters across all tags.
    TUntaggedTotalCounters<TCounter> CumulativeTaggedCounters;

    // Counters for untagged allocations.
    TUntaggedTotalCounters<TCounter> UntaggedCounters;

    // Access to tagged counters may involve creation of a new tag set.
    // For simplicity, we separate the read-side (TaggedCounterSets) and the write-side (TaggedCounterSetHolders).
    // These arrays contain virtually identical data (up to std::unique_ptr and std::atomic semantic differences).
    std::array<std::atomic<TTaggedTotalCounterSet<TCounter>*>, MaxTaggedCounterSets> TaggedCounterSets{};
    std::array<std::unique_ptr<TTaggedTotalCounterSet<TCounter>>, MaxTaggedCounterSets> TaggedCounterSetHolders;

    // Protects TaggedCounterSetHolders from concurrent updates.
    NThreading::TForkAwareSpinLock TaggedCounterSetsLock;

    // Returns null if the set is not yet constructed.
    Y_FORCE_INLINE TTaggedTotalCounterSet<TCounter>* FindTaggedCounterSet(size_t index) const
    {
        return TaggedCounterSets[index].load();
    }

    // Constructs the set on first access.
    TTaggedTotalCounterSet<TCounter>* GetOrCreateTaggedCounterSet(size_t index)
    {
        auto* set = TaggedCounterSets[index].load();
        if (Y_LIKELY(set)) {
            return set;
        }

        auto guard = GuardWithTiming(TaggedCounterSetsLock);
        auto& setHolder = TaggedCounterSetHolders[index];
        if (!setHolder) {
            setHolder = std::make_unique<TTaggedTotalCounterSet<TCounter>>();
            TaggedCounterSets[index] = setHolder.get();
        }
        return setHolder.get();
    }
};

using TLocalSystemCounters = TEnumIndexedArray<ESystemCounter, ssize_t>;
using TGlobalSystemCounters = TEnumIndexedArray<ESystemCounter, std::atomic<ssize_t>>;

using TLocalSmallCounters = TEnumIndexedArray<ESmallArenaCounter, ssize_t>;
using TGlobalSmallCounters = TEnumIndexedArray<ESmallArenaCounter, std::atomic<ssize_t>>;

using TLocalLargeCounters = TEnumIndexedArray<ELargeArenaCounter, ssize_t>;
using TGlobalLargeCounters = TEnumIndexedArray<ELargeArenaCounter, std::atomic<ssize_t>>;

using TLocalHugeCounters = TEnumIndexedArray<EHugeCounter, ssize_t>;
using TGlobalHugeCounters = TEnumIndexedArray<EHugeCounter, std::atomic<ssize_t>>;

using TLocalUndumpableCounters = TEnumIndexedArray<EUndumpableCounter, ssize_t>;
using TGlobalUndumpableCounters = TEnumIndexedArray<EUndumpableCounter, std::atomic<ssize_t>>;

Y_FORCE_INLINE ssize_t LoadCounter(ssize_t counter)
{
    return counter;
}

Y_FORCE_INLINE ssize_t LoadCounter(const std::atomic<ssize_t>& counter)
{
    return counter.load();
}

////////////////////////////////////////////////////////////////////////////////

struct TMmapObservationEvent
{
    size_t Size;
    std::array<void*, MaxAllocationProfilingBacktraceDepth> Frames;
    int FrameCount;
    TInstant Timestamp;
    TFiberId FiberId;

    TMmapObservationEvent() = default;

    TMmapObservationEvent(
        size_t size,
        std::array<void*, MaxAllocationProfilingBacktraceDepth> frames,
        int frameCount)
        : Size(size)
        , Frames(frames)
        , FrameCount(frameCount)
    { }
};

class TMmapObservationManager
    : public TEventLogManagerBase<TMmapObservationEvent, TMmapObservationManager>
{
public:
    void SetBacktraceFormatter(TBacktraceFormatter formatter)
    {
        BacktraceFormatter_.store(formatter);
    }

private:
    std::atomic<TBacktraceFormatter> BacktraceFormatter_ = nullptr;

    Y_POD_STATIC_THREAD(bool) DisabledForCurrentThread_;

    friend class TEventLogManagerBase<TMmapObservationEvent, TMmapObservationManager>;

    virtual void OnEvent(const TMmapObservationEvent& /*event*/) override
    { }

    virtual void ProcessEvent(const TMmapObservationEvent& event) override
    {
        YTALLOC_LOG_DEBUG("Large arena mmap observed (Size: %zu, Timestamp: %s, FiberId: %" PRIx64 ")",
            event.Size,
            ToString(event.Timestamp).c_str(),
            event.FiberId);

        if (auto backtraceFormatter = BacktraceFormatter_.load()) {
            auto backtrace = backtraceFormatter(const_cast<void**>(event.Frames.data()), event.FrameCount);
            YTALLOC_LOG_DEBUG("YTAlloc stack backtrace (Stack: %s)",
                backtrace.c_str());
        }
    }
};

Y_POD_THREAD(bool) TMmapObservationManager::DisabledForCurrentThread_;

TExplicitlyConstructableSingleton<TMmapObservationManager> MmapObservationManager;

////////////////////////////////////////////////////////////////////////////////

// A per-thread structure containing counters, chunk caches etc.
struct TThreadState
    : public TFreeListItemBase<TThreadState>
    , public TLocalShardedState
{
    // TThreadState instances of all alive threads are put into a double-linked intrusive list.
    // This is a pair of next/prev pointers connecting an instance of TThreadState to its neighbors.
    TIntrusiveLinkedListNode<TThreadState> RegistryNode;

    // Pointers to the respective parts of TThreadManager::ThreadControlWord_.
    // If null then the thread is already destroyed (but TThreadState may still live for a while
    // due to ref-counting).
    ui8* AllocationProfilingEnabled;
    ui8* BackgroundThreadStarted;

    // TThreadStates are ref-counted.
    // TThreadManager::EnumerateThreadStates enumerates the registered states and acquires
    // a temporary reference preventing these states from being destructed. This provides
    // for shorter periods of time the global lock needs to be held.
    int RefCounter = 1;

    // Per-thread counters.
    TTotalCounters<ssize_t> TotalCounters;
    std::array<TLocalLargeCounters, LargeRankCount> LargeArenaCounters;
    TLocalUndumpableCounters UndumpableCounters;

    // Each thread maintains caches of small chunks.
    // One cache is for tagged chunks; the other is for untagged ones.
    // Each cache contains up to MaxCachedChunksPerRank chunks per any rank.
    // Special sentinels are placed to distinguish the boundaries of region containing
    // pointers of a specific rank. This enables a tiny-bit faster inplace boundary checks.

    static constexpr uintptr_t LeftSentinel = 1;
    static constexpr uintptr_t RightSentinel = 2;

    struct TSmallBlobCache
    {
        TSmallBlobCache()
        {
            void** chunkPtrs = CachedChunks.data();
            for (size_t rank = 0; rank < SmallRankCount; ++rank) {
                RankToCachedChunkPtrHead[rank] = chunkPtrs;
                chunkPtrs[0] = reinterpret_cast<void*>(LeftSentinel);
                chunkPtrs[MaxCachedChunksPerRank + 1] = reinterpret_cast<void*>(RightSentinel);

#ifdef YTALLOC_PARANOID
                RankToCachedChunkPtrTail[rank] = chunkPtrs;
                CachedChunkFull[rank] = false;

                RankToCachedChunkLeftBorder[rank] = chunkPtrs;
                RankToCachedChunkRightBorder[rank] = chunkPtrs + MaxCachedChunksPerRank + 1;
#endif
                chunkPtrs += MaxCachedChunksPerRank + 2;
            }
        }

        // For each rank we have a segment of pointers in CachedChunks with the following layout:
        //   LCC[C]........R
        // Legend:
        //   .  = garbage
        //   L  = left sentinel
        //   R  = right sentinel
        //   C  = cached pointer
        //  [C] = current cached pointer
        //
        // Under YTALLOC_PARANOID the following layout is used:
        //   L.[T]CCC[H]...R
        // Legend:
        //   [H] = head cached pointer, put chunks here
        //   [T] = tail cached pointer, take chunks from here

        //  +2 is for two sentinels
        std::array<void*, SmallRankCount * (MaxCachedChunksPerRank + 2)> CachedChunks{};

        // Pointer to [P] for each rank.
        std::array<void**, SmallRankCount> RankToCachedChunkPtrHead{};

#ifdef YTALLOC_PARANOID
        // Pointers to [L] and [R] for each rank.
        std::array<void**, SmallRankCount> RankToCachedChunkLeftBorder{};
        std::array<void**, SmallRankCount> RankToCachedChunkRightBorder{};

        std::array<void**, SmallRankCount> RankToCachedChunkPtrTail{};
        std::array<bool, SmallRankCount> CachedChunkFull{};
#endif
    };
    TEnumIndexedArray<EAllocationKind, TSmallBlobCache> SmallBlobCache;
};

struct TThreadStateToRegistryNode
{
    auto operator() (TThreadState* state) const
    {
        return &state->RegistryNode;
    }
};

// Manages all registered threads and controls access to TThreadState.
class TThreadManager
{
public:
    TThreadManager()
    {
        pthread_key_create(&ThreadDtorKey_, DestroyThread);

        NThreading::RegisterAtForkHandlers(
            nullptr,
            nullptr,
            [=] { AfterFork(); });
    }

    // Returns TThreadState for the current thread; the caller guarantees that this
    // state is initialized and is not destroyed yet.
    static TThreadState* GetThreadStateUnchecked();

    // Returns TThreadState for the current thread; may return null.
    static TThreadState* FindThreadState();

    // Returns TThreadState for the current thread; may not return null
    // (but may crash if TThreadState is already destroyed).
    static TThreadState* GetThreadStateChecked()
    {
        auto* state = FindThreadState();
        YTALLOC_VERIFY(state);
        return state;
    }

    // Enumerates all threads and invokes func passing TThreadState instances.
    // func must not throw but can take arbitrary time; no locks are being held while it executes.
    template <class THandler>
    void EnumerateThreadStatesAsync(const THandler& handler) noexcept
    {
        TMemoryTagGuard guard(NullMemoryTag);

        std::vector<TThreadState*> states;
        states.reserve(1024); // must be enough in most cases

        auto unrefStates = [&] {
            // Releasing references also requires global lock to be held to avoid getting zombies above.
            auto guard = GuardWithTiming(ThreadRegistryLock_);
            for (auto* state : states) {
                UnrefThreadState(state);
            }
        };

        auto tryRefStates = [&] {
            // Only hold this guard for a small period of time to reference all the states.
            auto guard = GuardWithTiming(ThreadRegistryLock_);
            auto* current = ThreadRegistry_.GetFront();
            while (current) {
                if (states.size() == states.capacity()) {
                    // Cannot allocate while holding ThreadRegistryLock_ due to a possible deadlock as follows:
                    // EnumerateThreadStatesAsync -> StartBackgroundThread -> EnumerateThreadStatesSync
                    // (many other scenarios are also possible).
                    guard.Release();
                    unrefStates();
                    states.clear();
                    states.reserve(states.capacity() * 2);
                    return false;
                }
                RefThreadState(current);
                states.push_back(current);
                current = current->RegistryNode.Next;
            }
            return true;
        };

        while (!tryRefStates()) ;

        for (auto* state : states) {
            handler(state);
        }

        unrefStates();
    }

    // Similar to EnumerateThreadStatesAsync but holds the global lock while enumerating the threads.
    // Also invokes a given prologue functor while holding the thread registry lock.
    // Handler and prologue calls must be fast and must not allocate.
    template <class TPrologue, class THandler>
    void EnumerateThreadStatesSync(const TPrologue& prologue, const THandler& handler) noexcept
    {
        auto guard = GuardWithTiming(ThreadRegistryLock_);
        prologue();
        auto* current = ThreadRegistry_.GetFront();
        while (current) {
            handler(current);
            current = current->RegistryNode.Next;
        }
    }


    // We store a special 64-bit "thread control word" in TLS encapsulating the following
    // crucial per-thread parameters:
    // * the current memory tag
    // * a flag indicating that a valid TThreadState is known to exists
    // (and can be obtained via GetThreadStateUnchecked)
    // * a flag indicating that allocation profiling is enabled
    // * a flag indicating that background thread is started
    // Thread control word is fetched via GetThreadControlWord and is compared
    // against FastPathControlWord to see if the fast path can be taken.
    // The latter happens when no memory tagging is configured, TThreadState is
    // valid, allocation profiling is disabled, and background thread is started.

    // The mask for extracting memory tag from thread control word.
    static constexpr ui64 MemoryTagControlWordMask = 0xffffffff;
    // ThreadStateValid is on.
    static constexpr ui64 ThreadStateValidControlWordMask = (1ULL << 32);
    // AllocationProfiling is on.
    static constexpr ui64 AllocationProfilingEnabledControlWordMask = (1ULL << 40);
    // All background thread are properly started.
    static constexpr ui64 BackgroundThreadStartedControlWorkMask = (1ULL << 48);
    // Memory tag is NullMemoryTag; thread state is valid.
    static constexpr ui64 FastPathControlWord =
        BackgroundThreadStartedControlWorkMask |
        ThreadStateValidControlWordMask |
        NullMemoryTag;

    Y_FORCE_INLINE static ui64 GetThreadControlWord()
    {
        return (&ThreadControlWord_)->Value;
    }


    static TMemoryTag GetCurrentMemoryTag()
    {
        return (&ThreadControlWord_)->Parts.MemoryTag;
    }

    static void SetCurrentMemoryTag(TMemoryTag tag)
    {
        Y_ABORT_UNLESS(tag <= MaxMemoryTag);
        (&ThreadControlWord_)->Parts.MemoryTag = tag;
    }


    static EMemoryZone GetCurrentMemoryZone()
    {
        return CurrentMemoryZone_;
    }

    static void SetCurrentMemoryZone(EMemoryZone zone)
    {
        CurrentMemoryZone_ = zone;
    }


    static void SetCurrentFiberId(TFiberId id)
    {
        CurrentFiberId_ = id;
    }

    static TFiberId GetCurrentFiberId()
    {
        return CurrentFiberId_;
    }

private:
    static void DestroyThread(void*);

    TThreadState* AllocateThreadState();

    void RefThreadState(TThreadState* state)
    {
        auto result = ++state->RefCounter;
        Y_ABORT_UNLESS(result > 1);
    }

    void UnrefThreadState(TThreadState* state)
    {
        auto result = --state->RefCounter;
        Y_ABORT_UNLESS(result >= 0);
        if (result == 0) {
            DestroyThreadState(state);
        }
    }

    void DestroyThreadState(TThreadState* state);

    void AfterFork();

private:
    // TThreadState instance for the current thread.
    // Initially null, then initialized when first needed.
    // TThreadState is destroyed upon thread termination (which is detected with
    // the help of pthread_key_create machinery), so this pointer can become null again.
    Y_POD_STATIC_THREAD(TThreadState*) ThreadState_;

    // Initially false, then set to true then TThreadState is destroyed.
    // If the thread requests for its state afterwards, null is returned and no new state is (re-)created.
    // The caller must be able to deal with it.
    Y_POD_STATIC_THREAD(bool) ThreadStateDestroyed_;

    union TThreadControlWord
    {
        ui64 __attribute__((__may_alias__)) Value;
        struct TParts
        {
            // The current memory tag used in all allocations by this thread.
            ui32 __attribute__((__may_alias__)) MemoryTag;
            // Indicates if a valid TThreadState exists and can be obtained via GetThreadStateUnchecked.
            ui8 __attribute__((__may_alias__)) ThreadStateValid;
            // Indicates if allocation profiling is on.
            ui8 __attribute__((__may_alias__)) AllocationProfilingEnabled;
            // Indicates if all background threads are properly started.
            ui8 __attribute__((__may_alias__)) BackgroundThreadStarted;
            ui8 Padding[2];
        } Parts;
    };
    Y_POD_STATIC_THREAD(TThreadControlWord) ThreadControlWord_;

    // See memory zone API.
    Y_POD_STATIC_THREAD(EMemoryZone) CurrentMemoryZone_;

    // See fiber id API.
    Y_POD_STATIC_THREAD(TFiberId) CurrentFiberId_;

    pthread_key_t ThreadDtorKey_;

    static constexpr size_t ThreadStatesBatchSize = 1;
    TSystemPool<TThreadState, ThreadStatesBatchSize> ThreadStatePool_;

    NThreading::TForkAwareSpinLock ThreadRegistryLock_;
    TIntrusiveLinkedList<TThreadState, TThreadStateToRegistryNode> ThreadRegistry_;
};

Y_POD_THREAD(TThreadState*) TThreadManager::ThreadState_;
Y_POD_THREAD(bool) TThreadManager::ThreadStateDestroyed_;
Y_POD_THREAD(TThreadManager::TThreadControlWord) TThreadManager::ThreadControlWord_;
Y_POD_THREAD(EMemoryZone) TThreadManager::CurrentMemoryZone_;
Y_POD_THREAD(TFiberId) TThreadManager::CurrentFiberId_;

TExplicitlyConstructableSingleton<TThreadManager> ThreadManager;

////////////////////////////////////////////////////////////////////////////////

void TConfigurationManager::SetAllocationProfilingEnabled(bool value)
{
    // Update threads' TLS.
    ThreadManager->EnumerateThreadStatesSync(
        [&] {
            AllocationProfilingEnabled_.store(value);
        },
        [&] (auto* state) {
            if (state->AllocationProfilingEnabled) {
                *state->AllocationProfilingEnabled = value;
            }
        });
}

////////////////////////////////////////////////////////////////////////////////
// Backtrace Manager
//
// Captures backtraces observed during allocations and assigns memory tags to them.
// Memory tags are chosen sequentially starting from AllocationProfilingMemoryTagBase.
//
// For each backtrace we compute a 64-bit hash and use it as a key in a certain concurrent hashmap.
// This hashmap is organized into BucketCount buckets, each consisting of BucketSize slots.
//
// Backtrace hash is translated into bucket index by taking the appropriate number of
// its lower bits. For each slot, we remember a 32-bit fingerprint, which is
// just the next 32 bits of the backtrace's hash, and the (previously assigned) memory tag.
//
// Upon access to the hashtable, the bucket is first scanned optimistically, without taking
// any locks. In case of a miss, a per-bucket spinlock is acquired and the bucket is rescanned.
//
// The above scheme may involve collisions but we neglect their probability.
//
// If the whole hash table overflows (i.e. a total of MaxCapturedAllocationBacktraces
// backtraces are captured) or the bucket overflows (i.e. all of its slots become occupied),
// the allocation is annotated with AllocationProfilingUnknownMemoryTag. Such allocations
// appear as having no backtrace whatsoever in the profiling reports.

class TBacktraceManager
{
public:
    // Sets the provider used for collecting backtraces when allocation profiling
    // is turned ON.
    void SetBacktraceProvider(TBacktraceProvider provider)
    {
        BacktraceProvider_.store(provider);
    }

    // Captures the backtrace and inserts it into the hashtable.
    TMemoryTag GetMemoryTagFromBacktrace(int framesToSkip)
    {
        std::array<void*, MaxAllocationProfilingBacktraceDepth> frames;
        auto backtraceProvider = BacktraceProvider_.load();
        if (!backtraceProvider) {
            return NullMemoryTag;
        }
        auto frameCount  = backtraceProvider(frames.data(), ConfigurationManager->GetProfilingBacktraceDepth(), framesToSkip);
        auto hash = GetBacktraceHash(frames.data(), frameCount);
        return CaptureBacktrace(hash, frames.data(), frameCount);
    }

    // Returns the backtrace corresponding to the given tag, if any.
    std::optional<TBacktrace> FindBacktrace(TMemoryTag tag)
    {
        if (tag < AllocationProfilingMemoryTagBase ||
            tag >= AllocationProfilingMemoryTagBase + MaxCapturedAllocationBacktraces)
        {
            return std::nullopt;
        }
        const auto& entry = Backtraces_[tag - AllocationProfilingMemoryTagBase];
        if (!entry.Captured.load()) {
            return std::nullopt;
        }
        return entry.Backtrace;
    }

private:
    static constexpr int Log2BucketCount = 16;
    static constexpr int BucketCount = 1 << Log2BucketCount;
    static constexpr int BucketSize = 8;

    std::atomic<TBacktraceProvider> BacktraceProvider_ = nullptr;

    std::array<std::array<std::atomic<ui32>, BucketSize>, BucketCount> Fingerprints_= {};
    std::array<std::array<std::atomic<TMemoryTag>, BucketSize>, BucketCount> MemoryTags_ = {};
    std::array<NThreading::TForkAwareSpinLock, BucketCount> BucketLocks_;
    std::atomic<TMemoryTag> CurrentMemoryTag_ = AllocationProfilingMemoryTagBase;

    struct TBacktraceEntry
    {
        TBacktrace Backtrace;
        std::atomic<bool> Captured = false;
    };

    std::array<TBacktraceEntry, MaxCapturedAllocationBacktraces> Backtraces_;

private:
    static size_t GetBacktraceHash(void** frames, int frameCount)
    {
        size_t hash = 0;
        for (int index = 0; index < frameCount; ++index) {
            hash = CombineHashes(hash, THash<void*>()(frames[index]));
        }
        return hash;
    }

    TMemoryTag CaptureBacktrace(size_t hash, void** frames, int frameCount)
    {
        size_t bucketIndex = hash % BucketCount;
        ui32 fingerprint = (hash >> Log2BucketCount) & 0xffffffff;
        // Zero fingerprint indicates the slot is free; check and adjust to ensure
        // that regular fingerprints are non-zero.
        if (fingerprint == 0) {
            fingerprint = 1;
        }

        for (int slotIndex = 0; slotIndex < BucketSize; ++slotIndex) {
            auto currentFingerprint = Fingerprints_[bucketIndex][slotIndex].load(std::memory_order_relaxed);
            if (currentFingerprint == fingerprint) {
                return MemoryTags_[bucketIndex][slotIndex].load();
            }
        }

        auto guard = Guard(BucketLocks_[bucketIndex]);

        int spareSlotIndex = -1;
        for (int slotIndex = 0; slotIndex < BucketSize; ++slotIndex) {
            auto currentFingerprint = Fingerprints_[bucketIndex][slotIndex].load(std::memory_order_relaxed);
            if (currentFingerprint == fingerprint) {
                return MemoryTags_[bucketIndex][slotIndex];
            }
            if (currentFingerprint == 0) {
                spareSlotIndex = slotIndex;
                break;
            }
        }

        if (spareSlotIndex < 0) {
            return AllocationProfilingUnknownMemoryTag;
        }

        auto memoryTag = CurrentMemoryTag_++;
        if (memoryTag >= AllocationProfilingMemoryTagBase + MaxCapturedAllocationBacktraces) {
            return AllocationProfilingUnknownMemoryTag;
        }

        MemoryTags_[bucketIndex][spareSlotIndex].store(memoryTag);
        Fingerprints_[bucketIndex][spareSlotIndex].store(fingerprint);

        auto& entry = Backtraces_[memoryTag - AllocationProfilingMemoryTagBase];
        entry.Backtrace.FrameCount = frameCount;
        ::memcpy(entry.Backtrace.Frames.data(), frames, sizeof (void*) * frameCount);
        entry.Captured.store(true);

        return memoryTag;
    }
};

TExplicitlyConstructableSingleton<TBacktraceManager> BacktraceManager;

////////////////////////////////////////////////////////////////////////////////

// Mimics the counters of TThreadState but uses std::atomic to survive concurrent access.
struct TGlobalState
    : public TGlobalShardedState
{
    TTotalCounters<std::atomic<ssize_t>> TotalCounters;
    std::array<TGlobalLargeCounters, LargeRankCount> LargeArenaCounters;
    TGlobalUndumpableCounters UndumpableCounters;
};

TExplicitlyConstructableSingleton<TGlobalState> GlobalState;

////////////////////////////////////////////////////////////////////////////////

// Accumulates various allocation statistics.
class TStatisticsManager
{
public:
    template <EAllocationKind Kind = EAllocationKind::Tagged, class TState>
    static Y_FORCE_INLINE void IncrementTotalCounter(TState* state, TMemoryTag tag, EBasicCounter counter, ssize_t delta)
    {
        // This branch is typically resolved at compile time.
        if (Kind == EAllocationKind::Tagged && tag != NullMemoryTag) {
            IncrementTaggedTotalCounter(&state->TotalCounters, tag, counter, delta);
        } else {
            IncrementUntaggedTotalCounter(&state->TotalCounters, counter, delta);
        }
    }

    static Y_FORCE_INLINE void IncrementTotalCounter(TMemoryTag tag, EBasicCounter counter, ssize_t delta)
    {
        IncrementTotalCounter(GlobalState.Get(), tag, counter, delta);
    }

    void IncrementSmallArenaCounter(ESmallArenaCounter counter, size_t rank, ssize_t delta)
    {
        SmallArenaCounters_[rank][counter] += delta;
    }

    template <class TState>
    static Y_FORCE_INLINE void IncrementLargeArenaCounter(TState* state, size_t rank, ELargeArenaCounter counter, ssize_t delta)
    {
        state->LargeArenaCounters[rank][counter] += delta;
    }

    template <class TState>
    static Y_FORCE_INLINE void IncrementUndumpableCounter(TState* state, EUndumpableCounter counter, ssize_t delta)
    {
        state->UndumpableCounters[counter] += delta;
    }

    void IncrementHugeCounter(EHugeCounter counter, ssize_t delta)
    {
        HugeCounters_[counter] += delta;
    }

    void IncrementHugeUndumpableCounter(EUndumpableCounter counter, ssize_t delta)
    {
        HugeUndumpableCounters_[counter] += delta;
    }

    void IncrementSystemCounter(ESystemCounter counter, ssize_t delta)
    {
        SystemCounters_[counter] += delta;
    }

    // Computes memory usage for a list of tags by aggregating counters across threads.
    void GetTaggedMemoryCounters(const TMemoryTag* tags, size_t count, TEnumIndexedArray<EBasicCounter, ssize_t>* counters)
    {
        TMemoryTagGuard guard(NullMemoryTag);

        for (size_t index = 0; index < count; ++index) {
            counters[index][EBasicCounter::BytesAllocated] = 0;
            counters[index][EBasicCounter::BytesFreed] = 0;
        }

        for (size_t index = 0; index < count; ++index) {
            auto tag = tags[index];
            counters[index][EBasicCounter::BytesAllocated] += LoadTaggedTotalCounter(GlobalState->TotalCounters, tag, EBasicCounter::BytesAllocated);
            counters[index][EBasicCounter::BytesFreed] += LoadTaggedTotalCounter(GlobalState->TotalCounters, tag, EBasicCounter::BytesFreed);
        }

        ThreadManager->EnumerateThreadStatesAsync(
            [&] (const auto* state) {
                for (size_t index = 0; index < count; ++index) {
                    auto tag = tags[index];
                    counters[index][EBasicCounter::BytesAllocated] += LoadTaggedTotalCounter(state->TotalCounters, tag, EBasicCounter::BytesAllocated);
                    counters[index][EBasicCounter::BytesFreed] += LoadTaggedTotalCounter(state->TotalCounters, tag, EBasicCounter::BytesFreed);
                }
            });

        for (size_t index = 0; index < count; ++index) {
            counters[index][EBasicCounter::BytesUsed] = GetUsed(counters[index][EBasicCounter::BytesAllocated], counters[index][EBasicCounter::BytesFreed]);
        }
    }

    void GetTaggedMemoryUsage(const TMemoryTag* tags, size_t count, size_t* results)
    {
        TMemoryTagGuard guard(NullMemoryTag);

        std::vector<TEnumIndexedArray<EBasicCounter, ssize_t>> counters;
        counters.resize(count);
        GetTaggedMemoryCounters(tags, count, counters.data());

        for (size_t index = 0; index < count; ++index) {
            results[index] = counters[index][EBasicCounter::BytesUsed];
        }
    }

    TEnumIndexedArray<ETotalCounter, ssize_t> GetTotalAllocationCounters()
    {
        TEnumIndexedArray<ETotalCounter, ssize_t> result;

        auto accumulate = [&] (const auto& counters) {
            result[ETotalCounter::BytesAllocated] += LoadCounter(counters[EBasicCounter::BytesAllocated]);
            result[ETotalCounter::BytesFreed] += LoadCounter(counters[EBasicCounter::BytesFreed]);
        };

        accumulate(GlobalState->TotalCounters.UntaggedCounters);
        accumulate(GlobalState->TotalCounters.CumulativeTaggedCounters);

        ThreadManager->EnumerateThreadStatesAsync(
            [&] (const auto* state) {
                accumulate(state->TotalCounters.UntaggedCounters);
                accumulate(state->TotalCounters.CumulativeTaggedCounters);
            });

        result[ETotalCounter::BytesUsed] = GetUsed(
            result[ETotalCounter::BytesAllocated],
            result[ETotalCounter::BytesFreed]);

        auto systemCounters = GetSystemAllocationCounters();
        result[ETotalCounter::BytesCommitted] += systemCounters[EBasicCounter::BytesUsed];

        auto hugeCounters = GetHugeAllocationCounters();
        result[ETotalCounter::BytesCommitted] += hugeCounters[EHugeCounter::BytesUsed];

        auto smallArenaCounters = GetSmallArenaAllocationCounters();
        for (size_t rank = 0; rank < SmallRankCount; ++rank) {
            result[ETotalCounter::BytesCommitted] += smallArenaCounters[rank][ESmallArenaCounter::BytesCommitted];
        }

        auto largeArenaCounters = GetLargeArenaAllocationCounters();
        for (size_t rank = 0; rank < LargeRankCount; ++rank) {
            result[ETotalCounter::BytesCommitted] += largeArenaCounters[rank][ELargeArenaCounter::BytesCommitted];
        }

        result[ETotalCounter::BytesUnaccounted] = std::max<ssize_t>(GetProcessRss() - result[ETotalCounter::BytesCommitted], 0);

        return result;
    }

    TEnumIndexedArray<ESmallCounter, ssize_t> GetSmallAllocationCounters()
    {
        TEnumIndexedArray<ESmallCounter, ssize_t> result;

        auto totalCounters = GetTotalAllocationCounters();
        result[ESmallCounter::BytesAllocated] = totalCounters[ETotalCounter::BytesAllocated];
        result[ESmallCounter::BytesFreed] = totalCounters[ETotalCounter::BytesFreed];
        result[ESmallCounter::BytesUsed] = totalCounters[ETotalCounter::BytesUsed];

        auto largeArenaCounters = GetLargeArenaAllocationCounters();
        for (size_t rank = 0; rank < LargeRankCount; ++rank) {
            result[ESmallCounter::BytesAllocated] -= largeArenaCounters[rank][ELargeArenaCounter::BytesAllocated];
            result[ESmallCounter::BytesFreed] -= largeArenaCounters[rank][ELargeArenaCounter::BytesFreed];
            result[ESmallCounter::BytesUsed] -= largeArenaCounters[rank][ELargeArenaCounter::BytesUsed];
        }

        auto hugeCounters = GetHugeAllocationCounters();
        result[ESmallCounter::BytesAllocated] -= hugeCounters[EHugeCounter::BytesAllocated];
        result[ESmallCounter::BytesFreed] -= hugeCounters[EHugeCounter::BytesFreed];
        result[ESmallCounter::BytesUsed] -= hugeCounters[EHugeCounter::BytesUsed];

        return result;
    }

    std::array<TLocalSmallCounters, SmallRankCount> GetSmallArenaAllocationCounters()
    {
        std::array<TLocalSmallCounters, SmallRankCount> result;
        for (size_t rank = 0; rank < SmallRankCount; ++rank) {
            for (auto counter : TEnumTraits<ESmallArenaCounter>::GetDomainValues()) {
                result[rank][counter] = SmallArenaCounters_[rank][counter].load();
            }
        }
        return result;
    }

    TEnumIndexedArray<ELargeCounter, ssize_t> GetLargeAllocationCounters()
    {
        TEnumIndexedArray<ELargeCounter, ssize_t> result;
        auto largeArenaCounters = GetLargeArenaAllocationCounters();
        for (size_t rank = 0; rank < LargeRankCount; ++rank) {
            result[ESmallCounter::BytesAllocated] += largeArenaCounters[rank][ELargeArenaCounter::BytesAllocated];
            result[ESmallCounter::BytesFreed] += largeArenaCounters[rank][ELargeArenaCounter::BytesFreed];
            result[ESmallCounter::BytesUsed] += largeArenaCounters[rank][ELargeArenaCounter::BytesUsed];
        }
        return result;
    }

    std::array<TLocalLargeCounters, LargeRankCount> GetLargeArenaAllocationCounters()
    {
        std::array<TLocalLargeCounters, LargeRankCount> result{};

        for (size_t rank = 0; rank < LargeRankCount; ++rank) {
            for (auto counter : TEnumTraits<ELargeArenaCounter>::GetDomainValues()) {
                result[rank][counter] = GlobalState->LargeArenaCounters[rank][counter].load();
            }
        }

        ThreadManager->EnumerateThreadStatesAsync(
            [&] (const auto* state) {
                for (size_t rank = 0; rank < LargeRankCount; ++rank) {
                    for (auto counter : TEnumTraits<ELargeArenaCounter>::GetDomainValues()) {
                        result[rank][counter] += state->LargeArenaCounters[rank][counter];
                    }
                }
            });

        for (size_t rank = 0; rank < LargeRankCount; ++rank) {
            result[rank][ELargeArenaCounter::BytesUsed] = GetUsed(result[rank][ELargeArenaCounter::BytesAllocated], result[rank][ELargeArenaCounter::BytesFreed]);
            result[rank][ELargeArenaCounter::BlobsUsed] = GetUsed(result[rank][ELargeArenaCounter::BlobsAllocated], result[rank][ELargeArenaCounter::BlobsFreed]);
        }

        return result;
    }

    TLocalSystemCounters GetSystemAllocationCounters()
    {
        TLocalSystemCounters result;
        for (auto counter : TEnumTraits<ESystemCounter>::GetDomainValues()) {
            result[counter] = SystemCounters_[counter].load();
        }
        result[ESystemCounter::BytesUsed] = GetUsed(result[ESystemCounter::BytesAllocated], result[ESystemCounter::BytesFreed]);
        return result;
    }

    TLocalHugeCounters GetHugeAllocationCounters()
    {
        TLocalHugeCounters result;
        for (auto counter : TEnumTraits<EHugeCounter>::GetDomainValues()) {
            result[counter] = HugeCounters_[counter].load();
        }
        result[EHugeCounter::BytesUsed] = GetUsed(result[EHugeCounter::BytesAllocated], result[EHugeCounter::BytesFreed]);
        result[EHugeCounter::BlobsUsed] = GetUsed(result[EHugeCounter::BlobsAllocated], result[EHugeCounter::BlobsFreed]);
        return result;
    }

    TLocalUndumpableCounters GetUndumpableAllocationCounters()
    {
        TLocalUndumpableCounters result;
        for (auto counter : TEnumTraits<EUndumpableCounter>::GetDomainValues()) {
            result[counter] = HugeUndumpableCounters_[counter].load();
            result[counter] += GlobalState->UndumpableCounters[counter].load();
        }

        ThreadManager->EnumerateThreadStatesAsync(
            [&] (const auto* state) {
                result[EUndumpableCounter::BytesAllocated] += LoadCounter(state->UndumpableCounters[EUndumpableCounter::BytesAllocated]);
                result[EUndumpableCounter::BytesFreed] += LoadCounter(state->UndumpableCounters[EUndumpableCounter::BytesFreed]);
            });

        result[EUndumpableCounter::BytesUsed] = GetUsed(result[EUndumpableCounter::BytesAllocated], result[EUndumpableCounter::BytesFreed]);
        return result;
    }

    // Called before TThreadState is destroyed.
    // Adds the counter values from TThreadState to the global counters.
    void AccumulateLocalCounters(TThreadState* state)
    {
        for (auto counter : TEnumTraits<EBasicCounter>::GetDomainValues()) {
            GlobalState->TotalCounters.CumulativeTaggedCounters[counter] += state->TotalCounters.CumulativeTaggedCounters[counter];
            GlobalState->TotalCounters.UntaggedCounters[counter] += state->TotalCounters.UntaggedCounters[counter];
        }
        for (size_t index = 0; index < MaxTaggedCounterSets; ++index) {
            const auto* localSet = state->TotalCounters.FindTaggedCounterSet(index);
            if (!localSet) {
                continue;
            }
            auto* globalSet = GlobalState->TotalCounters.GetOrCreateTaggedCounterSet(index);
            for (size_t jndex = 0; jndex < TaggedCounterSetSize; ++jndex) {
                for (auto counter : TEnumTraits<EBasicCounter>::GetDomainValues()) {
                    globalSet->Counters[jndex][counter] += localSet->Counters[jndex][counter];
                }
            }
        }
        for (size_t rank = 0; rank < LargeRankCount; ++rank) {
            for (auto counter : TEnumTraits<ELargeArenaCounter>::GetDomainValues()) {
                GlobalState->LargeArenaCounters[rank][counter] += state->LargeArenaCounters[rank][counter];
            }
        }
        for (auto counter : TEnumTraits<EUndumpableCounter>::GetDomainValues()) {
            GlobalState->UndumpableCounters[counter] += state->UndumpableCounters[counter];
        }
    }

private:
    template <class TCounter>
    static ssize_t LoadTaggedTotalCounter(const TTotalCounters<TCounter>& counters, TMemoryTag tag, EBasicCounter counter)
    {
        const auto* set = counters.FindTaggedCounterSet(tag / TaggedCounterSetSize);
        if (Y_UNLIKELY(!set)) {
            return 0;
        }
        return LoadCounter(set->Counters[tag % TaggedCounterSetSize][counter]);
    }

    template <class TCounter>
    static Y_FORCE_INLINE void IncrementUntaggedTotalCounter(TTotalCounters<TCounter>* counters, EBasicCounter counter, ssize_t delta)
    {
        counters->UntaggedCounters[counter] += delta;
    }

    template <class TCounter>
    static Y_FORCE_INLINE void IncrementTaggedTotalCounter(TTotalCounters<TCounter>* counters, TMemoryTag tag, EBasicCounter counter, ssize_t delta)
    {
        counters->CumulativeTaggedCounters[counter] += delta;
        auto* set = counters->GetOrCreateTaggedCounterSet(tag / TaggedCounterSetSize);
        set->Counters[tag % TaggedCounterSetSize][counter] += delta;
    }


    static ssize_t GetProcessRss()
    {
        auto* file = ::fopen("/proc/self/statm", "r");
        if (!file) {
            return 0;
        }

        ssize_t dummy;
        ssize_t rssPages;
        auto readResult = fscanf(file, "%zd %zd", &dummy, &rssPages);

        ::fclose(file);

        if (readResult != 2) {
            return 0;
        }

        return rssPages * PageSize;
    }

private:
    TGlobalSystemCounters SystemCounters_;
    std::array<TGlobalSmallCounters, SmallRankCount> SmallArenaCounters_;
    TGlobalHugeCounters HugeCounters_;
    TGlobalUndumpableCounters HugeUndumpableCounters_;
};

TExplicitlyConstructableSingleton<TStatisticsManager> StatisticsManager;

////////////////////////////////////////////////////////////////////////////////

void* TSystemAllocator::Allocate(size_t size)
{
    auto rawSize = GetRawBlobSize<TSystemBlobHeader>(size);
    void* mmappedPtr;
    while (true) {
        auto currentPtr = CurrentPtr_.fetch_add(rawSize);
        Y_ABORT_UNLESS(currentPtr + rawSize <= SystemZoneEnd);
        mmappedPtr = MappedMemoryManager->Map(
            currentPtr,
            rawSize,
            MAP_FIXED_NOREPLACE | MAP_POPULATE);
        if (mmappedPtr == reinterpret_cast<void*>(currentPtr)) {
            break;
        }
        if (mmappedPtr != MAP_FAILED) {
            MappedMemoryManager->Unmap(mmappedPtr, rawSize);
        }
    }
    auto* blob = static_cast<TSystemBlobHeader*>(mmappedPtr);
    new (blob) TSystemBlobHeader(size);
    auto* result = HeaderToPtr(blob);
    PoisonUninitializedRange(result, size);
    StatisticsManager->IncrementSystemCounter(ESystemCounter::BytesAllocated, rawSize);
    return result;
}

void TSystemAllocator::Free(void* ptr)
{
    auto* blob = PtrToHeader<TSystemBlobHeader>(ptr);
    auto rawSize = GetRawBlobSize<TSystemBlobHeader>(blob->Size);
    MappedMemoryManager->Unmap(blob, rawSize);
    StatisticsManager->IncrementSystemCounter(ESystemCounter::BytesFreed, rawSize);
}

////////////////////////////////////////////////////////////////////////////////
// Small allocator
//
// Allocations (called small chunks) are grouped by their sizes. Two most-significant binary digits are
// used to determine the rank of a chunk, which guarantees 25% overhead in the worst case.
// A pair of helper arrays (SizeToSmallRank1 and SizeToSmallRank2) are used to compute ranks; we expect
// them to be permanently cached.
//
// Chunks of the same rank are served by a (small) arena allocator.
// In fact, there are two arenas for each rank: one is for tagged allocations and another is for untagged ones.
//
// We encode chunk's rank and whether it is tagged or not in the resulting pointer as follows:
//   0- 3:  must be zero due to alignment
//   4-39:  varies
//  40-44:  rank
//     45:  0 for untagged allocations, 1 for tagged ones
//  45-63:  zeroes
// This enables computing chunk's rank and also determining if it is tagged in constant time
// without any additional lookups. Also, one pays no space overhead for untagged allocations
// and pays 16 bytes for each tagged one.
//
// Each arena allocates extents of memory by calling mmap for each extent of SmallExtentSize bytes.
// (Recall that this memory is never reclaimed.)
// Each extent is then sliced into segments of SmallSegmentSize bytes.
// Whenever a new segment is acquired, its memory is pre-faulted by madvise(MADV_POPULATE).
// New segments are acquired in a lock-free manner.
//
// Each thread maintains a separate cache of chunks of each rank (two caches to be precise: one
// for tagged allocations and the other for untagged). These caches are fully thread-local and
// involve no atomic operations.
//
// There are also global caches (per rank, for tagged and untagged allocations).
// Instead of keeping individual chunks these work with chunk groups (collections of up to ChunksPerGroup
// arbitrary chunks).
//
// When the local cache becomes exhausted, a group of chunks is fetched from the global cache
// (if the latter is empty then the arena allocator is consulted).
// Vice versa, if the local cache overflows, a group of chunks is moved from it to the global cache.
//
// Global caches and arena allocators also take care of (rare) cases when Allocate/Free is called
// without a valid thread state (which happens during thread shutdown when TThreadState is already destroyed).
//
// Each arena allocates memory in a certain "data" zone of SmallZoneSize.
// In addition to that zone, up to two "shadow" zones are maintained.
//
// The first one contains memory tags of chunks residing in the primary zone.
// The second one (which is present if YTALLOC_NERVOUS is defined) contains
// states of chunks. These states enable some simple internal sanity checks
// (e.g. detect attempts to double-free a chunk).
//
// Addresses in the data zone are directly mapped to offsets in shadow zones.
// When a segment of a small arena zone is allocated, the relevant portions of shadow
// zones get initialized (and also accounted for as a system allocation).
//
// Shadow zones are memory-mapped with MAP_NORESERVE flag and are quite sparse.
// These zones are omitted from core dumps due to their huge size and sparsity.

// For each small rank i, gives max K such that 2^k <= SmallRankToSize[i].
// Chunk pointer is mapped to its shadow image via GetShadowOffset helper.
// Note that chunk size is not always a power of 2. To avoid costly integer division,
// chunk pointer is translated by means of bitwise shift only (leaving some bytes
// of shadow zones unused). This array provides the needed shifts.
constexpr int SmallRankToLogSize[SmallRankCount] = {
    0,
    4, 5, 5, 6, 6, 7,
    7, 8, 8, 9, 9, 10, 10, 11,
    11, 12, 12, 13, 13, 14, 14, 15
};

enum class ESmallChunkState : ui8
{
    Spare         = 0,
    Allocated     = 0x61, // a
    Freed         = 0x66  // f
};

class TSmallArenaAllocator
{
public:
    TSmallArenaAllocator(EAllocationKind kind, size_t rank, uintptr_t dataZoneStart)
        : Kind_(kind)
        , Rank_(rank)
        , LogSize_(SmallRankToLogSize[Rank_])
        , ChunkSize_(SmallRankToSize[Rank_])
        , DataZoneStart_(dataZoneStart)
        , DataZoneAllocator_(DataZoneStart_, DataZoneStart_ + SmallZoneSize)
    { }

    size_t PullMany(void** batch, size_t maxCount)
    {
        size_t count;
        while (true) {
            count = TryAllocateFromCurrentExtent(batch, maxCount);
            if (Y_LIKELY(count != 0)) {
                break;
            }
            PopulateAnotherExtent();
        }
        return count;
    }

    void* Allocate(size_t size)
    {
        void* ptr;
        auto count = PullMany(&ptr, 1);
        YTALLOC_PARANOID_ASSERT(count == 1);
        YTALLOC_PARANOID_ASSERT(PtrToSmallRank(ptr) == Rank_);
        PoisonUninitializedRange(ptr, size);
        UpdateChunkState(ptr, ESmallChunkState::Freed, ESmallChunkState::Allocated);
        return ptr;
    }

    TMemoryTag GetAndResetMemoryTag(const void* ptr)
    {
        auto& tag = MemoryTagZoneStart_[GetShadowOffset(ptr)];
        auto currentTag = tag;
        tag = NullMemoryTag;
        return currentTag;
    }

    void SetMemoryTag(void* ptr, TMemoryTag tag)
    {
        MemoryTagZoneStart_[GetShadowOffset(ptr)] = tag;
    }

    void UpdateChunkState(const void* ptr, ESmallChunkState expectedState, ESmallChunkState newState)
    {
#ifdef YTALLOC_NERVOUS
        auto& state = ChunkStateZoneStart_[GetShadowOffset(ptr)];
        auto actualState = state;
        if (Y_UNLIKELY(actualState != expectedState)) {
            char message[256];
            snprintf(message, sizeof(message), "Invalid small chunk state at %p: expected %" PRIx8 ", actual %" PRIx8,
                ptr,
                static_cast<ui8>(expectedState),
                static_cast<ui8>(actualState));
            YTALLOC_TRAP(message);
        }
        state = newState;
#else
        Y_UNUSED(ptr);
        Y_UNUSED(expectedState);
        Y_UNUSED(newState);
#endif
    }

private:
    size_t TryAllocateFromCurrentExtent(void** batch, size_t maxCount)
    {
        auto* oldPtr = CurrentPtr_.load();
        if (Y_UNLIKELY(!oldPtr)) {
            return 0;
        }

        auto* currentExtent = CurrentExtent_.load(std::memory_order_relaxed);
        if (Y_UNLIKELY(!currentExtent)) {
            return 0;
        }

        char* newPtr;
        while (true) {
            if (Y_UNLIKELY(oldPtr < currentExtent || oldPtr + ChunkSize_ + RightReadableAreaSize > currentExtent + SmallExtentSize)) {
                return 0;
            }

            newPtr = std::min(
                oldPtr + ChunkSize_ * maxCount,
                currentExtent + SmallExtentSize);

            auto* alignedNewPtr = AlignDownToSmallSegment(currentExtent, newPtr);
            if (alignedNewPtr > oldPtr) {
                newPtr = alignedNewPtr;
            }

            if (Y_LIKELY(CurrentPtr_.compare_exchange_weak(oldPtr, newPtr))) {
                break;
            }
        }

        auto* firstSegment = AlignUpToSmallSegment(currentExtent, oldPtr);
        auto* nextSegment = AlignUpToSmallSegment(currentExtent, newPtr);
        if (firstSegment != nextSegment) {
            auto size = nextSegment - firstSegment;
            MappedMemoryManager->PopulateReadOnly(firstSegment, size);

            StatisticsManager->IncrementSmallArenaCounter(ESmallArenaCounter::BytesCommitted, Rank_, size);
            StatisticsManager->IncrementSmallArenaCounter(ESmallArenaCounter::PagesCommitted, Rank_, size / PageSize);
            if (Kind_ == EAllocationKind::Tagged) {
                StatisticsManager->IncrementSystemCounter(ESystemCounter::BytesAllocated, size / ChunkSize_ * sizeof(TMemoryTag));
            }
#ifdef YTALLOC_NERVOUS
            StatisticsManager->IncrementSystemCounter(ESystemCounter::BytesAllocated, size / ChunkSize_ * sizeof(ESmallChunkState));
#endif
        }

        size_t count = 0;
        while (oldPtr != newPtr) {
            UpdateChunkState(oldPtr, ESmallChunkState::Spare, ESmallChunkState::Freed);

            batch[count] = oldPtr;

            oldPtr += ChunkSize_;
            count++;
        }
        return count;
    }

    void PopulateAnotherExtent()
    {
        auto lockGuard = GuardWithTiming(ExtentLock_);

        auto* currentPtr = CurrentPtr_.load();
        auto* currentExtent = CurrentExtent_.load();

        if (currentPtr && currentPtr + ChunkSize_ + RightReadableAreaSize <= currentExtent + SmallExtentSize) {
            // No need for a new extent.
            return;
        }

        auto* newExtent = static_cast<char*>(DataZoneAllocator_.Allocate(SmallExtentAllocSize, 0));

        AllocateShadowZones();

        YTALLOC_VERIFY(reinterpret_cast<uintptr_t>(newExtent) % SmallExtentAllocSize == 0);
        CurrentPtr_ = CurrentExtent_ = newExtent;

        StatisticsManager->IncrementSmallArenaCounter(ESmallArenaCounter::BytesMapped, Rank_, SmallExtentAllocSize);
        StatisticsManager->IncrementSmallArenaCounter(ESmallArenaCounter::PagesMapped, Rank_, SmallExtentAllocSize / PageSize);
    }

private:
    const EAllocationKind Kind_;
    const size_t Rank_;
    const size_t LogSize_;
    const size_t ChunkSize_;
    const uintptr_t DataZoneStart_;

    TZoneAllocator DataZoneAllocator_;

    bool ShadowZonesAllocated_ = false;
    TMemoryTag* MemoryTagZoneStart_;
#ifdef YTALLOC_NERVOUS
    ESmallChunkState* ChunkStateZoneStart_;
#endif

    NThreading::TForkAwareSpinLock ExtentLock_;
    std::atomic<char*> CurrentPtr_ = nullptr;
    std::atomic<char*> CurrentExtent_ = nullptr;

    size_t GetShadowOffset(const void* ptr)
    {
        return (reinterpret_cast<uintptr_t>(ptr) - DataZoneStart_) >> LogSize_;
    }

    void AllocateShadowZones()
    {
        if (ShadowZonesAllocated_) {
            return;
        }

        if (Kind_ == EAllocationKind::Tagged) {
            MemoryTagZoneStart_ = MapShadowZone<TMemoryTag>();
        }
#ifdef YTALLOC_NERVOUS
        ChunkStateZoneStart_ = MapShadowZone<ESmallChunkState>();
#endif

        ShadowZonesAllocated_ = true;
    }

    template <class T>
    T* MapShadowZone()
    {
        auto size = AlignUp((SmallZoneSize >> LogSize_) * sizeof (T), PageSize);
        auto* ptr = static_cast<T*>(MappedMemoryManager->Map(SystemZoneStart, size, MAP_NORESERVE));
        MappedMemoryManager->DontDump(ptr, size);
        return ptr;
    }
};

TExplicitlyConstructableSingleton<TEnumIndexedArray<EAllocationKind, std::array<TExplicitlyConstructableSingleton<TSmallArenaAllocator>, SmallRankCount>>> SmallArenaAllocators;

////////////////////////////////////////////////////////////////////////////////

constexpr size_t ChunksPerGroup = 128;
constexpr size_t GroupsBatchSize = 1024;

static_assert(ChunksPerGroup <= MaxCachedChunksPerRank, "ChunksPerGroup > MaxCachedChunksPerRank");

class TChunkGroup
    : public TFreeListItemBase<TChunkGroup>
{
public:
    bool IsEmpty() const
    {
        return Size_ == 0;
    }

    size_t ExtractAll(void** ptrs)
    {
        auto count = Size_;
        ::memcpy(ptrs, Ptrs_.data(), count * sizeof(void*));
        Size_ = 0;
        return count;
    }

    void PutOne(void* ptr)
    {
        PutMany(&ptr, 1);
    }

    void PutMany(void** ptrs, size_t count)
    {
        YTALLOC_PARANOID_ASSERT(Size_ == 0);
        YTALLOC_PARANOID_ASSERT(count <= ChunksPerGroup);
        ::memcpy(Ptrs_.data(), ptrs, count * sizeof(void*));
        Size_ = count;
    }

private:
    size_t Size_ = 0; // <= ChunksPerGroup
    std::array<void*, ChunksPerGroup> Ptrs_;
};

class TGlobalSmallChunkCache
{
public:
    explicit TGlobalSmallChunkCache(EAllocationKind kind)
        : Kind_(kind)
    { }

#ifdef YTALLOC_PARANOID
    void CanonizeChunkPtrs(TThreadState* state, size_t rank)
    {
        auto& chunkPtrPtr = state->SmallBlobCache[Kind_].RankToCachedChunkPtrHead[rank];

        auto leftBorder = state->SmallBlobCache[Kind_].RankToCachedChunkLeftBorder[rank];
        auto rightBorder = state->SmallBlobCache[Kind_].RankToCachedChunkRightBorder[rank];

        state->SmallBlobCache[Kind_].CachedChunkFull[rank] = false;
        if (chunkPtrPtr + 1 == rightBorder) {
            chunkPtrPtr = leftBorder;
            state->SmallBlobCache[Kind_].CachedChunkFull[rank] = true;
        }

        state->SmallBlobCache[Kind_].RankToCachedChunkPtrTail[rank] = leftBorder;
    }
#endif

    bool TryMoveGroupToLocal(TThreadState* state, size_t rank)
    {
        auto& groups = RankToChunkGroups_[rank];
        auto* group = groups.Extract(state);
        if (!Y_LIKELY(group)) {
            return false;
        }

        YTALLOC_PARANOID_ASSERT(!group->IsEmpty());

        auto& chunkPtrPtr = state->SmallBlobCache[Kind_].RankToCachedChunkPtrHead[rank];
#ifdef YTALLOC_PARANOID
        chunkPtrPtr = state->SmallBlobCache[Kind_].RankToCachedChunkLeftBorder[rank];
        state->SmallBlobCache[Kind_].RankToCachedChunkPtrTail[rank] = chunkPtrPtr;
#endif
        auto chunkCount = group->ExtractAll(chunkPtrPtr + 1);
        chunkPtrPtr += chunkCount;

#ifdef YTALLOC_PARANOID
        CanonizeChunkPtrs(state, rank);
#endif
        GroupPool_.Free(state, group);
        return true;
    }

    void MoveGroupToGlobal(TThreadState* state, size_t rank)
    {
        auto* group = GroupPool_.Allocate(state);

        auto& chunkPtrPtr = state->SmallBlobCache[Kind_].RankToCachedChunkPtrHead[rank];
        YTALLOC_PARANOID_ASSERT(*(chunkPtrPtr + 1) == reinterpret_cast<void*>(TThreadState::RightSentinel));
        group->PutMany(chunkPtrPtr - ChunksPerGroup + 1, ChunksPerGroup);
        chunkPtrPtr -= ChunksPerGroup;
#ifdef YTALLOC_PARANOID
        ::memset(chunkPtrPtr + 1, 0, sizeof(void*) * ChunksPerGroup);
        CanonizeChunkPtrs(state, rank);
#endif

        auto& groups = RankToChunkGroups_[rank];
        YTALLOC_PARANOID_ASSERT(!group->IsEmpty());
        groups.Put(state, group);
    }

    void MoveOneToGlobal(void* ptr, size_t rank)
    {
        auto* group = GroupPool_.Allocate(&GlobalShardedState_);
        group->PutOne(ptr);

        auto& groups = RankToChunkGroups_[rank];
        YTALLOC_PARANOID_ASSERT(!group->IsEmpty());
        groups.Put(&GlobalShardedState_, group);
    }

#ifdef YTALLOC_PARANOID
    void MoveAllToGlobal(TThreadState* state, size_t rank)
    {
        auto leftSentinelBorder = state->SmallBlobCache[Kind_].RankToCachedChunkLeftBorder[rank];
        auto rightSentinelBorder = state->SmallBlobCache[Kind_].RankToCachedChunkRightBorder[rank];

        auto& headPtr = state->SmallBlobCache[Kind_].RankToCachedChunkPtrHead[rank];
        auto& tailPtr = state->SmallBlobCache[Kind_].RankToCachedChunkPtrTail[rank];

        if (tailPtr == headPtr && !state->SmallBlobCache[Kind_].CachedChunkFull[rank]) {
            headPtr = leftSentinelBorder;
            return;
        }

        // (leftBorder, rightBorder]
        auto moveIntervalToGlobal = [=] (void** leftBorder, void** rightBorder) {
            while (true) {
                size_t count = 0;
                while (count < ChunksPerGroup && rightBorder != leftBorder) {
                    --rightBorder;
                    ++count;
                }

                if (count == 0) {
                    break;
                }

                auto* group = GroupPool_.Allocate(state);
                group->PutMany(rightBorder + 1, count);
                ::memset(rightBorder + 1, 0, sizeof(void*) * count);
                auto& groups = RankToChunkGroups_[rank];
                groups.Put(state, group);
            }
        };

        if (tailPtr >= headPtr) {
            moveIntervalToGlobal(tailPtr, rightSentinelBorder - 1);
            moveIntervalToGlobal(leftSentinelBorder, headPtr);
        } else {
            moveIntervalToGlobal(tailPtr, headPtr);
        }

        headPtr = leftSentinelBorder;
    }
#else
    void MoveAllToGlobal(TThreadState* state, size_t rank)
    {
        auto& chunkPtrPtr = state->SmallBlobCache[Kind_].RankToCachedChunkPtrHead[rank];
        while (true) {
            size_t count = 0;
            while (count < ChunksPerGroup && *chunkPtrPtr != reinterpret_cast<void*>(TThreadState::LeftSentinel)) {
                --chunkPtrPtr;
                ++count;
            }

            if (count == 0) {
                break;
            }

            auto* group = GroupPool_.Allocate(state);
            group->PutMany(chunkPtrPtr + 1, count);
            auto& groups = RankToChunkGroups_[rank];
            groups.Put(state, group);
        }
    }
#endif

private:
    const EAllocationKind Kind_;

    TGlobalShardedState GlobalShardedState_;
    TShardedSystemPool<TChunkGroup, GroupsBatchSize> GroupPool_;
    std::array<TShardedFreeList<TChunkGroup>, SmallRankCount> RankToChunkGroups_;
};

TExplicitlyConstructableSingleton<TEnumIndexedArray<EAllocationKind, TExplicitlyConstructableSingleton<TGlobalSmallChunkCache>>> GlobalSmallChunkCaches;

////////////////////////////////////////////////////////////////////////////////

class TSmallAllocator
{
public:
    template <EAllocationKind Kind>
    static Y_FORCE_INLINE void* Allocate(TMemoryTag tag, size_t rank)
    {
        auto* state = TThreadManager::FindThreadState();
        if (Y_LIKELY(state)) {
            return Allocate<Kind>(tag, rank, state);
        }
        auto size = SmallRankToSize[rank];
        return AllocateGlobal<Kind>(tag, rank, size);
    }

#ifdef YTALLOC_PARANOID
    template <EAllocationKind Kind>
    static Y_FORCE_INLINE void* Allocate(TMemoryTag tag, size_t rank, TThreadState* state)
    {
        auto& localCache = state->SmallBlobCache[Kind];
        auto& allocator = *(*SmallArenaAllocators)[Kind][rank];

        size_t size = SmallRankToSize[rank];
        StatisticsManager->IncrementTotalCounter<Kind>(state, tag, EBasicCounter::BytesAllocated, size);

        auto leftBorder = localCache.RankToCachedChunkLeftBorder[rank];
        auto rightBorder = localCache.RankToCachedChunkRightBorder[rank];

        void* result;
        while (true) {
            auto& chunkHeadPtr = localCache.RankToCachedChunkPtrHead[rank];
            auto& cachedHeadPtr = *(chunkHeadPtr + 1);
            auto* headPtr = cachedHeadPtr;

            auto& chunkTailPtr = localCache.RankToCachedChunkPtrTail[rank];
            auto& cachedTailPtr = *(chunkTailPtr + 1);
            auto* tailPtr = cachedTailPtr;

            auto& chunkFull = localCache.CachedChunkFull[rank];

            if (Y_LIKELY(chunkFull || headPtr != tailPtr)) {
                YTALLOC_PARANOID_ASSERT(tailPtr);
                cachedTailPtr = nullptr;
                ++chunkTailPtr;
                if (Y_LIKELY(chunkTailPtr + 1 == rightBorder)) {
                    chunkTailPtr = leftBorder;
                }

                chunkFull = false;
                result = tailPtr;
                PoisonUninitializedRange(result, size);
                allocator.UpdateChunkState(result, ESmallChunkState::Freed, ESmallChunkState::Allocated);
                break;
            }

            auto& globalCache = *(*GlobalSmallChunkCaches)[Kind];
            if (!globalCache.TryMoveGroupToLocal(state, rank)) {
                result = allocator.Allocate(size);
                break;
            }
        }

        if constexpr(Kind == EAllocationKind::Tagged) {
            allocator.SetMemoryTag(result, tag);
        }

        return result;
    }

    template <EAllocationKind Kind>
    static Y_FORCE_INLINE void Free(void* ptr)
    {
        auto rank = PtrToSmallRank(ptr);
        auto size = SmallRankToSize[rank];

        auto& allocator = *(*SmallArenaAllocators)[Kind][rank];

        auto tag = NullMemoryTag;
        if constexpr(Kind == EAllocationKind::Tagged) {
            tag = allocator.GetAndResetMemoryTag(ptr);
            YTALLOC_PARANOID_ASSERT(tag != NullMemoryTag);
        }

        allocator.UpdateChunkState(ptr, ESmallChunkState::Allocated, ESmallChunkState::Freed);
        PoisonFreedRange(ptr, size);

        auto* state = TThreadManager::FindThreadState();
        if (Y_UNLIKELY(!state)) {
            FreeGlobal<Kind>(tag, ptr, rank, size);
            return;
        }

        StatisticsManager->IncrementTotalCounter<Kind>(state, tag, EBasicCounter::BytesFreed, size);

        auto& localCache = state->SmallBlobCache[Kind];

        auto leftBorder = localCache.RankToCachedChunkLeftBorder[rank];
        auto rightBorder = localCache.RankToCachedChunkRightBorder[rank];

        while (true) {
            auto& chunkHeadPtr = localCache.RankToCachedChunkPtrHead[rank];
            auto& headPtr = *(chunkHeadPtr + 1);

            auto& chunkTailPtr = localCache.RankToCachedChunkPtrTail[rank];
            auto& chunkFull = localCache.CachedChunkFull[rank];

            if (Y_LIKELY(!chunkFull)) {
                headPtr = ptr;
                ++chunkHeadPtr;
                if (Y_LIKELY(chunkHeadPtr + 1 == rightBorder)) {
                    chunkHeadPtr = leftBorder;
                }
                chunkFull = (chunkHeadPtr == chunkTailPtr);
                break;
            }

            chunkHeadPtr = rightBorder - 1;
            chunkTailPtr = leftBorder;

            auto& globalCache = *(*GlobalSmallChunkCaches)[Kind];
            globalCache.MoveGroupToGlobal(state, rank);
        }
    }

#else

    template <EAllocationKind Kind>
    static Y_FORCE_INLINE void* Allocate(TMemoryTag tag, size_t rank, TThreadState* state)
    {
        size_t size = SmallRankToSize[rank];
        StatisticsManager->IncrementTotalCounter<Kind>(state, tag, EBasicCounter::BytesAllocated, size);

        auto& localCache = state->SmallBlobCache[Kind];
        auto& allocator = *(*SmallArenaAllocators)[Kind][rank];

        void* result;
        while (true) {
            auto& chunkPtr = localCache.RankToCachedChunkPtrHead[rank];
            auto& cachedPtr = *chunkPtr;
            auto* ptr = cachedPtr;
            if (Y_LIKELY(ptr != reinterpret_cast<void*>(TThreadState::LeftSentinel))) {
                --chunkPtr;
                result = ptr;
                allocator.UpdateChunkState(result, ESmallChunkState::Freed, ESmallChunkState::Allocated);
                PoisonUninitializedRange(result, size);
                break;
            }

            auto& globalCache = *(*GlobalSmallChunkCaches)[Kind];
            if (globalCache.TryMoveGroupToLocal(state, rank)) {
                continue;
            }

            auto count = allocator.PullMany(
                chunkPtr + 1,
                SmallRankBatchSize[rank]);
            chunkPtr += count;
        }

        if constexpr(Kind == EAllocationKind::Tagged) {
            allocator.SetMemoryTag(result, tag);
        }

        return result;
    }

    template <EAllocationKind Kind>
    static Y_FORCE_INLINE void Free(void* ptr)
    {
        auto rank = PtrToSmallRank(ptr);
        auto size = SmallRankToSize[rank];

        auto& allocator = *(*SmallArenaAllocators)[Kind][rank];

        auto tag = NullMemoryTag;
        if constexpr(Kind == EAllocationKind::Tagged) {
            tag = allocator.GetAndResetMemoryTag(ptr);
            YTALLOC_PARANOID_ASSERT(tag != NullMemoryTag);
        }

        allocator.UpdateChunkState(ptr, ESmallChunkState::Allocated, ESmallChunkState::Freed);
        PoisonFreedRange(ptr, size);

        auto* state = TThreadManager::FindThreadState();
        if (Y_UNLIKELY(!state)) {
            FreeGlobal<Kind>(tag, ptr, rank, size);
            return;
        }

        StatisticsManager->IncrementTotalCounter<Kind>(state, tag, EBasicCounter::BytesFreed, size);

        auto& localCache = state->SmallBlobCache[Kind];

        while (true) {
            auto& chunkPtrPtr = localCache.RankToCachedChunkPtrHead[rank];
            auto& chunkPtr = *(chunkPtrPtr + 1);
            if (Y_LIKELY(chunkPtr != reinterpret_cast<void*>(TThreadState::RightSentinel))) {
                chunkPtr = ptr;
                ++chunkPtrPtr;
                break;
            }

            auto& globalCache = *(*GlobalSmallChunkCaches)[Kind];
            globalCache.MoveGroupToGlobal(state, rank);
        }
    }
#endif

    static size_t GetAllocationSize(const void* ptr)
    {
        return SmallRankToSize[PtrToSmallRank(ptr)];
    }

    static size_t GetAllocationSize(size_t size)
    {
        return SmallRankToSize[SizeToSmallRank(size)];
    }

    static void PurgeCaches()
    {
        DoPurgeCaches<EAllocationKind::Untagged>();
        DoPurgeCaches<EAllocationKind::Tagged>();
    }

private:
    template <EAllocationKind Kind>
    static void DoPurgeCaches()
    {
        auto* state = TThreadManager::GetThreadStateChecked();
        for (size_t rank = 0; rank < SmallRankCount; ++rank) {
            (*GlobalSmallChunkCaches)[Kind]->MoveAllToGlobal(state, rank);
        }
    }

    template <EAllocationKind Kind>
    static void* AllocateGlobal(TMemoryTag tag, size_t rank, size_t size)
    {
        StatisticsManager->IncrementTotalCounter(tag, EBasicCounter::BytesAllocated, size);

        auto& allocator = *(*SmallArenaAllocators)[Kind][rank];
        auto* result = allocator.Allocate(size);

        if constexpr(Kind == EAllocationKind::Tagged) {
            allocator.SetMemoryTag(result, tag);
        }

        return result;
    }

    template <EAllocationKind Kind>
    static void FreeGlobal(TMemoryTag tag, void* ptr, size_t rank, size_t size)
    {
        StatisticsManager->IncrementTotalCounter(tag, EBasicCounter::BytesFreed, size);

        auto& globalCache = *(*GlobalSmallChunkCaches)[Kind];
        globalCache.MoveOneToGlobal(ptr, rank);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Large blob allocator
//
// Like for small chunks, large blobs are grouped into arenas, where arena K handles
// blobs of size (2^{K-1},2^K]. Memory is mapped in extents of LargeExtentSize bytes.
// Each extent is split into segments of size 2^K (here segment is just a memory region, which may fully consist of
// unmapped pages). When a segment is actually allocated, it becomes a blob and a TLargeBlobHeader
// structure is placed at its start.
//
// When an extent is allocated, it is sliced into segments (not blobs, since no headers are placed and
// no memory is touched). These segments are put into disposed segments list.
//
// For each blob two separate sizes are maintained: BytesAcquired indicates the number of bytes
// acquired via madvise(MADV_POPULATE) from the system; BytesAllocated (<= BytesAcquired) corresponds
// to the number of bytes claimed by the user (including the header and page size alignment).
// If BytesAllocated == 0 then this blob is spare, i.e.
// was freed and remains cached for further possible reuse.
//
// When a new blob is being allocated, the allocator first tries to extract a spare blob. On success,
// its acquired size is extended (if needed); the acquired size never shrinks on allocation.
// If no spare blobs exist, a disposed segment is extracted and is turned into a blob (i.e.
// its header is initialized) and the needed number of bytes is acquired. If no disposed segments
// exist, then a new extent is allocated and sliced into segments.
//
// The above algorithm only claims memory from the system (by means of madvise(MADV_POPULATE));
// the reclaim is handled by a separate background mechanism. Two types of reclaimable memory
// regions are possible:
// * spare: these correspond to spare blobs; upon reclaiming this region becomes a disposed segment
// * overhead: these correspond to trailing parts of allocated blobs in [BytesAllocated, BytesAcquired) byte range
//
// Reclaiming spare blobs is easy as these are explicitly tracked by spare blob lists. To reclaim,
// we atomically extract a blob from a spare list, call madvise(MADV_FREE), and put the pointer to
// the disposed segment list.
//
// Reclaiming overheads is more complicated since (a) allocated blobs are never tracked directly and
// (b) reclaiming them may interfere with Allocate and Free.
//
// To overcome (a), for each extent we maintain a bitmap marking segments that are actually blobs
// (i.e. contain a header). (For simplicity and efficiency this bitmap is just a vector of bytes.)
// These flags are updated in Allocate/Free with appropriate memory ordering. Note that
// blobs are only disposed (and are turned into segments) by the background thread; if this
// thread discovers a segment that is marked as a blob, then it is safe to assume that this segment
// remains a blob unless the thread disposes it.
//
// To overcome (b), each large blob header maintains a spin lock. When blob B is extracted
// from a spare list in Allocate, an acquisition is tried. If successful, B is returned to the
// user. Otherwise it is assumed that B is currently being examined by the background
// reclaimer thread. Allocate then skips this blob and retries extraction; the problem is that
// since the spare list is basically a stack one cannot just push B back into the spare list.
// Instead, B is pushed into a special locked spare list. This list is purged by the background
// thread on each tick and its items are pushed back into the usual spare list.
//
// A similar trick is used by Free: when invoked for blob B its spin lock acquisition is first
// tried. Upon success, B is moved to the spare list. On failure, Free has to postpone this deallocation
// by moving B into the freed locked list. This list, similarly, is being purged by the background thread.
//
// It remains to explain how the background thread computes the number of bytes to be reclaimed from
// each arena. To this aim, we first compute the total number of reclaimable bytes.
// This is the sum of spare and overhead bytes in all arenas minus the number of unreclaimable bytes
// The latter grows linearly in the number of used bytes and is capped from below by a MinUnreclaimableLargeBytes;
// and from above by MaxUnreclaimableLargeBytes. SetLargeUnreclaimableCoeff and Set(Min|Max)LargeUnreclaimableBytes
// enable tuning these control knobs. The reclaimable bytes are being taken from arenas starting from those
// with the largest spare and overhead volumes.
//
// The above implies that each large blob contains a fixed-size header preceeding it.
// Hence ptr % PageSize == sizeof (TLargeBlobHeader) for each ptr returned by Allocate
// (since large blob sizes are larger than PageSize and are divisible by PageSize).
// For AllocatePageAligned, however, ptr must be divisible by PageSize. To handle such an allocation, we
// artificially increase its size and align the result of Allocate up to the next page boundary.
// When handling a deallocation, ptr is moved back by UnalignPtr (which is capable of dealing
// with both the results of Allocate and AllocatePageAligned).
// This technique applies to both large and huge blobs.

enum ELargeBlobState : ui64
{
    Allocated   = 0x6c6c61656772616cULL, // largeall
    Spare       = 0x727073656772616cULL, // largespr
    LockedSpare = 0x70736c656772616cULL, // largelsp
    LockedFreed = 0x72666c656772616cULL  // largelfr
};

// Every large blob (either tagged or not) is prepended with this header.
struct TLargeBlobHeader
    : public TFreeListItemBase<TLargeBlobHeader>
{
    TLargeBlobHeader(
        TLargeBlobExtent* extent,
        size_t bytesAcquired,
        size_t bytesAllocated,
        TMemoryTag tag)
        : Extent(extent)
        , BytesAcquired(bytesAcquired)
        , Tag(tag)
        , BytesAllocated(bytesAllocated)
        , State(ELargeBlobState::Allocated)
    { }

    TLargeBlobExtent* Extent;
    // Number of bytes in all acquired pages.
    size_t BytesAcquired;
    std::atomic<bool> Locked = false;
    TMemoryTag Tag = NullMemoryTag;
    // For spare blobs this is zero.
    // For allocated blobs this is the number of bytes requested by user (not including header of any alignment).
    size_t BytesAllocated;
    ELargeBlobState State;
    char Padding[12];
};

CHECK_HEADER_ALIGNMENT(TLargeBlobHeader)

struct TLargeBlobExtent
{
    TLargeBlobExtent(size_t segmentCount, char* ptr)
        : SegmentCount(segmentCount)
        , Ptr(ptr)
    { }

    size_t SegmentCount;
    char* Ptr;
    TLargeBlobExtent* NextExtent = nullptr;

    std::atomic<bool> DisposedFlags[0];
};

// A helper node that enables storing a number of extent's segments
// in a free list. Recall that segments themselves do not posses any headers.
struct TDisposedSegment
    : public TFreeListItemBase<TDisposedSegment>
{
    size_t Index;
    TLargeBlobExtent* Extent;
};

struct TLargeArena
{
    size_t Rank = 0;
    size_t SegmentSize = 0;

    TShardedFreeList<TLargeBlobHeader> SpareBlobs;
    TFreeList<TLargeBlobHeader> LockedSpareBlobs;
    TFreeList<TLargeBlobHeader> LockedFreedBlobs;
    TFreeList<TDisposedSegment> DisposedSegments;
    std::atomic<TLargeBlobExtent*> FirstExtent = nullptr;

    TLargeBlobExtent* CurrentOverheadScanExtent = nullptr;
    size_t CurrentOverheadScanSegment = 0;
};

template <bool Dumpable>
class TLargeBlobAllocator
{
public:
    TLargeBlobAllocator()
        : ZoneAllocator_(LargeZoneStart(Dumpable), LargeZoneEnd(Dumpable))
    {
        for (size_t rank = 0; rank < Arenas_.size(); ++rank) {
            auto& arena = Arenas_[rank];
            arena.Rank = rank;
            arena.SegmentSize = (1ULL << rank);
        }
    }

    void* Allocate(size_t size)
    {
        auto* state = TThreadManager::FindThreadState();
        return Y_LIKELY(state)
            ? DoAllocate(state, size)
            : DoAllocate(GlobalState.Get(), size);
    }

    void Free(void* ptr)
    {
        auto* state = TThreadManager::FindThreadState();
        if (Y_LIKELY(state)) {
            DoFree(state, ptr);
        } else {
            DoFree(GlobalState.Get(), ptr);
        }
    }

    static size_t GetAllocationSize(const void* ptr)
    {
        UnalignPtr<TLargeBlobHeader>(ptr);
        const auto* blob = PtrToHeader<TLargeBlobHeader>(ptr);
        return blob->BytesAllocated;
    }

    static size_t GetAllocationSize(size_t size)
    {
        return GetBlobAllocationSize<TLargeBlobHeader>(size);
    }

    void RunBackgroundTasks()
    {
        ReinstallLockedBlobs();
        ReclaimMemory();
    }

    void SetBacktraceProvider(TBacktraceProvider provider)
    {
        BacktraceProvider_.store(provider);
    }

private:
    template <class TState>
    void PopulateArenaPages(TState* state, TLargeArena* arena, void* ptr, size_t size)
    {
        MappedMemoryManager->Populate(ptr, size);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::BytesPopulated, size);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::PagesPopulated, size / PageSize);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::BytesCommitted, size);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::PagesCommitted, size / PageSize);
    }

    template <class TState>
    void ReleaseArenaPages(TState* state, TLargeArena* arena, void* ptr, size_t size)
    {
        MappedMemoryManager->Release(ptr, size);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::BytesReleased, size);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::PagesReleased, size / PageSize);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::BytesCommitted, -size);
        StatisticsManager->IncrementLargeArenaCounter(state, arena->Rank, ELargeArenaCounter::PagesCommitted, -size / PageSize);
    }

    bool TryLockBlob(TLargeBlobHeader* blob)
    {
        bool expected = false;
        return blob->Locked.compare_exchange_strong(expected, true);
    }

    void UnlockBlob(TLargeBlobHeader* blob)
    {
        blob->Locked.store(false);
    }

    template <class TState>
    void MoveBlobToSpare(TState* state, TLargeArena* arena, TLargeBlobHeader* blob, bool unlock)
    {
        auto rank = arena->Rank;
        auto size = blob->BytesAllocated;
        auto rawSize = GetRawBlobSize<TLargeBlobHeader>(size);
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesSpare, blob->BytesAcquired);
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesOverhead, -(blob->BytesAcquired - rawSize));
        blob->BytesAllocated = 0;
        if (unlock) {
            UnlockBlob(blob);
        } else {
            YTALLOC_VERIFY(!blob->Locked.load());
        }
        blob->State = ELargeBlobState::Spare;
        arena->SpareBlobs.Put(state, blob);
    }

    size_t GetBytesToReclaim(const std::array<TLocalLargeCounters, LargeRankCount>& arenaCounters)
    {
        size_t totalBytesAllocated = 0;
        size_t totalBytesFreed = 0;
        size_t totalBytesSpare = 0;
        size_t totalBytesOverhead = 0;
        for (size_t rank = 0; rank < Arenas_.size(); ++rank) {
            const auto& counters = arenaCounters[rank];
            totalBytesAllocated += counters[ELargeArenaCounter::BytesAllocated];
            totalBytesFreed += counters[ELargeArenaCounter::BytesFreed];
            totalBytesSpare += counters[ELargeArenaCounter::BytesSpare];
            totalBytesOverhead += counters[ELargeArenaCounter::BytesOverhead];
        }

        auto totalBytesUsed = totalBytesAllocated - totalBytesFreed;
        auto totalBytesReclaimable = totalBytesSpare + totalBytesOverhead;

        auto threshold = ClampVal(
            static_cast<size_t>(ConfigurationManager->GetLargeUnreclaimableCoeff() * totalBytesUsed),
            ConfigurationManager->GetMinLargeUnreclaimableBytes(),
            ConfigurationManager->GetMaxLargeUnreclaimableBytes());
        if (totalBytesReclaimable < threshold) {
            return 0;
        }

        auto bytesToReclaim = totalBytesReclaimable - threshold;
        return AlignUp(bytesToReclaim, PageSize);
    }

    void ReinstallLockedSpareBlobs(TLargeArena* arena)
    {
        auto* blob = arena->LockedSpareBlobs.ExtractAll();
        auto* state = TThreadManager::GetThreadStateChecked();

        size_t count = 0;
        while (blob) {
            auto* nextBlob = blob->Next.load();
            YTALLOC_VERIFY(!blob->Locked.load());
            AssertBlobState(blob, ELargeBlobState::LockedSpare);
            blob->State = ELargeBlobState::Spare;
            arena->SpareBlobs.Put(state, blob);
            blob = nextBlob;
            ++count;
        }

        if (count > 0) {
            YTALLOC_LOG_DEBUG("Locked spare blobs reinstalled (Rank: %d, Blobs: %zu)",
                arena->Rank,
                count);
        }
    }

    void ReinstallLockedFreedBlobs(TLargeArena* arena)
    {
        auto* state = TThreadManager::GetThreadStateChecked();
        auto* blob = arena->LockedFreedBlobs.ExtractAll();

        size_t count = 0;
        while (blob) {
            auto* nextBlob = blob->Next.load();
            AssertBlobState(blob, ELargeBlobState::LockedFreed);
            MoveBlobToSpare(state, arena, blob, false);
            ++count;
            blob = nextBlob;
        }

        if (count > 0) {
            YTALLOC_LOG_DEBUG("Locked freed blobs reinstalled (Rank: %d, Blobs: %zu)",
                arena->Rank,
                count);
        }
    }

    void ReclaimSpareMemory(TLargeArena* arena, ssize_t bytesToReclaim)
    {
        if (bytesToReclaim <= 0) {
            return;
        }

        auto rank = arena->Rank;
        auto* state = TThreadManager::GetThreadStateChecked();

        YTALLOC_LOG_DEBUG("Started processing spare memory in arena (BytesToReclaim: %zdM, Rank: %d)",
            bytesToReclaim / 1_MB,
            rank);

        size_t bytesReclaimed = 0;
        size_t blobsReclaimed = 0;
        while (bytesToReclaim > 0) {
            auto* blob = arena->SpareBlobs.ExtractRoundRobin(state);
            if (!blob) {
                break;
            }

            AssertBlobState(blob, ELargeBlobState::Spare);
            YTALLOC_VERIFY(blob->BytesAllocated == 0);

            auto bytesAcquired = blob->BytesAcquired;
            StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesSpare, -bytesAcquired);
            bytesToReclaim -= bytesAcquired;
            bytesReclaimed += bytesAcquired;
            blobsReclaimed += 1;

            auto* extent = blob->Extent;
            auto* ptr = reinterpret_cast<char*>(blob);
            ReleaseArenaPages(
                state,
                arena,
                ptr,
                bytesAcquired);

            size_t segmentIndex = (ptr - extent->Ptr) / arena->SegmentSize;
            extent->DisposedFlags[segmentIndex].store(true, std::memory_order_relaxed);

            auto* disposedSegment = DisposedSegmentPool_.Allocate();
            disposedSegment->Index = segmentIndex;
            disposedSegment->Extent = extent;
            arena->DisposedSegments.Put(disposedSegment);
        }

        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::SpareBytesReclaimed, bytesReclaimed);

        YTALLOC_LOG_DEBUG("Finished processing spare memory in arena (Rank: %d, BytesReclaimed: %zdM, BlobsReclaimed: %zu)",
            arena->Rank,
            bytesReclaimed / 1_MB,
            blobsReclaimed);
    }

    void ReclaimOverheadMemory(TLargeArena* arena, ssize_t bytesToReclaim)
    {
        if (bytesToReclaim == 0) {
            return;
        }

        auto* state = TThreadManager::GetThreadStateChecked();
        auto rank = arena->Rank;

        YTALLOC_LOG_DEBUG("Started processing overhead memory in arena (BytesToReclaim: %zdM, Rank: %d)",
            bytesToReclaim / 1_MB,
            rank);

        size_t extentsTraversed = 0;
        size_t segmentsTraversed = 0;
        size_t bytesReclaimed = 0;

        bool restartedFromFirstExtent = false;
        auto& currentExtent = arena->CurrentOverheadScanExtent;
        auto& currentSegment = arena->CurrentOverheadScanSegment;
        while (bytesToReclaim > 0) {
            if (!currentExtent) {
                if (restartedFromFirstExtent) {
                    break;
                }
                currentExtent = arena->FirstExtent.load();
                if (!currentExtent) {
                    break;
                }
                restartedFromFirstExtent = true;
            }

            while (currentSegment  < currentExtent->SegmentCount && bytesToReclaim > 0) {
                ++segmentsTraversed;
                if (!currentExtent->DisposedFlags[currentSegment].load(std::memory_order_acquire)) {
                    auto* ptr = currentExtent->Ptr + currentSegment * arena->SegmentSize;
                    auto* blob = reinterpret_cast<TLargeBlobHeader*>(ptr);
                    YTALLOC_PARANOID_ASSERT(blob->Extent == currentExtent);
                    if (TryLockBlob(blob)) {
                        if (blob->BytesAllocated > 0) {
                            size_t rawSize = GetRawBlobSize<TLargeBlobHeader>(blob->BytesAllocated);
                            size_t bytesToRelease = blob->BytesAcquired - rawSize;
                            if (bytesToRelease > 0) {
                                ReleaseArenaPages(
                                    state,
                                    arena,
                                    ptr + blob->BytesAcquired - bytesToRelease,
                                    bytesToRelease);
                                StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesOverhead, -bytesToRelease);
                                blob->BytesAcquired = rawSize;
                                bytesToReclaim -= bytesToRelease;
                                bytesReclaimed += bytesToRelease;
                            }
                        }
                        UnlockBlob(blob);
                    }
                }
                ++currentSegment;
            }

            ++extentsTraversed;
            currentSegment = 0;
            currentExtent = currentExtent->NextExtent;
        }

        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::OverheadBytesReclaimed, bytesReclaimed);

        YTALLOC_LOG_DEBUG("Finished processing overhead memory in arena (Rank: %d, Extents: %zu, Segments: %zu, BytesReclaimed: %zuM)",
            arena->Rank,
            extentsTraversed,
            segmentsTraversed,
            bytesReclaimed / 1_MB);
    }

    void ReinstallLockedBlobs()
    {
        for (auto& arena : Arenas_) {
            ReinstallLockedSpareBlobs(&arena);
            ReinstallLockedFreedBlobs(&arena);
        }
    }

    void ReclaimMemory()
    {
        auto arenaCounters = StatisticsManager->GetLargeArenaAllocationCounters();
        ssize_t bytesToReclaim = GetBytesToReclaim(arenaCounters);
        if (bytesToReclaim == 0) {
            return;
        }

        YTALLOC_LOG_DEBUG("Memory reclaim started (BytesToReclaim: %zdM)",
            bytesToReclaim / 1_MB);

        std::array<ssize_t, LargeRankCount * 2> bytesReclaimablePerArena;
        for (size_t rank = 0; rank < LargeRankCount; ++rank) {
            bytesReclaimablePerArena[rank * 2] = arenaCounters[rank][ELargeArenaCounter::BytesOverhead];
            bytesReclaimablePerArena[rank * 2 + 1] = arenaCounters[rank][ELargeArenaCounter::BytesSpare];
        }

        std::array<ssize_t, LargeRankCount * 2> bytesToReclaimPerArena{};
        while (bytesToReclaim > 0) {
            ssize_t maxBytes = std::numeric_limits<ssize_t>::min();
            int maxIndex = -1;
            for (int index = 0; index < LargeRankCount * 2; ++index) {
                if (bytesReclaimablePerArena[index] > maxBytes) {
                    maxBytes = bytesReclaimablePerArena[index];
                    maxIndex = index;
                }
            }

            if (maxIndex < 0) {
                break;
            }

            auto bytesToReclaimPerStep = std::min<ssize_t>({bytesToReclaim, maxBytes, 4_MB});
            if (bytesToReclaimPerStep < 0) {
                break;
            }

            bytesToReclaimPerArena[maxIndex] += bytesToReclaimPerStep;
            bytesReclaimablePerArena[maxIndex] -= bytesToReclaimPerStep;
            bytesToReclaim -= bytesToReclaimPerStep;
        }

        for (auto& arena : Arenas_) {
            auto rank = arena.Rank;
            ReclaimOverheadMemory(&arena, bytesToReclaimPerArena[rank * 2]);
            ReclaimSpareMemory(&arena, bytesToReclaimPerArena[rank * 2 + 1]);
        }

        YTALLOC_LOG_DEBUG("Memory reclaim finished");
    }

    template <class TState>
    void AllocateArenaExtent(TState* state, TLargeArena* arena)
    {
        auto rank = arena->Rank;
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::ExtentsAllocated, 1);

        size_t segmentCount = LargeExtentSize / arena->SegmentSize;
        size_t extentHeaderSize = AlignUp(sizeof (TLargeBlobExtent) + sizeof (TLargeBlobExtent::DisposedFlags[0]) * segmentCount, PageSize);
        size_t allocationSize = extentHeaderSize + LargeExtentSize;

        auto* ptr = ZoneAllocator_.Allocate(allocationSize, MAP_NORESERVE);
        if (!Dumpable) {
            MappedMemoryManager->DontDump(ptr, allocationSize);
        }

        if (auto backtraceProvider = BacktraceProvider_.load()) {
            std::array<void*, MaxAllocationProfilingBacktraceDepth> frames;
            auto frameCount = backtraceProvider(
                frames.data(),
                MaxAllocationProfilingBacktraceDepth,
                3);
            MmapObservationManager->EnqueueEvent(allocationSize, frames, frameCount);
        }

        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesMapped, allocationSize);
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::PagesMapped, allocationSize / PageSize);

        auto* extent = static_cast<TLargeBlobExtent*>(ptr);
        MappedMemoryManager->Populate(ptr, extentHeaderSize);
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesPopulated, extentHeaderSize);
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::PagesPopulated, extentHeaderSize / PageSize);
        StatisticsManager->IncrementSystemCounter(ESystemCounter::BytesAllocated, extentHeaderSize);

        new (extent) TLargeBlobExtent(segmentCount, static_cast<char*>(ptr) + extentHeaderSize);

        for (size_t index = 0; index < segmentCount; ++index) {
            auto* disposedSegment = DisposedSegmentPool_.Allocate();
            disposedSegment->Index = index;
            disposedSegment->Extent = extent;
            arena->DisposedSegments.Put(disposedSegment);
            extent->DisposedFlags[index].store(true);
        }

        auto* expectedFirstExtent = arena->FirstExtent.load();
        do {
            extent->NextExtent = expectedFirstExtent;
        } while (Y_UNLIKELY(!arena->FirstExtent.compare_exchange_weak(expectedFirstExtent, extent)));
    }

    template <class TState>
    void* DoAllocate(TState* state, size_t size)
    {
        auto rawSize = GetRawBlobSize<TLargeBlobHeader>(size);
        auto rank = GetLargeRank(rawSize);
        auto tag = ConfigurationManager->IsLargeArenaAllocationProfiled(rank)
            ? BacktraceManager->GetMemoryTagFromBacktrace(3)
            : TThreadManager::GetCurrentMemoryTag();
        auto& arena = Arenas_[rank];
        YTALLOC_PARANOID_ASSERT(rawSize <= arena.SegmentSize);

        TLargeBlobHeader* blob;
        while (true) {
            blob = arena.SpareBlobs.Extract(state);
            if (blob) {
                AssertBlobState(blob, ELargeBlobState::Spare);
                if (TryLockBlob(blob)) {
                    StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesSpare, -blob->BytesAcquired);
                    if (blob->BytesAcquired < rawSize) {
                        PopulateArenaPages(
                            state,
                            &arena,
                            reinterpret_cast<char*>(blob) + blob->BytesAcquired,
                            rawSize - blob->BytesAcquired);
                        blob->BytesAcquired = rawSize;
                    } else {
                        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesOverhead, blob->BytesAcquired - rawSize);
                    }
                    YTALLOC_PARANOID_ASSERT(blob->BytesAllocated == 0);
                    blob->BytesAllocated = size;
                    blob->Tag = tag;
                    blob->State = ELargeBlobState::Allocated;
                    UnlockBlob(blob);
                    break;
                } else {
                    blob->State = ELargeBlobState::LockedSpare;
                    arena.LockedSpareBlobs.Put(blob);
                }
            }

            auto* disposedSegment = arena.DisposedSegments.Extract();
            if (disposedSegment) {
                auto index = disposedSegment->Index;
                auto* extent = disposedSegment->Extent;
                DisposedSegmentPool_.Free(disposedSegment);

                auto* ptr = extent->Ptr + index * arena.SegmentSize;
                PopulateArenaPages(
                    state,
                    &arena,
                    ptr,
                    rawSize);

                blob = reinterpret_cast<TLargeBlobHeader*>(ptr);
                new (blob) TLargeBlobHeader(extent, rawSize, size, tag);

                extent->DisposedFlags[index].store(false, std::memory_order_release);

                break;
            }

            AllocateArenaExtent(state, &arena);
        }

        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BlobsAllocated, 1);
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesAllocated, size);
        StatisticsManager->IncrementTotalCounter(state, tag, EBasicCounter::BytesAllocated, size);
        if (!Dumpable) {
            StatisticsManager->IncrementUndumpableCounter(state, EUndumpableCounter::BytesAllocated, size);
        }

        auto* result = HeaderToPtr(blob);
        YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(result) >= LargeZoneStart(Dumpable) && reinterpret_cast<uintptr_t>(result) < LargeZoneEnd(Dumpable));
        PoisonUninitializedRange(result, size);
        return result;
    }

    template <class TState>
    void DoFree(TState* state, void* ptr)
    {
        YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) >= LargeZoneStart(Dumpable) && reinterpret_cast<uintptr_t>(ptr) < LargeZoneEnd(Dumpable));

        auto* blob = PtrToHeader<TLargeBlobHeader>(ptr);
        AssertBlobState(blob, ELargeBlobState::Allocated);

        auto size = blob->BytesAllocated;
        PoisonFreedRange(ptr, size);

        auto rawSize = GetRawBlobSize<TLargeBlobHeader>(size);
        auto rank = GetLargeRank(rawSize);
        auto& arena = Arenas_[rank];
        YTALLOC_PARANOID_ASSERT(blob->BytesAcquired <= arena.SegmentSize);

        auto tag = blob->Tag;

        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BlobsFreed, 1);
        StatisticsManager->IncrementLargeArenaCounter(state, rank, ELargeArenaCounter::BytesFreed, size);
        StatisticsManager->IncrementTotalCounter(state, tag, EBasicCounter::BytesFreed, size);
        if (!Dumpable) {
            StatisticsManager->IncrementUndumpableCounter(state, EUndumpableCounter::BytesFreed, size);
        }

        if (TryLockBlob(blob)) {
            MoveBlobToSpare(state, &arena, blob, true);
        } else {
            blob->State = ELargeBlobState::LockedFreed;
            arena.LockedFreedBlobs.Put(blob);
        }
    }

private:
    TZoneAllocator ZoneAllocator_;
    std::array<TLargeArena, LargeRankCount> Arenas_;

    static constexpr size_t DisposedSegmentsBatchSize = 1024;
    TSystemPool<TDisposedSegment, DisposedSegmentsBatchSize> DisposedSegmentPool_;

    std::atomic<TBacktraceProvider> BacktraceProvider_ = nullptr;
};

TExplicitlyConstructableSingleton<TLargeBlobAllocator<true>> DumpableLargeBlobAllocator;
TExplicitlyConstructableSingleton<TLargeBlobAllocator<false>> UndumpableLargeBlobAllocator;

////////////////////////////////////////////////////////////////////////////////
// Huge blob allocator
//
// Basically a wrapper for TZoneAllocator.

// Acts as a signature to detect broken headers.
enum class EHugeBlobState : ui64
{
    Allocated = 0x72666c656772616cULL // hugeallc
};

// Every huge blob (both tagged or not) is prepended with this header.
struct THugeBlobHeader
{
    THugeBlobHeader(TMemoryTag tag, size_t size, bool dumpable)
        : Tag(tag)
        , Size(size)
        , State(EHugeBlobState::Allocated)
        , Dumpable(dumpable)
    { }

    TMemoryTag Tag;
    size_t Size;
    EHugeBlobState State;
    bool Dumpable;
    char Padding[7];
};

CHECK_HEADER_ALIGNMENT(THugeBlobHeader)

class THugeBlobAllocator
{
public:
    THugeBlobAllocator()
        : ZoneAllocator_(HugeZoneStart, HugeZoneEnd)
    { }

    void* Allocate(size_t size, bool dumpable)
    {
        YTALLOC_VERIFY(size <= MaxAllocationSize);
        auto tag = TThreadManager::GetCurrentMemoryTag();
        auto rawSize = GetRawBlobSize<THugeBlobHeader>(size);
        auto* blob = static_cast<THugeBlobHeader*>(ZoneAllocator_.Allocate(rawSize, MAP_POPULATE));
        if (!dumpable) {
            MappedMemoryManager->DontDump(blob, rawSize);
        }
        new (blob) THugeBlobHeader(tag, size, dumpable);

        StatisticsManager->IncrementTotalCounter(tag, EBasicCounter::BytesAllocated, size);
        StatisticsManager->IncrementHugeCounter(EHugeCounter::BlobsAllocated, 1);
        StatisticsManager->IncrementHugeCounter(EHugeCounter::BytesAllocated, size);
        if (!dumpable) {
            StatisticsManager->IncrementHugeUndumpableCounter(EUndumpableCounter::BytesAllocated, size);
        }

        auto* result = HeaderToPtr(blob);
        PoisonUninitializedRange(result, size);
        return result;
    }

    void Free(void* ptr)
    {
        auto* blob = PtrToHeader<THugeBlobHeader>(ptr);
        AssertBlobState(blob, EHugeBlobState::Allocated);
        auto tag = blob->Tag;
        auto size = blob->Size;
        auto dumpable = blob->Dumpable;
        PoisonFreedRange(ptr, size);

        auto rawSize = GetRawBlobSize<THugeBlobHeader>(size);
        ZoneAllocator_.Free(blob, rawSize);

        StatisticsManager->IncrementTotalCounter(tag, EBasicCounter::BytesFreed, size);
        StatisticsManager->IncrementHugeCounter(EHugeCounter::BlobsFreed, 1);
        StatisticsManager->IncrementHugeCounter(EHugeCounter::BytesFreed, size);
        if (!dumpable) {
            StatisticsManager->IncrementHugeUndumpableCounter(EUndumpableCounter::BytesFreed, size);
        }
    }

    static size_t GetAllocationSize(const void* ptr)
    {
        UnalignPtr<THugeBlobHeader>(ptr);
        const auto* blob = PtrToHeader<THugeBlobHeader>(ptr);
        return blob->Size;
    }

    static size_t GetAllocationSize(size_t size)
    {
        return GetBlobAllocationSize<THugeBlobHeader>(size);
    }

private:
    TZoneAllocator ZoneAllocator_;
};

TExplicitlyConstructableSingleton<THugeBlobAllocator> HugeBlobAllocator;

////////////////////////////////////////////////////////////////////////////////
// A thunk to large and huge blob allocators

class TBlobAllocator
{
public:
    static void* Allocate(size_t size)
    {
        InitializeGlobals();
        bool dumpable = GetCurrentMemoryZone() != EMemoryZone::Undumpable;
        // NB: Account for the header. Also note that we may safely ignore the alignment since
        // HugeAllocationSizeThreshold is already page-aligned.
        if (Y_LIKELY(size < HugeAllocationSizeThreshold - sizeof(TLargeBlobHeader) - RightReadableAreaSize)) {
            void* result = dumpable
                ? DumpableLargeBlobAllocator->Allocate(size)
                : UndumpableLargeBlobAllocator->Allocate(size);
            YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(result) >= LargeZoneStart(dumpable) && reinterpret_cast<uintptr_t>(result) < LargeZoneEnd(dumpable));
            return result;
        } else {
            auto* result = HugeBlobAllocator->Allocate(size, dumpable);
            YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(result) >= HugeZoneStart && reinterpret_cast<uintptr_t>(result) < HugeZoneEnd);
            return result;
        }
    }

    static void Free(void* ptr)
    {
        InitializeGlobals();
        if (reinterpret_cast<uintptr_t>(ptr) < LargeZoneEnd(true)) {
            YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) >= LargeZoneStart(true) && reinterpret_cast<uintptr_t>(ptr) < LargeZoneEnd(true));
            UnalignPtr<TLargeBlobHeader>(ptr);
            DumpableLargeBlobAllocator->Free(ptr);
        } else if (reinterpret_cast<uintptr_t>(ptr) < LargeZoneEnd(false)) {
            YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) >= LargeZoneStart(false) && reinterpret_cast<uintptr_t>(ptr) < LargeZoneEnd(false));
            UnalignPtr<TLargeBlobHeader>(ptr);
            UndumpableLargeBlobAllocator->Free(ptr);
        } else if (reinterpret_cast<uintptr_t>(ptr) < HugeZoneEnd) {
            YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) >= HugeZoneStart && reinterpret_cast<uintptr_t>(ptr) < HugeZoneEnd);
            UnalignPtr<THugeBlobHeader>(ptr);
            HugeBlobAllocator->Free(ptr);
        } else {
            YTALLOC_TRAP("Wrong ptr passed to Free");
        }
    }
};

////////////////////////////////////////////////////////////////////////////////

Y_POD_THREAD(bool) CurrentThreadIsBackground;

// Base class for all background threads.
template <class T>
class TBackgroundThreadBase
{
public:
    TBackgroundThreadBase()
        : State_(new TState())
    {
        NThreading::RegisterAtForkHandlers(
            [=] { BeforeFork(); },
            [=] { AfterForkParent(); },
            [=] { AfterForkChild(); });
    }

    virtual ~TBackgroundThreadBase()
    {
        Stop();
    }

private:
    struct TState
        : public TSystemAllocatable
    {
        std::mutex StartStopMutex;
        std::optional<std::thread> Thread;

        std::mutex StopFlagMutex;
        std::condition_variable StopFlagVariable;
        std::chrono::system_clock::time_point LastInvocationTime;
        bool StopFlag = false;
        bool Paused = false;

        std::atomic<int> ForkDepth = 0;
        bool RestartAfterFork = false;
    };

    TState* State_;

private:
    void BeforeFork()
    {
        bool stopped = Stop();
        if (State_->ForkDepth++ == 0) {
            State_->RestartAfterFork = stopped;
        }
    }

    void AfterForkParent()
    {
        if (--State_->ForkDepth == 0) {
            if (State_->RestartAfterFork) {
                Start(false);
            }
        }
    }

    void AfterForkChild()
    {
        bool restart = State_->RestartAfterFork;
        State_ = new TState();
        if (restart) {
            Start(false);
        }
    }

    virtual void ThreadMain() = 0;

protected:
    void Start(bool fromAlloc)
    {
        std::unique_lock<std::mutex> guard(State_->StartStopMutex, std::defer_lock);
        if (fromAlloc) {
            if (!guard.try_lock()) {
                return;
            }

            if (State_->Paused) {
                return;
            }
        } else {
            guard.lock();
        }

        State_->Paused = false;
        if (State_->Thread) {
            return;
        }

        State_->StopFlag = false;

        State_->Thread.emplace([=] {
            CurrentThreadIsBackground = true;
            ThreadMain();
        });

        OnStart();
    }

    bool Stop()
    {
        std::unique_lock<std::mutex> guard(State_->StartStopMutex);

        State_->Paused = true;
        if (!State_->Thread) {
            return false;
        }

        std::unique_lock<std::mutex> flagGuard(State_->StopFlagMutex);
        State_->StopFlag = true;
        flagGuard.unlock();
        State_->StopFlagVariable.notify_one();

        State_->Thread->join();
        State_->Thread.reset();

        OnStop();

        return true;
    }

    bool IsDone(TDuration interval)
    {
        std::unique_lock<std::mutex> flagGuard(State_->StopFlagMutex);
        auto result = State_->StopFlagVariable.wait_until(
            flagGuard,
            State_->LastInvocationTime + std::chrono::microseconds(interval.MicroSeconds()),
            [&] { return State_->StopFlag; });
        State_->LastInvocationTime = std::chrono::system_clock::now();
        return result;
    }

    virtual void OnStart()
    { }

    virtual void OnStop()
    { }
};

////////////////////////////////////////////////////////////////////////////////

// Invokes madvise(MADV_STOCKPILE) periodically.
class TStockpileThread
    : public TBackgroundThreadBase<TStockpileThread>
{
public:
    explicit TStockpileThread(int index)
        : Index_(index)
    {
        Start(false);
    }

private:
    const int Index_;

    virtual void ThreadMain() override
    {
        TThread::SetCurrentThreadName(Sprintf("%s:%d", StockpileThreadName, Index_).c_str());

        while (!IsDone(ConfigurationManager->GetStockpileInterval())) {
            if (!MappedMemoryManager->Stockpile(ConfigurationManager->GetStockpileSize())) {
                // No use to proceed.
                YTALLOC_LOG_INFO("Stockpile call failed; terminating stockpile thread");
                break;
            }
        }
    }
};

// Manages a bunch of TStockpileThreads.
class TStockpileManager
{
public:
    void SpawnIfNeeded()
    {
        if (!ConfigurationManager->IsStockpileEnabled()) {
            return;
        }

        int threadCount = ConfigurationManager->GetStockpileThreadCount();
        while (static_cast<int>(Threads_.size()) > threadCount) {
            Threads_.pop_back();
        }
        while (static_cast<int>(Threads_.size()) < threadCount) {
            Threads_.push_back(std::make_unique<TStockpileThread>(static_cast<int>(Threads_.size())));
        }
    }

private:
    std::vector<std::unique_ptr<TStockpileThread>> Threads_;
};

TExplicitlyConstructableSingleton<TStockpileManager> StockpileManager;

////////////////////////////////////////////////////////////////////////////////

// Time to wait before re-spawning the thread after a fork.
static constexpr auto BackgroundThreadRespawnDelay = TDuration::Seconds(3);

// Runs basic background activities: reclaim, logging, profiling etc.
class TBackgroundThread
    : public TBackgroundThreadBase<TBackgroundThread>
{
public:
    bool IsStarted()
    {
        return Started_.load();
    }

    void SpawnIfNeeded()
    {
        if (CurrentThreadIsBackground) {
            return;
        }
        Start(true);
    }

private:
    std::atomic<bool> Started_ = false;

private:
    virtual void ThreadMain() override
    {
        TThread::SetCurrentThreadName(BackgroundThreadName);
        TimingManager->DisableForCurrentThread();
        MmapObservationManager->DisableForCurrentThread();

        while (!IsDone(BackgroundInterval)) {
            DumpableLargeBlobAllocator->RunBackgroundTasks();
            UndumpableLargeBlobAllocator->RunBackgroundTasks();
            MappedMemoryManager->RunBackgroundTasks();
            TimingManager->RunBackgroundTasks();
            MmapObservationManager->RunBackgroundTasks();
            StockpileManager->SpawnIfNeeded();
        }
    }

    virtual void OnStart() override
    {
        DoUpdateAllThreadsControlWord(true);
    }

    virtual void OnStop() override
    {
        DoUpdateAllThreadsControlWord(false);
    }

    void DoUpdateAllThreadsControlWord(bool started)
    {
        // Update threads' TLS.
        ThreadManager->EnumerateThreadStatesSync(
            [&] {
                Started_.store(started);
            },
            [&] (auto* state) {
                if (state->BackgroundThreadStarted) {
                    *state->BackgroundThreadStarted = started;
                }
            });
    }
};

TExplicitlyConstructableSingleton<TBackgroundThread> BackgroundThread;

////////////////////////////////////////////////////////////////////////////////

Y_FORCE_INLINE TThreadState* TThreadManager::GetThreadStateUnchecked()
{
    YTALLOC_PARANOID_ASSERT(ThreadState_);
    return ThreadState_;
}

Y_FORCE_INLINE TThreadState* TThreadManager::FindThreadState()
{
    if (Y_LIKELY(ThreadState_)) {
        return ThreadState_;
    }

    if (ThreadStateDestroyed_) {
        return nullptr;
    }

    InitializeGlobals();

    // InitializeGlobals must not allocate.
    Y_ABORT_UNLESS(!ThreadState_);
    ThreadState_ = ThreadManager->AllocateThreadState();
    (&ThreadControlWord_)->Parts.ThreadStateValid = true;

    return ThreadState_;
}

void TThreadManager::DestroyThread(void*)
{
    TSmallAllocator::PurgeCaches();

    TThreadState* state = ThreadState_;
    ThreadState_ = nullptr;
    ThreadStateDestroyed_ = true;
    (&ThreadControlWord_)->Parts.ThreadStateValid = false;

    {
        auto guard = GuardWithTiming(ThreadManager->ThreadRegistryLock_);
        state->AllocationProfilingEnabled = nullptr;
        state->BackgroundThreadStarted = nullptr;
        ThreadManager->UnrefThreadState(state);
    }
}

void TThreadManager::DestroyThreadState(TThreadState* state)
{
    StatisticsManager->AccumulateLocalCounters(state);
    ThreadRegistry_.Remove(state);
    ThreadStatePool_.Free(state);
}

void TThreadManager::AfterFork()
{
    auto guard = GuardWithTiming(ThreadRegistryLock_);
    ThreadRegistry_.Clear();
    TThreadState* state = ThreadState_;
    if (state) {
        ThreadRegistry_.PushBack(state);
    }
}

TThreadState* TThreadManager::AllocateThreadState()
{
    auto* state = ThreadStatePool_.Allocate();
    state->AllocationProfilingEnabled = &(*&ThreadControlWord_).Parts.AllocationProfilingEnabled;
    state->BackgroundThreadStarted = &(*&ThreadControlWord_).Parts.BackgroundThreadStarted;

    {
        auto guard = GuardWithTiming(ThreadRegistryLock_);
        // NB: These flags must be initialized under ThreadRegistryLock_; see EnumerateThreadStatesSync.
        *state->AllocationProfilingEnabled = ConfigurationManager->IsAllocationProfilingEnabled();
        *state->BackgroundThreadStarted = BackgroundThread->IsStarted();
        ThreadRegistry_.PushBack(state);
    }

    // Need to pass some non-null value for DestroyThread to be called.
    pthread_setspecific(ThreadDtorKey_, (void*)-1);

    return state;
}

////////////////////////////////////////////////////////////////////////////////

void InitializeGlobals()
{
    static std::once_flag Initialized;
    std::call_once(Initialized, [] () {
        LogManager.Construct();
        BacktraceManager.Construct();
        StatisticsManager.Construct();
        MappedMemoryManager.Construct();
        ThreadManager.Construct();
        GlobalState.Construct();
        DumpableLargeBlobAllocator.Construct();
        UndumpableLargeBlobAllocator.Construct();
        HugeBlobAllocator.Construct();
        ConfigurationManager.Construct();
        SystemAllocator.Construct();
        TimingManager.Construct();
        MmapObservationManager.Construct();
        StockpileManager.Construct();
        BackgroundThread.Construct();

        SmallArenaAllocators.Construct();
        auto constructSmallArenaAllocators = [&] (EAllocationKind kind, uintptr_t zonesStart) {
            for (size_t rank = 1; rank < SmallRankCount; ++rank) {
                (*SmallArenaAllocators)[kind][rank].Construct(kind, rank, zonesStart + rank * SmallZoneSize);
            }
        };
        constructSmallArenaAllocators(EAllocationKind::Untagged, UntaggedSmallZonesStart);
        constructSmallArenaAllocators(EAllocationKind::Tagged, TaggedSmallZonesStart);

        GlobalSmallChunkCaches.Construct();
        (*GlobalSmallChunkCaches)[EAllocationKind::Tagged].Construct(EAllocationKind::Tagged);
        (*GlobalSmallChunkCaches)[EAllocationKind::Untagged].Construct(EAllocationKind::Untagged);
    });
}

////////////////////////////////////////////////////////////////////////////////

void StartBackgroundThread()
{
    InitializeGlobals();
    BackgroundThread->SpawnIfNeeded();
}

////////////////////////////////////////////////////////////////////////////////

template <class... Ts>
Y_FORCE_INLINE void* AllocateSmallUntagged(size_t rank, Ts... args)
{
    auto* result = TSmallAllocator::Allocate<EAllocationKind::Untagged>(NullMemoryTag, rank, std::forward<Ts>(args)...);
    YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(result) >= MinUntaggedSmallPtr && reinterpret_cast<uintptr_t>(result) < MaxUntaggedSmallPtr);
    return result;
}

template <class... Ts>
Y_FORCE_INLINE void* AllocateSmallTagged(ui64 controlWord, size_t rank, Ts... args)
{
    auto tag = Y_UNLIKELY((controlWord & TThreadManager::AllocationProfilingEnabledControlWordMask) && ConfigurationManager->IsSmallArenaAllocationProfiled(rank))
        ? BacktraceManager->GetMemoryTagFromBacktrace(2)
        : static_cast<TMemoryTag>(controlWord & TThreadManager::MemoryTagControlWordMask);
    auto* result = TSmallAllocator::Allocate<EAllocationKind::Tagged>(tag, rank, std::forward<Ts>(args)...);
    YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(result) >= MinTaggedSmallPtr && reinterpret_cast<uintptr_t>(result) < MaxTaggedSmallPtr);
    return result;
}

Y_FORCE_INLINE void* AllocateInline(size_t size)
{
    size_t rank;
    if (Y_LIKELY(size <= 512)) {
        rank = SizeToSmallRank1[(size + 7) >> 3];
    } else if (Y_LIKELY(size < LargeAllocationSizeThreshold)) {
        rank = SizeToSmallRank2[(size - 1) >> 8];
    } else {
        StartBackgroundThread();
        return TBlobAllocator::Allocate(size);
    }

    auto controlWord = TThreadManager::GetThreadControlWord();
    if (Y_LIKELY(controlWord == TThreadManager::FastPathControlWord)) {
        return AllocateSmallUntagged(rank, TThreadManager::GetThreadStateUnchecked());
    }

    if (Y_UNLIKELY(!(controlWord & TThreadManager::BackgroundThreadStartedControlWorkMask))) {
        StartBackgroundThread();
    }

    if (!(controlWord & (TThreadManager::MemoryTagControlWordMask | TThreadManager::AllocationProfilingEnabledControlWordMask))) {
        return AllocateSmallUntagged(rank);
    } else {
        return AllocateSmallTagged(controlWord, rank);
    }
}

Y_FORCE_INLINE void* AllocateSmallInline(size_t rank)
{
    auto controlWord = TThreadManager::GetThreadControlWord();
    if (Y_LIKELY(controlWord == TThreadManager::FastPathControlWord)) {
        return AllocateSmallUntagged(rank, TThreadManager::GetThreadStateUnchecked());
    }

    if (!(controlWord & (TThreadManager::MemoryTagControlWordMask | TThreadManager::AllocationProfilingEnabledControlWordMask))) {
        return AllocateSmallUntagged(rank);
    } else {
        return AllocateSmallTagged(controlWord, rank);
    }
}

Y_FORCE_INLINE void* AllocatePageAlignedInline(size_t size)
{
    size = std::max(AlignUp(size, PageSize), PageSize);
    void* result = size >= LargeAllocationSizeThreshold
        ? AlignUp(TBlobAllocator::Allocate(size + PageSize), PageSize)
        : Allocate(size);
    YTALLOC_ASSERT(reinterpret_cast<uintptr_t>(result) % PageSize == 0);
    return result;
}

Y_FORCE_INLINE void FreeNonNullInline(void* ptr)
{
    YTALLOC_ASSERT(ptr);
    if (Y_LIKELY(reinterpret_cast<uintptr_t>(ptr) < UntaggedSmallZonesEnd)) {
        YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) >= MinUntaggedSmallPtr && reinterpret_cast<uintptr_t>(ptr) < MaxUntaggedSmallPtr);
        TSmallAllocator::Free<EAllocationKind::Untagged>(ptr);
    } else if (Y_LIKELY(reinterpret_cast<uintptr_t>(ptr) < TaggedSmallZonesEnd)) {
        YTALLOC_PARANOID_ASSERT(reinterpret_cast<uintptr_t>(ptr) >= MinTaggedSmallPtr && reinterpret_cast<uintptr_t>(ptr) < MaxTaggedSmallPtr);
        TSmallAllocator::Free<EAllocationKind::Tagged>(ptr);
    } else {
        TBlobAllocator::Free(ptr);
    }
}

Y_FORCE_INLINE void FreeInline(void* ptr)
{
    if (Y_LIKELY(ptr)) {
        FreeNonNullInline(ptr);
    }
}

Y_FORCE_INLINE size_t GetAllocationSizeInline(const void* ptr)
{
    if (Y_UNLIKELY(!ptr)) {
        return 0;
    }

    auto uintptr = reinterpret_cast<uintptr_t>(ptr);
    if (uintptr < UntaggedSmallZonesEnd) {
        YTALLOC_PARANOID_ASSERT(uintptr >= MinUntaggedSmallPtr && uintptr < MaxUntaggedSmallPtr);
        return TSmallAllocator::GetAllocationSize(ptr);
    } else if (uintptr < TaggedSmallZonesEnd) {
        YTALLOC_PARANOID_ASSERT(uintptr >= MinTaggedSmallPtr && uintptr < MaxTaggedSmallPtr);
        return TSmallAllocator::GetAllocationSize(ptr);
    } else if (uintptr < LargeZoneEnd(true)) {
        YTALLOC_PARANOID_ASSERT(uintptr >= LargeZoneStart(true) && uintptr < LargeZoneEnd(true));
        return TLargeBlobAllocator<true>::GetAllocationSize(ptr);
    } else if (uintptr < LargeZoneEnd(false)) {
        YTALLOC_PARANOID_ASSERT(uintptr >= LargeZoneStart(false) && uintptr < LargeZoneEnd(false));
        return TLargeBlobAllocator<false>::GetAllocationSize(ptr);
    } else if (uintptr < HugeZoneEnd) {
        YTALLOC_PARANOID_ASSERT(uintptr >= HugeZoneStart && uintptr < HugeZoneEnd);
        return THugeBlobAllocator::GetAllocationSize(ptr);
    } else {
        YTALLOC_TRAP("Wrong ptr passed to GetAllocationSizeInline");
    }
}

Y_FORCE_INLINE size_t GetAllocationSizeInline(size_t size)
{
    if (size <= LargeAllocationSizeThreshold) {
        return TSmallAllocator::GetAllocationSize(size);
    } else if (size <= HugeAllocationSizeThreshold) {
        return TLargeBlobAllocator<true>::GetAllocationSize(size);
    } else {
        return THugeBlobAllocator::GetAllocationSize(size);
    }
}

void EnableLogging(TLogHandler logHandler)
{
    InitializeGlobals();
    LogManager->EnableLogging(logHandler);
}

void SetBacktraceProvider(TBacktraceProvider provider)
{
    InitializeGlobals();
    BacktraceManager->SetBacktraceProvider(provider);
    DumpableLargeBlobAllocator->SetBacktraceProvider(provider);
    UndumpableLargeBlobAllocator->SetBacktraceProvider(provider);
}

void SetBacktraceFormatter(TBacktraceFormatter provider)
{
    InitializeGlobals();
    MmapObservationManager->SetBacktraceFormatter(provider);
}

void EnableStockpile()
{
    InitializeGlobals();
    ConfigurationManager->EnableStockpile();
}

void SetStockpileInterval(TDuration value)
{
    InitializeGlobals();
    ConfigurationManager->SetStockpileInterval(value);
}

void SetStockpileThreadCount(int value)
{
    InitializeGlobals();
    ConfigurationManager->SetStockpileThreadCount(value);
}

void SetStockpileSize(size_t value)
{
    InitializeGlobals();
    ConfigurationManager->SetStockpileSize(value);
}

void SetLargeUnreclaimableCoeff(double value)
{
    InitializeGlobals();
    ConfigurationManager->SetLargeUnreclaimableCoeff(value);
}

void SetTimingEventThreshold(TDuration value)
{
    InitializeGlobals();
    ConfigurationManager->SetTimingEventThreshold(value);
}

void SetMinLargeUnreclaimableBytes(size_t value)
{
    InitializeGlobals();
    ConfigurationManager->SetMinLargeUnreclaimableBytes(value);
}

void SetMaxLargeUnreclaimableBytes(size_t value)
{
    InitializeGlobals();
    ConfigurationManager->SetMaxLargeUnreclaimableBytes(value);
}

void SetAllocationProfilingEnabled(bool value)
{
    ConfigurationManager->SetAllocationProfilingEnabled(value);
}

void SetAllocationProfilingSamplingRate(double rate)
{
    ConfigurationManager->SetAllocationProfilingSamplingRate(rate);
}

void SetSmallArenaAllocationProfilingEnabled(size_t rank, bool value)
{
    ConfigurationManager->SetSmallArenaAllocationProfilingEnabled(rank, value);
}

void SetLargeArenaAllocationProfilingEnabled(size_t rank, bool value)
{
    ConfigurationManager->SetLargeArenaAllocationProfilingEnabled(rank, value);
}

void SetProfilingBacktraceDepth(int depth)
{
    ConfigurationManager->SetProfilingBacktraceDepth(depth);
}

void SetMinProfilingBytesUsedToReport(size_t size)
{
    ConfigurationManager->SetMinProfilingBytesUsedToReport(size);
}

void SetEnableEagerMemoryRelease(bool value)
{
    ConfigurationManager->SetEnableEagerMemoryRelease(value);
}

void SetEnableMadvisePopulate(bool value)
{
    ConfigurationManager->SetEnableMadvisePopulate(value);
}

TEnumIndexedArray<ETotalCounter, ssize_t> GetTotalAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetTotalAllocationCounters();
}

TEnumIndexedArray<ESystemCounter, ssize_t> GetSystemAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetSystemAllocationCounters();
}

TEnumIndexedArray<ESystemCounter, ssize_t> GetUndumpableAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetUndumpableAllocationCounters();
}

TEnumIndexedArray<ESmallCounter, ssize_t> GetSmallAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetSmallAllocationCounters();
}

TEnumIndexedArray<ESmallCounter, ssize_t> GetLargeAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetLargeAllocationCounters();
}

std::array<TEnumIndexedArray<ESmallArenaCounter, ssize_t>, SmallRankCount> GetSmallArenaAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetSmallArenaAllocationCounters();
}

std::array<TEnumIndexedArray<ELargeArenaCounter, ssize_t>, LargeRankCount> GetLargeArenaAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetLargeArenaAllocationCounters();
}

TEnumIndexedArray<EHugeCounter, ssize_t> GetHugeAllocationCounters()
{
    InitializeGlobals();
    return StatisticsManager->GetHugeAllocationCounters();
}

std::vector<TProfiledAllocation> GetProfiledAllocationStatistics()
{
    InitializeGlobals();

    if (!ConfigurationManager->IsAllocationProfilingEnabled()) {
        return {};
    }

    std::vector<TMemoryTag> tags;
    tags.reserve(MaxCapturedAllocationBacktraces + 1);
    for (TMemoryTag tag = AllocationProfilingMemoryTagBase;
        tag < AllocationProfilingMemoryTagBase + MaxCapturedAllocationBacktraces;
        ++tag)
    {
        tags.push_back(tag);
    }
    tags.push_back(AllocationProfilingUnknownMemoryTag);

    std::vector<TEnumIndexedArray<EBasicCounter, ssize_t>> counters;
    counters.resize(tags.size());
    StatisticsManager->GetTaggedMemoryCounters(tags.data(), tags.size(), counters.data());

    std::vector<TProfiledAllocation> statistics;
    for (size_t index = 0; index < tags.size(); ++index) {
        if (counters[index][EBasicCounter::BytesUsed] < static_cast<ssize_t>(ConfigurationManager->GetMinProfilingBytesUsedToReport())) {
            continue;
        }
        auto tag = tags[index];
        auto optionalBacktrace = BacktraceManager->FindBacktrace(tag);
        if (!optionalBacktrace && tag != AllocationProfilingUnknownMemoryTag) {
            continue;
        }
        statistics.push_back(TProfiledAllocation{
            optionalBacktrace.value_or(TBacktrace()),
            counters[index]
        });
    }
    return statistics;
}

TEnumIndexedArray<ETimingEventType, TTimingEventCounters> GetTimingEventCounters()
{
    InitializeGlobals();
    return TimingManager->GetTimingEventCounters();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NYTAlloc
