#include <library/cpp/balloc/lib/alloc_stats.h>

#include <util/system/compiler.h>
#include <atomic>


namespace NAllocStats {

struct TThreadAllocStats {
    i64 CurrSize = 0;
    i64 MaxSize = 0;
};

struct TGlobalAllocStats {
    std::atomic<ui64> LiveLock = {0};
    std::atomic<ui64> Mmap = {0};
};

#if defined(_unix_) && !defined(_darwin_)

__thread bool isEnabled = false;

bool IsEnabled() noexcept {
    return isEnabled;
}

void EnableAllocStats(bool enable) noexcept {
    isEnabled = enable;
}

__thread TThreadAllocStats threadAllocStats;

void IncThreadAllocStats(i64 size) noexcept {
    threadAllocStats.CurrSize += size;
    if (Y_UNLIKELY(threadAllocStats.CurrSize > threadAllocStats.MaxSize)) {
        threadAllocStats.MaxSize = threadAllocStats.CurrSize;
    }
}

void DecThreadAllocStats(i64 size) noexcept {
    threadAllocStats.CurrSize -= size;
}

void ResetThreadAllocStats() noexcept {
    threadAllocStats.CurrSize = 0;
    threadAllocStats.MaxSize = 0;
}

i64 GetThreadAllocMax() noexcept {
    return threadAllocStats.MaxSize;
}

#else // _unix_ && ! _darwin_

bool IsEnabled() noexcept {
    return false;
}
void EnableAllocStats(bool /*enable*/) noexcept {
}
void IncThreadAllocStats(i64 /*size*/) noexcept {
}
void DecThreadAllocStats(i64 /*size*/) noexcept {
}
void ResetThreadAllocStats() noexcept {
}
i64 GetThreadAllocMax() noexcept {
    return 0;
}

#endif // _unix_ && ! _darwin_


#if defined(_x86_64_) || defined(_i386_)
    static constexpr size_t CACHE_LINE_SIZE = 64;
#elif defined(_arm64_) || defined(_ppc64_)
    static constexpr size_t CACHE_LINE_SIZE = 128;
#else
    static constexpr size_t CACHE_LINE_SIZE = 256; // default large enough
#endif

template <typename T>
struct alignas(sizeof(T)) TCacheLineDoublePaddedAtomic {
    char Prefix[CACHE_LINE_SIZE - sizeof(T)];
    T Value;
    char Postfix[CACHE_LINE_SIZE - sizeof(T)];
};

TCacheLineDoublePaddedAtomic<TGlobalAllocStats> GlobalCounters;

void IncLiveLockCounter() noexcept {
    GlobalCounters.Value.LiveLock.fetch_add(1, std::memory_order_seq_cst);
}

ui64 GetLiveLockCounter() noexcept {
    return GlobalCounters.Value.LiveLock.load(std::memory_order_acquire);
}

void IncMmapCounter(ui64 amount) noexcept {
    GlobalCounters.Value.Mmap.fetch_add(amount, std::memory_order_seq_cst);
}

ui64 GetMmapCounter() noexcept {
    return GlobalCounters.Value.Mmap.load(std::memory_order_acquire);
}

}  // namespace NAllocStats
