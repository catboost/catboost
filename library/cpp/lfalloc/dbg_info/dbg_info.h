#pragma once

#include <util/generic/ptr.h>
#include <util/system/types.h>

namespace NAllocDbg {
    ////////////////////////////////////////////////////////////////////////////////
    // Allocation statistics

    enum ELFAllocCounter {
        CT_USER_ALLOC,     // accumulated size requested by user code
        CT_MMAP,           // accumulated mmapped size
        CT_MMAP_CNT,       // number of mmapped regions
        CT_MUNMAP,         // accumulated unmmapped size
        CT_MUNMAP_CNT,     // number of munmaped regions
        CT_SYSTEM_ALLOC,   // accumulated allocated size for internal lfalloc needs
        CT_SYSTEM_FREE,    // accumulated deallocated size for internal lfalloc needs
        CT_SMALL_ALLOC,    // accumulated allocated size for fixed-size blocks
        CT_SMALL_FREE,     // accumulated deallocated size for fixed-size blocks
        CT_LARGE_ALLOC,    // accumulated allocated size for large blocks
        CT_LARGE_FREE,     // accumulated deallocated size for large blocks
        CT_SLOW_ALLOC_CNT, // number of slow (not LF) allocations
        CT_DEGRAGMENT_CNT, // number of memory defragmentations
        CT_MAX
    };

    i64 GetAllocationCounterFast(ELFAllocCounter counter);
    i64 GetAllocationCounterFull(ELFAllocCounter counter);

    ////////////////////////////////////////////////////////////////////////////////
    // Allocation statistics could be tracked on per-tag basis

    int SetThreadAllocTag(int tag);

    class TScopedTag {
    private:
        int PrevTag;

    public:
        explicit TScopedTag(int tag) {
            PrevTag = SetThreadAllocTag(tag);
        }

        ~TScopedTag() {
            SetThreadAllocTag(PrevTag);
        }
    };

    struct TPerTagAllocInfo {
        ssize_t Count;
        ssize_t Size;
    };

    TArrayPtr<TPerTagAllocInfo> GetPerTagAllocInfo(
        bool flushPerThreadCounters,
        int& maxTag,
        int& numSizes);

    ////////////////////////////////////////////////////////////////////////////////
    // Allocation sampling could be used to collect detailed information

    bool SetProfileCurrentThread(bool newVal);
    bool SetProfileAllThreads(bool newVal);
    bool SetAllocationSamplingEnabled(bool newVal);

    size_t SetAllocationSampleRate(size_t newVal);
    size_t SetAllocationSampleMaxSize(size_t newVal);

#define DBG_ALLOC_INVALID_COOKIE (-1)

    using TAllocationCallback = int(int tag, size_t size, int sizeIdx);
    using TDeallocationCallback = void(int cookie, int tag, size_t size, int sizeIdx);

    TAllocationCallback* SetAllocationCallback(TAllocationCallback* newVal);
    TDeallocationCallback* SetDeallocationCallback(TDeallocationCallback* newVal);

}
