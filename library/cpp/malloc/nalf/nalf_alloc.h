#pragma once

#include <util/generic/vector.h>
#include <util/stream/output.h>

#ifndef NALF_ALLOC_DEFAULTMODE
#define NALF_ALLOC_DEFAULTMODE (TAllocHint::Chunked)
#endif

#ifndef NALF_ALLOC_DEFAULTALIGN
#define NALF_ALLOC_DEFAULTALIGN (16)
#endif

#if defined(_tsan_enabled_) || defined(_msan_enabled_) || defined(_asan_enabled_) || defined(WITH_VALGRIND)
#define NALF_FORCE_MALLOC_FREE 1
#define NALF_DONOT_DEFINE_GLOBALS 1
#endif

namespace NNumaAwareLockFreeAllocator {
    struct TAllocHint {
        enum EHint {
            Undefined,
            Incremental,
            Chunked,
            System,
            ForceIncremental,
            ForceChunked,
            ForceSystem,
            Bootstrap,
        };
        // valid op hint values: incremental, chunked, system, force*
        // valid thread hint values: undefined, incremental, chunked, system
        // bootstrap is used in node initialization only
    };

    class TPerThreadAllocator;

    TPerThreadAllocator* GetThreadAllocator();
    void* Allocate(ui64 len, TAllocHint::EHint hint = NALF_ALLOC_DEFAULTMODE, ui64 align = NALF_ALLOC_DEFAULTALIGN);
    void Free(void* mem);
    void* Realloc(void* mem, ui64 len);
    TAllocHint::EHint SwapHint(TAllocHint::EHint hint) noexcept;
    std::pair<ui64, TAllocHint::EHint> MemBlockSize(void* mem);

    void* Allocate(TPerThreadAllocator* pta, ui64 len, TAllocHint::EHint hint = NALF_ALLOC_DEFAULTMODE, ui64 align = NALF_ALLOC_DEFAULTALIGN);
    void Free(TPerThreadAllocator* pta, void* mem);
    void* Realloc(TPerThreadAllocator* pta, void* mem, ui64 len);

    TAllocHint::EHint SwapHint(TPerThreadAllocator* pta, TAllocHint::EHint hint) noexcept;

    template <TAllocHint::EHint Hint>
    struct TSwapHint : TNonCopyable {
        const TAllocHint::EHint Old;
        TSwapHint()
            : Old(SwapHint(Hint))
        {
        }
        ~TSwapHint() {
            SwapHint(Old);
        }
    };

    void* SystemAllocation(ui64 size);
    void SystemFree(void* mem, ui64 size);
    void* SystemRemap(void* mem, ui64 oldsize, ui64 newsize);
    ui32 GetNumaNode();

    struct TAllocatorStats {
        ui64 TotalBytesReserved; // w/o system bytes!
        ui32 PerThreadEntries;

        struct TSizeStats {
            ui32 PageSize;
            ui32 ChunkSize;

            ui64 TotalPagesReserved;
            ui64 TotalPagesCached;
            ui64 TotalAllocations;
            ui64 TotalReclaimed;
            ui64 PagesClaimed;
            ui64 PagesFromCache;
            ui64 PagesReleased;

            TSizeStats();
        };

        struct TIncrementalStats {
            ui64 TotalPagesReserved;
            ui64 TotalPagesCached;
            ui64 TotalAllocations;
            ui64 TotalReclaimed;
            ui64 PagesClaimed;
            ui64 PagesFromCache;
            ui64 PagesReleased;

            TIncrementalStats();
        };

        struct TSysStats {
            ui64 TotalBytesReserved;
            ui64 TotalBytesCached;
            ui64 TotalAllocations;
            ui64 TotalReclaimed;

            TSysStats();
        };

        TVector<TSizeStats> BySizeStats;
        TIncrementalStats IncrementalStats;
        TSysStats SysStats;

        TAllocatorStats();
        void Out(IOutputStream& out) const;
    };

    TVector<TAllocatorStats> GetAllocatorStats(); // one entry per numa-node

    static const ui64 SystemPageSize = 4096;

}
