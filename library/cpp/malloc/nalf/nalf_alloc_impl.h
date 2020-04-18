#pragma once

#include "defs.h"
#include "nalf_alloc.h"
#include "alloc_helpers.h"
#include "nalf_alloc_extmap.h"
#include "nalf_alloc_pagepool.h"
#include "nalf_alloc_chunkheader.h"
#include "nalf_alloc_cannibalizing_4k_cache.h"

#include <util/system/thread.h>
#include <util/generic/intrlist.h>
#include <array>

#include <util/system/winint.h>

namespace NNumaAwareLockFreeAllocator {
    class TPerNodeAllocator;
    class TPerThreadAllocator;

    class TGlobalAllocator : TNonCopyable, public TSystemAllocHelper<TGlobalAllocator> {
        TPerNodeAllocator* Nodes[NALF_ALLOC_NUMANODES]; // pointers for being able to place cache on correct numa-node
        TExtMappingsMap ExtMappingsMap;

        TSpinLock Lock;

    public:
        TGlobalAllocator();
        ~TGlobalAllocator();

        TPerNodeAllocator* PerNodeAllocator(ui32 numaNode);

        void InitPerThreadAllocator(ui32 numaNode, TPerThreadAllocator** thl) noexcept;

        void PlaceExtMapping(void* x, ui32 pages, ui32 numaNode);
        std::pair<ui32, ui32> ClearExtMapping(void* x); // returns (pages, numa-node)
        ui64 LookupExtSize(void* x);                    // returns size

        TVector<TAllocatorStats> AllocatorStats();
    };

    class TPerNodeAllocator : TNonCopyable, public TSystemAllocHelper<TPerNodeAllocator> {
        // TODO: make it tunable by defines
        struct TChunkCache {
            struct TChunk: public TQueueChunkDerived<TChunkHeader*, 4096, TChunk>, public TWithNalfForceChunkedAlloc {};

            ui32 PoolIdx;
            TUnorderedCache<TChunkHeader*, 4096, 1, TChunk> Cache;
            TAtomic ReadRotation;
            TAtomic WriteRotation;
            TAtomic CachedEntries;

            TChunkCache()
                : PoolIdx(Max<ui32>())
                , ReadRotation(0)
                , WriteRotation(0)
                , CachedEntries(0)
            {
            }
        };

        template <ui32 T, ui32 X>
        struct TSysCache {
            struct TChunk: public TQueueChunkDerived<void*, 4096, TChunk>, public TWithNalfForceChunkedAlloc {};

            static const ui32 PageSize = T;
            static const ui32 MaxCache = X;

            TUnorderedCache<void*, 4096, 1, TChunk> Cache;
            TAtomic ReadRotation;
            TAtomic WriteRotation;
            TAtomic CachedEntries;

            TSysCache()
                : ReadRotation(0)
                , WriteRotation(0)
                , CachedEntries(0)
            {
            }
        };

        template <ui32 T>
        struct TPagedCache {
            struct TChunk: public TQueueChunkDerived<void*, 4096, TChunk>, public TWithNalfForceChunkedAlloc {};

            static const ui32 PageSize = T;

            ui32 PoolIdx;
            TUnorderedCache<void*, 4096, 1, TChunk> Cache;
            TAtomic ReadRotation;
            TAtomic WriteRotation;
            TAtomic CachedEntries;

            TPagedCache()
                : PoolIdx(Max<ui32>())
                , ReadRotation(0)
                , WriteRotation(0)
                , CachedEntries(0)
            {
            }
        };

        struct TPagedCache4k {
            static const ui32 PageSize = 4096;

            ui32 PoolIdx;
            TCannibalizing4kCache Cache;
            TAtomic ReadRotation;
            TAtomic WriteRotation;
            TAtomic CachedEntries;

            TPagedCache4k()
                : PoolIdx(Max<ui32>())
                , ReadRotation(0)
                , WriteRotation(0)
                , CachedEntries(0)
            {
            }
        };

        struct TCache: public TWithNalfForceSystemAlloc {
            std::array<TChunkCache, 3> ChunkCache[TChunkHeader::Chunk6528 + 1];
            TAtomic ChunkPagesReserved[TChunkHeader::Chunk6528 + 1];

            // TSysCache<TChunkHeader::SmallChunkSize, NALF_CACHE_4K_PAGES> Sys4k;
            // TSysCache<TChunkHeader::SmallChunkSize * 2, NALF_CACHE_8K_PAGES> Sys8k;
            // TSysCache<TChunkHeader::SmallChunkSize * 4, NALF_CACHE_16K_PAGES> Sys16k;
            // TSysCache<TChunkHeader::SmallChunkSize * 8, NALF_CACHE_32K_PAGES> Sys32k;

            TPagedCache4k Paged4k;
            TPagedCache<TChunkHeader::SmallChunkSize * 2> Paged8k;
            TPagedCache<TChunkHeader::SmallChunkSize * 4> Paged16k;
            TPagedCache<TChunkHeader::SmallChunkSize * 8> Paged32k;

            TSysCache<TChunkHeader::SmallChunkSize * 3, NALF_CACHE_12K_PAGES> Sys12k;
            TSysCache<TChunkHeader::SmallChunkSize * 6, NALF_CACHE_24K_PAGES> Sys24k;
            TSysCache<TChunkHeader::SmallChunkSize * 16, NALF_CACHE_64K_PAGES> Sys64k;

            std::array<TChunkCache, NALF_ALLOC_CACHEINC_POOLS> ChunkIncremental;
            TAtomic IncrementalPagesReserved;

            template <typename TArr>
            void Init(TArr& arr, const ui32* pdx);
        };

        struct TThreadEntry: public TIntrusiveSListItem<TThreadEntry>, public TWithNalfChunkedAlloc {
            TPerThreadAllocator* const Allocator;
#ifdef _win_
            HANDLE ThreadHandle;
#else
            volatile bool Released;
#endif
            TThreadEntry(TPerNodeAllocator* node);

            bool IsReleased();
        };

        TSpinLock Lock;
        TIntrusiveSList<TThreadEntry> Threads;
        void BindThread(TPerThreadAllocator** thl, TThreadEntry& entry);

        std::array<TPagePool, 4> PagePools;
        THolder<TCache> Cache;

        template <typename T>
        TChunkHeader* ClaimChunkDo(TChunkHeader::EChunkType chunkType, T& chunkCache, TAtomic& reservedCounter);

        TChunkHeader* ClaimDummyIncremental();

        template <typename T>
        void RecycleChunkDo(TChunkHeader* header, T& chunkCache);

        template <typename T>
        void* ClaimSys(T& chunkCache);

        template <typename T>
        void* ClaimPaged(T& chunkCache);

        template <typename T>
        void RecycleSys(void* x, T& chunkCache);

        template <typename T>
        void RecyclePaged(void* x, T& chunkCache);

    public:
        TGlobalAllocator* const GlobalAllocator;
        const ui32 NumaNode;

        TPerNodeAllocator(TGlobalAllocator* globalAllocator, ui32 numaNode)
            : GlobalAllocator(globalAllocator)
            , NumaNode(numaNode)
        {
        }

        ~TPerNodeAllocator() {
        }

        TChunkHeader* ClaimChunk(TChunkHeader::EChunkType chunkType);
        void RecycleChunk(TChunkHeader* header);

        void* ClaimSysPage(ui64 sz);
        void RecycleSysPage(void* x);

        void Init();
        void ClaimPerThreadAllocator(TPerThreadAllocator** thl);

        void AllocatorStats(TAllocatorStats& stats);
    };

    class TPerThreadAllocator : TNonCopyable, public TSystemAllocHelper<TPerThreadAllocator> {
    public:
        struct TCacheStats {
            ui64 PagesClaimed;
            ui64 PagesFromCache;
            ui64 PagesReleased;
            ui64 TotalAllocations;
            ui64 TotalReclaimed;

            TCacheStats()
                : PagesClaimed(0)
                , PagesFromCache(0)
                , PagesReleased(0)
                , TotalAllocations(0)
                , TotalReclaimed(0)
            {
            }
        };

    private:
        TPerNodeAllocator* Node;
        ui32 NumaNodeId;

        TAllocHint::EHint Hint;

        struct TChunkCache {
            static const ui32 CurrentCacheLength = 256;
            static const ui32 DelCaches = 3;
            static const ui32 DelCacheLength = 32;

            ui32 ChunkSize;       // 4
            ui32 LengthToRecycle; // 4

            TChunkHeader* Current;          // +8 : 16
            TChunkHeader* Pages[DelCaches]; // +24 : 40
            ui32 CurrentAllocated;          // +4 // how many entries locally allocated (but not yet marked in page Counter field)? (could be negative)
            ui32 CurrentUsed;               // +4 : 48 // how many entries in cache left?
            ui64 RecycleCounter;            // +8 : 56
            ui8 IsCurrentOwner;             //  // could we read more from cached-page, or must reclaim
            ui8 Used[DelCaches];            // +4 : 60
            ui32 FreeChunksToRelease;       // +4

            // cache stores offsets in the chunk
            ui16 CurrentCache[CurrentCacheLength]; // +256*2
            ui16 Cache[DelCaches][DelCacheLength]; // + 32*2 * 3
            ui64 RecycleIndex[DelCaches];          // +24
            ui32 Reserved2nd[2];                   // +8

            TChunkCache() {
                memset(this, 0, sizeof(TChunkCache));
            }
        };

        static_assert(sizeof(TChunkCache) == 32 * 25, "expect sizeof(TChunkCache) == 32 * 25");

        struct TIncrementalCache {
            TChunkHeader* Current;
            ui32 CurrentAllocated;
            ui32 CurrentPosition;

            TChunkHeader* Next;

            TChunkHeader* ToClean;
            ui32 ToCleanCounter;

            TIncrementalCache() {
                memset(this, 0, sizeof(TIncrementalCache));
            }
        };

        TChunkCache CacheChunked[TChunkHeader::Chunk6528 + 1];
        TIncrementalCache CacheIncremental;

        volatile bool* ReleasedFlag;

        void EnsureNumaNode();

        void* AllocateSystem(ui64 len, ui64 align);
        void* AllocateIncremental(ui64 len, ui64 align);
        void* AllocateChunked(ui64 len, ui64 align);

        void FreeIncremental(TChunkHeader* header);
        void FreeChunked(TChunkHeader* header, void* x);

        void MarkIncrementalRecycle(TChunkHeader* header, ui32 sub);

        void* AllocateChunkedCached(TChunkHeader::EChunkType chunkType);

        void MarkChunkRecycle(TChunkHeader* header, TCacheStats* stats, ui32 freeChunksToRecycle, ui32 sub);

    public:
        TPerThreadAllocator(volatile bool* releasedFlag = nullptr);
        ~TPerThreadAllocator();

        void BeginSetup(TPerNodeAllocator* node);
        void CompleteSetup();

        void* Allocate(ui64 len, TAllocHint::EHint hint, ui64 align);
        void Free(void* mem);
        void* Realloc(void* mem, ui64 len);

        ui32 NumaNode() const;
        TAllocHint::EHint SwapHint(TAllocHint::EHint hint);

#if defined NALF_COLLECT_STATS
        TCacheStats ChunkedStats[TChunkHeader::Chunk6528 + 1];
        TCacheStats IncrementalStats;
#endif
    };

}
