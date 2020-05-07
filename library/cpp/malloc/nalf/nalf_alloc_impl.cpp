#include <util/thread/singleton.h>
#include "nalf_alloc_impl.h"
#include "nalf_alloc_extmap.h"
#include "alloc_helpers.h"
#include "nalf_alloc_pagepool.h"

#ifdef _unix_
#include <sys/mman.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#endif

#ifdef NALF_FORCE_MALLOC_FREE
#include <stdlib.h>
#endif

namespace NNumaAwareLockFreeAllocator {
    Y_POD_STATIC_THREAD(TPerThreadAllocator*)
    TlsAllocator((TPerThreadAllocator*)nullptr);

    ui32 GetNumaNode() {
#if !defined _linux_ || !defined SYS_getcpu || NALF_ALLOC_NUMANODES == 1
        return 0;
#else
        unsigned cpu = 0;
        unsigned node = 0;
        void* unused = nullptr;

        syscall(SYS_getcpu, &cpu, &node, unused);

        return node % NALF_ALLOC_NUMANODES;
#endif
    }

#ifndef _win_
    static pthread_key_t PthreadKey;

    static void PerNodeReleaseThread(void* x) {
        TlsAllocator = nullptr;
        bool* relflag = (bool*)x;
        AtomicStore(relflag, true);
    }
#endif

    static TGlobalAllocator* GlobalAllocator() {
        static TGlobalAllocator* volatile x;

        TGlobalAllocator* s = AtomicLoad(&x);
        if (Y_LIKELY(s))
            return s;

        static TAtomic lock;

        TGuard<TAtomic> guard(lock);
        if ((s = AtomicLoad(&x)))
            return s;

#ifndef _win_
        if (pthread_key_create(&PthreadKey, PerNodeReleaseThread) && pthread_key_create(&PthreadKey, PerNodeReleaseThread))
            Y_FAIL();
#endif

        TGlobalAllocator* alx = ::new (SystemAllocation(sizeof(TGlobalAllocator))) TGlobalAllocator();
        AtomicStore(&x, alx);
        return alx;
    }

    TVector<TAllocatorStats> GetAllocatorStats() {
        return GlobalAllocator()->AllocatorStats();
    }

    template <typename TArr>
    void TPerNodeAllocator::TCache::Init(TArr& arr, const ui32* pdx) {
        for (ui32 i = 0; i < arr.size(); ++i)
            arr[i].PoolIdx = pdx[i];
    }

    void TPerNodeAllocator::Init() {
        // we are still under global lock
        // bootstrap allocator active (we must be able to allocate incremental pages w/o going to cache)
        PagePools[0].Init(TPagePool::Class1Gb, 1);
        PagePools[1].Init(TPagePool::Class2Mb, 1);
        PagePools[2].Init(TPagePool::Class2Mb, 8);
        PagePools[3].Init(TPagePool::Class4k, Max<ui32>());

        Cache.Reset(new TCache());

        const ui32 chunk16Pools[3] = NALF_ALLOC_CACHE16_POOLSV;
        const ui32 chunk32Pools[3] = NALF_ALLOC_CACHE32_POOLSV;
        const ui32 chunk48Pools[3] = NALF_ALLOC_CACHE48_POOLSV;
        const ui32 chunk64Pools[3] = NALF_ALLOC_CACHE64_POOLSV;
        const ui32 chunk96Pools[3] = NALF_ALLOC_CACHE96_POOLSV;
        const ui32 chunk128Pools[3] = NALF_ALLOC_CACHE128_POOLSV;
        const ui32 chunk192Pools[3] = NALF_ALLOC_CACHE192_POOLSV;
        const ui32 chunk256Pools[3] = NALF_ALLOC_CACHE256_POOLSV;
        const ui32 chunk384Pools[3] = NALF_ALLOC_CACHE384_POOLSV;
        const ui32 chunk512Pools[3] = NALF_ALLOC_CACHE512_POOLSV;
        const ui32 chunk768Pools[3] = NALF_ALLOC_CACHE768_POOLSV;
        const ui32 chunk1024Pools[3] = NALF_ALLOC_CACHE1024_POOLSV;
        const ui32 chunk1536Pools[3] = NALF_ALLOC_CACHE1536_POOLSV;
        const ui32 chunk2176Pools[3] = NALF_ALLOC_CACHE2176_POOLSV;
        const ui32 chunk3584Pools[3] = NALF_ALLOC_CACHE3584_POOLSV;
        const ui32 chunk6528Pools[3] = NALF_ALLOC_CACHE6528_POOLSV;
        const ui32 chunkIncPools[3] = NALF_ALLOC_CACHEINC_POOLSV;

        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk16], chunk16Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk32], chunk32Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk48], chunk48Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk64], chunk64Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk96], chunk96Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk128], chunk128Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk192], chunk192Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk256], chunk256Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk384], chunk384Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk512], chunk512Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk768], chunk768Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk1024], chunk1024Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk1536], chunk1536Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk2176], chunk2176Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk3584], chunk3584Pools);
        Cache->Init(Cache->ChunkCache[TChunkHeader::Chunk6528], chunk6528Pools);

        Cache->Init(Cache->ChunkIncremental, chunkIncPools);

        Cache->Paged4k.PoolIdx = NALF_ALLOC_CACHE_4K_POOL;
        Cache->Paged8k.PoolIdx = NALF_ALLOC_CACHE_8K_POOL;
        Cache->Paged16k.PoolIdx = NALF_ALLOC_CACHE_16K_POOL;
        Cache->Paged32k.PoolIdx = NALF_ALLOC_CACHE_32K_POOL;
    }

    TPerNodeAllocator::TThreadEntry::TThreadEntry(TPerNodeAllocator* node)
#ifdef _win_
        : Allocator(new TPerThreadAllocator())
        , ThreadHandle(0)
#else
        : Allocator(new TPerThreadAllocator(&this->Released))
        , Released(true)
#endif
    {
        Allocator->BeginSetup(node);
        Allocator->CompleteSetup();
    }

    bool TPerNodeAllocator::TThreadEntry::IsReleased() {
#ifdef _win_
        if (ThreadHandle == 0)
            return true;
        DWORD exitStatus;
        bool isDead = !::GetExitCodeThread(ThreadHandle, &exitStatus) || exitStatus != STILL_ACTIVE;
        if (isDead) {
            ::CloseHandle(ThreadHandle);
            ThreadHandle = 0;
            return true;
        }
        return false;
#else
        return AtomicLoad(&Released);
#endif
    }

    void TPerNodeAllocator::BindThread(TPerThreadAllocator** thl, TPerNodeAllocator::TThreadEntry& entry) {
#ifdef _win_
        BOOL b = DuplicateHandle(
            GetCurrentProcess(), GetCurrentThread(),
            GetCurrentProcess(), &entry.ThreadHandle,
            0, FALSE, DUPLICATE_SAME_ACCESS);
        Y_VERIFY(b);
#else
        AtomicStore(&entry.Released, false);
        pthread_setspecific(PthreadKey, (void*)&entry.Released);
#endif
        AtomicStore(thl, entry.Allocator);
    }

    void TPerNodeAllocator::ClaimPerThreadAllocator(TPerThreadAllocator** thl) {
        TGuard<TSpinLock> guard(&Lock);

        // try to reuse one of already allocated
        for (TIntrusiveSList<TThreadEntry>::TIterator it = Threads.Begin(), end = Threads.End(); it != end; ++it) {
            if (it->IsReleased()) {
                BindThread(thl, *it);
                return;
            }
        }

        // allocate new one
        TThreadEntry* x = new TThreadEntry(this);
        Threads.PushFront(x);
        BindThread(thl, *x);
    }

    void TPerNodeAllocator::AllocatorStats(TAllocatorStats& stats) {
        stats.IncrementalStats.TotalPagesReserved = AtomicLoad(&Cache->IncrementalPagesReserved);

        for (const auto& c : Cache->ChunkIncremental)
            stats.IncrementalStats.TotalPagesCached += AtomicLoad(&c.CachedEntries);

        stats.TotalBytesReserved += stats.IncrementalStats.TotalPagesReserved * TChunkHeader::LargeChunkSize;

        stats.BySizeStats.resize(TChunkHeader::Chunk6528);
        for (TChunkHeader::EChunkType i = TChunkHeader::Chunk16; i < TChunkHeader::Chunk6528 + 1; i = static_cast<TChunkHeader::EChunkType>((ui32)i + 1)) {
            auto& x = stats.BySizeStats[i - 1];
            x.ChunkSize = TChunkHeader::TypeToSize(i);
            x.PageSize = TChunkHeader::TypeToSubpageSize(i);
            x.TotalPagesReserved = AtomicLoad(&Cache->ChunkPagesReserved[i]);
            stats.TotalBytesReserved += AtomicLoad(&x.TotalPagesReserved) * x.PageSize;

            for (const auto& c : Cache->ChunkCache[i])
                x.TotalPagesCached += AtomicLoad(&c.CachedEntries);
        }

#if defined NALF_COLLECT_STATS
        {
            TGuard<TSpinLock> guard(&Lock);

            for (TIntrusiveSList<TThreadEntry>::TIterator it = Threads.Begin(), end = Threads.End(); it != end; ++it) {
                ++stats.PerThreadEntries;
                const TPerThreadAllocator& xal = *it->Allocator;

                stats.IncrementalStats.TotalAllocations += RelaxedLoad(&xal.IncrementalStats.TotalAllocations);
                stats.IncrementalStats.TotalReclaimed += RelaxedLoad(&xal.IncrementalStats.TotalReclaimed);
                stats.IncrementalStats.PagesClaimed += RelaxedLoad(&xal.IncrementalStats.PagesClaimed);
                stats.IncrementalStats.PagesFromCache += RelaxedLoad(&xal.IncrementalStats.PagesFromCache);
                stats.IncrementalStats.PagesReleased += RelaxedLoad(&xal.IncrementalStats.PagesReleased);

                for (TChunkHeader::EChunkType i = TChunkHeader::Chunk16; i < TChunkHeader::Chunk6528 + 1; i = static_cast<TChunkHeader::EChunkType>((ui32)i + 1)) {
                    auto& x = stats.BySizeStats[i - 1];
                    const TPerThreadAllocator::TCacheStats& z = xal.ChunkedStats[i];

                    x.TotalAllocations += RelaxedLoad(&z.TotalAllocations);
                    x.TotalReclaimed += RelaxedLoad(&z.TotalReclaimed);
                    x.PagesClaimed += RelaxedLoad(&z.PagesClaimed);
                    x.PagesFromCache += RelaxedLoad(&z.PagesFromCache);
                    x.PagesReleased += RelaxedLoad(&z.PagesReleased);
                }
            }
        }
#endif
    }

    TChunkHeader* TPerNodeAllocator::ClaimDummyIncremental() {
        void* x = SystemAllocation(TChunkHeader::LargeChunkSize * 2 - SystemPageSize);
        void* aligned = AlignUp(x, TChunkHeader::LargeChunkSize);
        TChunkHeader* ret = ::new (aligned) TChunkHeader(TChunkHeader::ChunkIncremental, x, NumaNode, Max<ui16>());
        return ret;
    }

    template <typename T>
    TChunkHeader* TPerNodeAllocator::ClaimChunkDo(TChunkHeader::EChunkType chunkType, T& chunkCache, TAtomic& reservedCounter) {
        for (ui32 ipool = 0, epool = (ui32)chunkCache.size(); ipool != epool; ++ipool) {
            TChunkCache& c = chunkCache[ipool];
            Y_VERIFY_DEBUG(c.PoolIdx < PagePools.size());

            // first - try get recycled page
            if (AtomicLoad(&c.CachedEntries)) {
                do {
                    if (TChunkHeader* x = c.Cache.Pop(AtomicIncrement(c.ReadRotation))) {
                        AtomicDecrement(c.CachedEntries);
                        return x;
                    }
                } while (AtomicLoad(&c.CachedEntries) > c.Cache.Concurrency * 512);
            }

            // if no recycled page - try allocate more
            TPagePool& pool = PagePools[c.PoolIdx];
            if (ui8* pagemem = (ui8*)pool.Pop()) {
                // ok, we got 32k page, split in pages of appropriate size
                const ui32 subpageSize = TChunkHeader::TypeToSubpageSize(chunkType);
                const ui32 subpagesPerPage = TChunkHeader::LargeChunkSize / subpageSize;
                TChunkHeader* ret = ::new (pagemem) TChunkHeader(chunkType, nullptr, NumaNode, ipool);
                for (ui32 subpageShift = subpageSize; subpageShift != TChunkHeader::LargeChunkSize; subpageShift += subpageSize) {
                    TChunkHeader* x = ::new (pagemem + subpageShift) TChunkHeader(chunkType, nullptr, NumaNode, ipool);
                    c.Cache.Push(x, AtomicIncrement(c.WriteRotation));
                    AtomicIncrement(c.CachedEntries);
                }
                AtomicAdd(reservedCounter, subpagesPerPage);
                return ret;
            }
        }
        Y_FAIL("not able to claim memory chunk"); // TODO: try to steal from other nodes
    }

    TChunkHeader* TPerNodeAllocator::ClaimChunk(TChunkHeader::EChunkType chunkType) {
        if (chunkType != TChunkHeader::ChunkIncremental) {
            return ClaimChunkDo(chunkType, Cache->ChunkCache[chunkType], Cache->ChunkPagesReserved[chunkType]);
        } else {
            if (Y_LIKELY(!!Cache))
                return ClaimChunkDo(TChunkHeader::ChunkIncremental, Cache->ChunkIncremental, Cache->IncrementalPagesReserved);
            else
                return ClaimDummyIncremental();
        }
    }

    template <typename T>
    void TPerNodeAllocator::RecycleChunkDo(TChunkHeader* header, T& chunkCache) {
        const ui32 idx = header->PoolIndex;
        Y_VERIFY_DEBUG(idx < chunkCache.size());
        TChunkCache& c = chunkCache[idx];
        c.Cache.Push(header, AtomicIncrement(c.WriteRotation));
        AtomicIncrement(c.CachedEntries);
    }

    void TPerNodeAllocator::RecycleChunk(TChunkHeader* header) {
        TPerNodeAllocator* node = (header->NumaNode == NumaNode) ? this : GlobalAllocator->PerNodeAllocator(header->NumaNode);

        const TChunkHeader::EChunkType chunkType = (TChunkHeader::EChunkType)header->ChunkType;
        if (chunkType != TChunkHeader::ChunkIncremental) {
            node->RecycleChunkDo(header, node->Cache->ChunkCache[chunkType]);
        } else {
            if (Y_UNLIKELY(header->PoolIndex == Max<ui16>()))
                SystemFree(header->LineStart, TChunkHeader::LargeChunkSize * 2 - SystemPageSize);
            else
                node->RecycleChunkDo(header, node->Cache->ChunkIncremental);
        }
    }

    template <typename T>
    void* TPerNodeAllocator::ClaimSys(T& c) {
        if (AtomicLoad(&c.CachedEntries)) {
            do {
                if (void* x = c.Cache.Pop(AtomicIncrement(c.ReadRotation))) {
                    AtomicDecrement(c.CachedEntries);
                    return x;
                }
            } while (AtomicLoad(&c.CachedEntries) > c.Cache.Concurrency * 16);
        }

        return SystemAllocation(c.PageSize);
    }

    template <typename T>
    void* TPerNodeAllocator::ClaimPaged(T& c) {
        if (AtomicLoad(&c.CachedEntries)) {
            do {
                if (void* x = c.Cache.Pop(AtomicIncrement(c.ReadRotation))) {
                    AtomicDecrement(c.CachedEntries);
                    return x;
                }
            } while (AtomicLoad(&c.CachedEntries) > c.Cache.Concurrency * 64);
        }

        TPagePool& pool = PagePools[c.PoolIdx];
        if (ui8* pagemem = (ui8*)pool.Pop()) {
            // ok, we got 32k page, split in pages of appropriate size
            const ui32 subpageSize = T::PageSize;
            void* ret = pagemem;
            for (ui32 subpageShift = subpageSize; subpageShift != TChunkHeader::LargeChunkSize; subpageShift += subpageSize) {
                void* x = pagemem + subpageShift;
                c.Cache.Push(x, AtomicIncrement(c.WriteRotation));
                AtomicIncrement(c.CachedEntries);
            }
            return ret;
        }

        Y_FAIL("not able to claim memory chunk"); // TODO: try to steal from other nodes
    }

    void* TPerNodeAllocator::ClaimSysPage(ui64 sz) {
#ifndef NALF_ALLOC_DONOTCHECK_NODE
        const ui32 actualNumaNode = GetNumaNode();
        if (actualNumaNode != NumaNode)
            return GlobalAllocator->PerNodeAllocator(actualNumaNode)->ClaimSysPage(sz);
#endif

        const ui64 size = AlignUp<ui64>(sz, SystemPageSize);
        const ui32 pages = (ui32)(size / SystemPageSize);
        if (pages < MaxTrackedSysPages - 1) {
            void* x = nullptr;
            ui32 pagesToMark = 0;
            switch (pages) {
                case 0:
                    Y_FAIL(); // never happens
                    break;
                case 1:
                    pagesToMark = 2;
                    x = ClaimPaged(Cache->Paged4k);
                    break;
                case 2:
                    pagesToMark = 3;
                    x = ClaimPaged(Cache->Paged8k);
                    break;
                case 3:
                    pagesToMark = 4;
                    x = ClaimSys(Cache->Sys12k);
                    break;
                case 4:
                    pagesToMark = 5;
                    x = ClaimPaged(Cache->Paged16k);
                    break;
                case 5:
                case 6:
                    pagesToMark = 7;
                    x = ClaimSys(Cache->Sys24k);
                    break;
                case 7:
                case 8:
                    pagesToMark = 9;
                    x = ClaimPaged(Cache->Paged32k);
                    break;
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                case 16:
                    pagesToMark = 17;
                    x = ClaimSys(Cache->Sys64k);
                    break;
                default:
                    pagesToMark = pages + 1;
                    x = SystemAllocation(size);
                    break;
            }

            GlobalAllocator->PlaceExtMapping(x, pagesToMark, NumaNode);
            return x;
        } else {
            const ui64 sizeToAlloc = size + SystemPageSize;
            ui8* x = (ui8*)SystemAllocation(sizeToAlloc);
            ui8* ret = x + SystemPageSize;
            *(ui32*)x = TChunkHeader::HiddenMagicTag;
            *((ui64*)x + 1) = sizeToAlloc;
            GlobalAllocator->PlaceExtMapping(ret, 1, NumaNode);
            return ret;
        }
    }

    template <typename T>
    void TPerNodeAllocator::RecycleSys(void* x, T& c) {
        if (AtomicLoad(&c.CachedEntries) < c.MaxCache) {
            c.Cache.Push(x, AtomicIncrement(c.WriteRotation));
            AtomicIncrement(c.CachedEntries);
        } else {
            SystemFree(x, c.PageSize);
        }
    }

    template <typename T>
    void TPerNodeAllocator::RecyclePaged(void* x, T& c) {
        c.Cache.Push(x, AtomicIncrement(c.WriteRotation));
        AtomicIncrement(c.CachedEntries);
    }

    void TPerNodeAllocator::RecycleSysPage(void* x) {
        std::pair<ui32, ui32> szn = GlobalAllocator->ClearExtMapping(x);
        Y_VERIFY_DEBUG(szn.first != 0);

        TPerNodeAllocator* node = (szn.second == NumaNode) ? this : GlobalAllocator->PerNodeAllocator(szn.second);

        switch (szn.first) {
            case 0:
                Y_FAIL();
                break;
            case 1: {
                ui8* hiddenPage = (ui8*)x - SystemPageSize;
                const ui32 magic = *(ui32*)hiddenPage;
                Y_VERIFY_DEBUG(magic == TChunkHeader::HiddenMagicTag);
                const ui64 sz = *((ui64*)hiddenPage + 1);
                SystemFree(hiddenPage, sz);
            } break;
            case 2:
                node->RecyclePaged(x, node->Cache->Paged4k);
                break;
            case 3:
                node->RecyclePaged(x, node->Cache->Paged8k);
                break;
            case 4:
                node->RecycleSys(x, node->Cache->Sys12k);
                break;
            case 5:
                node->RecyclePaged(x, node->Cache->Paged16k);
                break;
            case 6:
            case 7:
                node->RecycleSys(x, node->Cache->Sys24k);
                break;
            case 8:
            case 9:
                node->RecyclePaged(x, node->Cache->Paged32k);
                break;
            case 10:
            case 11:
            case 12:
            case 13:
            case 14:
            case 15:
            case 16:
            case 17:
                node->RecycleSys(x, node->Cache->Sys64k);
                break;
            default:
                SystemFree(x, (szn.first - 1) * SystemPageSize);
                break;
        }
    }

    // per-thread allocator

    TPerThreadAllocator* GetThreadAllocator() {
#ifdef NALF_FORCE_MALLOC_FREE
        Y_FAIL();
#endif
        TPerThreadAllocator** thlptr = &TlsAllocator;
        TPerThreadAllocator* thl = AtomicLoad(thlptr);
        if (Y_LIKELY(thl)) {
            return thl;
        } else {
            TGlobalAllocator* global = GlobalAllocator();
            const ui32 numaNode = GetNumaNode();
            global->InitPerThreadAllocator(numaNode, thlptr);
            return AtomicLoad(thlptr);
        }
    }

    // external interface

    void* Allocate(TPerThreadAllocator* pta, ui64 len, TAllocHint::EHint hint, ui64 align) {
        if (len)
            return pta->Allocate(len, hint, align);
        else
            return pta->Allocate(1, hint, align);
    }

    void* Allocate(ui64 len, TAllocHint::EHint hint, ui64 align) {
#ifdef NALF_FORCE_MALLOC_FREE
        if (len == 0)
            len = 1;
#ifdef _win_
        Y_VERIFY(align <= 16);
        return malloc(len);
#else
        void* ptr;
        int res = posix_memalign(&ptr, Max<ui64>(PLATFORM_DATA_ALIGN, align), len);
        return res ? nullptr : ptr;
#endif
#endif

        void* x = Allocate(GetThreadAllocator(), len, hint, align);

#ifdef NALF_ALLOC_DEBUG
        *(ui8*)x = 0x11;
#endif

        return x;
    }

    void Free(TPerThreadAllocator* pta, void* mem) {
        if (mem)
            pta->Free(mem);
    }

    void Free(void* mem) {
#ifdef NALF_FORCE_MALLOC_FREE
        if (mem)
            free(mem);
        return;
#endif

        Free(GetThreadAllocator(), mem);
    }

    void* Realloc(TPerThreadAllocator* pta, void* mem, ui64 len) {
        return pta->Realloc(mem, len);
    }

    void* Realloc(void* mem, ui64 len) {
#ifdef NALF_FORCE_MALLOC_FREE
        return realloc(mem, len);
#endif

        return Realloc(GetThreadAllocator(), mem, len);
    }

    std::pair<ui64, TAllocHint::EHint> MemBlockSize(void* mem) {
#ifdef NALF_FORCE_MALLOC_FREE
        return std::pair<ui64, TAllocHint::EHint>(0, TAllocHint::Undefined);
#endif

        if ((ui64(mem) % SystemPageSize) == 0) {
            return std::pair<ui64, TAllocHint::EHint>(GlobalAllocator()->LookupExtSize(mem), TAllocHint::System);
        } else {
            TChunkHeader* largeChunkHeader = (TChunkHeader*)((ui64)mem & ~(TChunkHeader::LargeChunkSize - 1));
            if (largeChunkHeader->ChunkType == TChunkHeader::ChunkIncremental) {
                const ui64 sz = *((ui16*)mem - 1);
                return std::pair<ui64, TAllocHint::EHint>(sz, TAllocHint::Incremental);
            } else {
                const ui64 sz = TChunkHeader::TypeToSize((TChunkHeader::EChunkType)largeChunkHeader->ChunkType);
                return std::pair<ui64, TAllocHint::EHint>(sz, TAllocHint::Chunked);
            }
        }
    }

    TAllocHint::EHint SwapHint(TPerThreadAllocator* pta, TAllocHint::EHint hint) noexcept {
        return pta->SwapHint(hint);
    }

    TAllocHint::EHint SwapHint(TAllocHint::EHint hint) noexcept {
#ifdef NALF_FORCE_MALLOC_FREE
        return TAllocHint::Undefined;
#endif

        return GetThreadAllocator()->SwapHint(hint);
    }

    // global allocator

    TGlobalAllocator::TGlobalAllocator() {
        for (ui32 i = 0; i != NALF_ALLOC_NUMANODES; ++i)
            Nodes[i] = nullptr;
    }

    TGlobalAllocator::~TGlobalAllocator() {
        // yes, we leak all memory on deallocation, as it happens in process shutdown - who cares?
    }

    TPerNodeAllocator* TGlobalAllocator::PerNodeAllocator(ui32 numaNode) {
        Y_VERIFY_DEBUG(numaNode < NALF_ALLOC_NUMANODES);
        TPerNodeAllocator* pernode = AtomicLoad(&Nodes[numaNode]);

        if (Y_LIKELY(pernode != nullptr)) {
            return pernode;
        }

        TGuard<TSpinLock> guard(&Lock);
        pernode = AtomicLoad(&Nodes[numaNode]);
        if (pernode == nullptr) {
            TPerThreadAllocator** thlptr = &TlsAllocator;
            TPerThreadAllocator* oldthl = *thlptr;

            pernode = new TPerNodeAllocator(this, numaNode);

            TPerThreadAllocator tmp;
            tmp.BeginSetup(pernode);
            AtomicStore(thlptr, &tmp);
            pernode->Init();
            AtomicStore(thlptr, oldthl);

            AtomicStore(&Nodes[numaNode], pernode);
        }

        return pernode;
    }

    void TGlobalAllocator::InitPerThreadAllocator(ui32 numaNode, TPerThreadAllocator** thl) noexcept {
        const ui32 nx = numaNode;
        TPerNodeAllocator* pernode = AtomicLoad(&Nodes[nx]);

        if (pernode == nullptr) {
            TGuard<TSpinLock> guard(&Lock);
            pernode = AtomicLoad(&Nodes[nx]);
            if (pernode == nullptr) {
                pernode = new TPerNodeAllocator(this, numaNode);

                // all allocations during node setup would be handled by incremental pages from boostrap pool
                // there must be no allocation larger then one possibly handled by incremental allocator
                {
                    TPerThreadAllocator tmp;
                    tmp.BeginSetup(pernode);
                    AtomicStore(thl, &tmp);

                    pernode->Init();

                    // now we must replace 'stack-placed' allocator with actual one
                    pernode->ClaimPerThreadAllocator(thl);
                }

                AtomicStore(&Nodes[nx], pernode);
                return;
            }
        }

        TPerThreadAllocator tmp;
        tmp.BeginSetup(pernode);
        AtomicStore(thl, &tmp);

        pernode->ClaimPerThreadAllocator(thl);
    }

    void TGlobalAllocator::PlaceExtMapping(void* x, ui32 pages, ui32 numaNode) {
        ExtMappingsMap.Push(x, pages, numaNode);
    }

    std::pair<ui32, ui32> TGlobalAllocator::ClearExtMapping(void* x) {
        return ExtMappingsMap.Lookup(x, true);
    }

    ui64 TGlobalAllocator::LookupExtSize(void* x) {
        std::pair<ui32, ui32> p = ExtMappingsMap.Lookup(x, false);
        switch (p.first) {
            case 0:
                return 0;
            case 1: {
                ui8* hiddenPage = (ui8*)x - SystemPageSize;
                Y_VERIFY_DEBUG(*(ui32*)hiddenPage == TChunkHeader::HiddenMagicTag);
                return *((ui64*)hiddenPage + 1) - SystemPageSize;
            }
            default:
                return (p.first - 1) * SystemPageSize;
        }
    }

    TVector<TAllocatorStats> TGlobalAllocator::AllocatorStats() {
        TVector<TAllocatorStats> ret(NALF_ALLOC_NUMANODES);
        for (ui64 node = 0; node < NALF_ALLOC_NUMANODES; ++node) {
            if (TPerNodeAllocator* pernode = AtomicLoad(&Nodes[node])) {
                pernode->AllocatorStats(ret[node]);
            }
        }
        return ret;
    }

    TAllocatorStats::TAllocatorStats()
        : TotalBytesReserved(0)
        , PerThreadEntries(0)
    {
    }

    TAllocatorStats::TSizeStats::TSizeStats()
        : PageSize(0)
        , ChunkSize(0)
        , TotalPagesReserved(0)
        , TotalPagesCached(0)
        , TotalAllocations(0)
        , TotalReclaimed(0)
        , PagesClaimed(0)
        , PagesFromCache(0)
        , PagesReleased(0)
    {
    }

    TAllocatorStats::TIncrementalStats::TIncrementalStats()
        : TotalPagesReserved(0)
        , TotalPagesCached(0)
        , TotalAllocations(0)
        , TotalReclaimed(0)
        , PagesClaimed(0)
        , PagesFromCache(0)
        , PagesReleased(0)
    {
    }

    TAllocatorStats::TSysStats::TSysStats()
        : TotalBytesReserved(0)
        , TotalBytesCached(0)
        , TotalAllocations(0)
        , TotalReclaimed(0)
    {
    }

    void TAllocatorStats::Out(IOutputStream& out) const {
        out << "  Total bytes reserved: " << TotalBytesReserved << Endl << Endl;
        out
            << "  Incremental (page size: 32k):" << Endl
            << "    PagesReserved: " << IncrementalStats.TotalPagesReserved << Endl
            << "    PagesCached: " << IncrementalStats.TotalPagesCached << Endl
#if defined NALF_COLLECT_STATS
            << "    Active/Allocated/Reclaimed: "
            << i64(IncrementalStats.TotalAllocations) - i64(IncrementalStats.TotalReclaimed) << "/"
            << IncrementalStats.TotalAllocations << "/" << IncrementalStats.TotalReclaimed
            << Endl
            << "    Pages Claimed/Released/FromCache: "
            << IncrementalStats.PagesClaimed << "/"
            << IncrementalStats.PagesReleased << "/"
            << IncrementalStats.PagesFromCache
            << Endl
#endif
            << Endl;

        for (auto& x : BySizeStats) {
            out << "  Chunked, sz: " << x.ChunkSize << " (page sz: " << x.PageSize << ")" << Endl;
            if (x.TotalAllocations)
                out
                    << "    PagesReserved: " << x.TotalPagesReserved << Endl
                    << "    PagesCached: " << x.TotalPagesCached << Endl
#if defined NALF_COLLECT_STATS
                    << "    Active/Allocated/Reclaimed: "
                    << i64(x.TotalAllocations) - i64(x.TotalReclaimed) << "/"
                    << x.TotalAllocations << "/" << x.TotalReclaimed
                    << Endl
                    << "    Pages Claimed/Released/FromCache: "
                    << x.PagesClaimed << "/"
                    << x.PagesReleased << "/"
                    << x.PagesFromCache
                    << Endl;
#endif
            out << Endl;
        }

        out
            << "  System:" << Endl
            << "  BytesReserved: " << SysStats.TotalBytesReserved << Endl
            << "  BytesCached: " << SysStats.TotalBytesCached << Endl
            //<< "  Active/Allocated/Reclaimed: "
            //                            << SysStats.TotalAllocations - SysStats.TotalReclaimed << "/"
            //                            << SysStats.TotalAllocations << "/" << SysStats.TotalReclaimed;
            << Endl << Endl;
    }

    void* SystemAllocation(ui64 size) {
        void* mem = nullptr;
#ifdef _win_
        mem = VirtualAlloc(0, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        Y_VERIFY(mem != 0, "mmap failed to allocated %" PRIu64 " bytes", size);
#else
        mem = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, 0, 0);
        Y_VERIFY(mem != (void*)-1, "mmap failed to allocated %" PRIu64 " bytes", size);
#endif
        return mem;
    }

    void SystemFree(void* mem, ui64 size) {
#ifdef _win_
        Y_UNUSED(size);
        const bool success = VirtualFree(mem, 0, MEM_RELEASE);
        Y_VERIFY(success, "system memory decommit failed, must never happens in sane program");
#else
        const bool success = 0 == munmap(mem, size);
        if (!success)
            Y_FAIL("munmap failed, must never happens is sane program, errno: %d", errno);
#endif
    }

    void* SystemRemap(void* mem, ui64 oldsize, ui64 newsize) {
        // TODO: utilize remap when possible
        void* newmem = SystemAllocation(newsize);
        memcpy(mem, newmem, Max(oldsize, newsize));
        SystemFree(mem, oldsize);
        return newmem;
    }

    // per-thread allocator

    TPerThreadAllocator::TPerThreadAllocator(volatile bool* releasedFlag /* = nullptr */)
        : Node(nullptr)
        , NumaNodeId(Max<ui32>())
        , Hint(TAllocHint::Bootstrap)
        , ReleasedFlag(releasedFlag)
    {
        for (TChunkHeader::EChunkType i = TChunkHeader::Chunk16; i <= TChunkHeader::Chunk6528; i = (TChunkHeader::EChunkType)(i + 1)) {
            CacheChunked[i].ChunkSize = TChunkHeader::TypeToSize(i);
            CacheChunked[i].FreeChunksToRelease = TChunkHeader::FreeChunksToReleasePage(i);
        }

        CacheChunked[TChunkHeader::Chunk16].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk32].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk48].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk64].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk96].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk128].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk192].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk256].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk384].LengthToRecycle = 17;
        CacheChunked[TChunkHeader::Chunk512].LengthToRecycle = 9;
        CacheChunked[TChunkHeader::Chunk768].LengthToRecycle = 9;
        CacheChunked[TChunkHeader::Chunk1024].LengthToRecycle = 5;
        CacheChunked[TChunkHeader::Chunk1536].LengthToRecycle = 5;
        CacheChunked[TChunkHeader::Chunk2176].LengthToRecycle = 3;
        CacheChunked[TChunkHeader::Chunk3584].LengthToRecycle = 1;
        CacheChunked[TChunkHeader::Chunk6528].LengthToRecycle = 1;
    }

    TPerThreadAllocator::~TPerThreadAllocator(){
        // per-thread allocators are never disposed, just reused
    };

    void TPerThreadAllocator::BeginSetup(TPerNodeAllocator* node) {
        Node = node;
        NumaNodeId = Node->NumaNode;
    }

    void TPerThreadAllocator::CompleteSetup() {
        Hint = TAllocHint::Undefined;
    }

    void TPerThreadAllocator::EnsureNumaNode() {
        const ui32 nx = GetNumaNode();
        if (nx != NumaNodeId && Hint != TAllocHint::Bootstrap) {
            Node = GlobalAllocator()->PerNodeAllocator(nx);
            NumaNodeId = nx;
        }
    }

    TAllocHint::EHint TPerThreadAllocator::SwapHint(TAllocHint::EHint hint) {
        Y_VERIFY_DEBUG(hint < TAllocHint::Bootstrap);
        if (Y_LIKELY(Hint != TAllocHint::Bootstrap)) {
            const TAllocHint::EHint ret = Hint;
            Hint = hint;
            return ret;
        } else {
            return TAllocHint::Undefined;
        }
    }

    void* TPerThreadAllocator::Allocate(ui64 len, TAllocHint::EHint hint, ui64 align) {
        Y_VERIFY_DEBUG((align & (align - 1)) == 0); // align must be power of 2
        Y_VERIFY_DEBUG(ReleasedFlag == nullptr || AtomicLoad(ReleasedFlag) == false);

        switch (Hint) {
            case TAllocHint::Undefined:
                switch (hint) {
                    case TAllocHint::Undefined:
                        return AllocateChunked(len, align);
                    case TAllocHint::Incremental:
                        return AllocateIncremental(len, align);
                    case TAllocHint::Chunked:
                        return AllocateChunked(len, align);
                    case TAllocHint::System:
                        return AllocateSystem(len, align);
                    case TAllocHint::ForceIncremental:
                        return AllocateIncremental(len, align);
                    case TAllocHint::ForceChunked:
                        return AllocateChunked(len, align);
                    case TAllocHint::ForceSystem:
                        return AllocateSystem(len, align);
                    default:
                        Y_VERIFY_DEBUG(false);
                        return nullptr;
                }
            case TAllocHint::Incremental:
                switch (hint) {
                    case TAllocHint::Undefined:
                    case TAllocHint::Incremental:
                    case TAllocHint::Chunked:
                    case TAllocHint::System:
                        return AllocateIncremental(len, align);
                    case TAllocHint::ForceIncremental:
                        return AllocateIncremental(len, align);
                    case TAllocHint::ForceChunked:
                        return AllocateChunked(len, align);
                    case TAllocHint::ForceSystem:
                        return AllocateSystem(len, align);
                    default:
                        Y_VERIFY_DEBUG(false);
                        return nullptr;
                }
            case TAllocHint::Chunked:
                switch (hint) {
                    case TAllocHint::Undefined:
                    case TAllocHint::Incremental:
                    case TAllocHint::Chunked:
                    case TAllocHint::System:
                        return AllocateChunked(len, align);
                    case TAllocHint::ForceIncremental:
                        return AllocateIncremental(len, align);
                    case TAllocHint::ForceChunked:
                        return AllocateChunked(len, align);
                    case TAllocHint::ForceSystem:
                        return AllocateSystem(len, align);
                    default:
                        Y_VERIFY_DEBUG(false);
                        return nullptr;
                }
            case TAllocHint::System:
                switch (hint) {
                    case TAllocHint::Undefined:
                    case TAllocHint::Incremental:
                    case TAllocHint::Chunked:
                    case TAllocHint::System:
                        return AllocateSystem(len, align);
                    case TAllocHint::ForceIncremental:
                        return AllocateIncremental(len, align);
                    case TAllocHint::ForceChunked:
                        return AllocateChunked(len, align);
                    case TAllocHint::ForceSystem:
                        return AllocateSystem(len, align);
                    default:
                        Y_VERIFY_DEBUG(false);
                        return nullptr;
                }
            case TAllocHint::Bootstrap:
                return AllocateIncremental(len, align);
            default:
                Y_VERIFY_DEBUG(false);
                return nullptr;
        }
    }

    void* TPerThreadAllocator::AllocateSystem(ui64 len, ui64 align) {
        Y_VERIFY_DEBUG(align <= 4096);
        EnsureNumaNode();
        return Node->ClaimSysPage(len);
    }

    void TPerThreadAllocator::MarkIncrementalRecycle(TChunkHeader* header, ui32 sub) {
        const ui32 left = AtomicUi32Sub(&header->Counter, sub);
        if (left == 0) {
            if (CacheIncremental.Next == nullptr && header->NumaNode == NumaNode())
                CacheIncremental.Next = header;
            else {
#if defined NALF_COLLECT_STATS
                RelaxedStore(&IncrementalStats.PagesReleased, IncrementalStats.PagesReleased + 1);
#endif
                Node->RecycleChunk(header);
            }
        }
    }

    void* TPerThreadAllocator::AllocateIncremental(ui64 len, ui64 align) {
        ui64 sz = AlignUp<ui64>(len, align) + 2 + 2 * align; // could be overcommit, but that's is simplest form of check
        if (sz > TChunkHeader::MaxIncrementalAllocation) {
            Y_VERIFY_DEBUG(Hint != TAllocHint::Bootstrap);
            return AllocateSystem(len, align);
        }

        if (CacheIncremental.CurrentPosition + sz >= TChunkHeader::LargeChunkSize) {
            Y_VERIFY_DEBUG(CacheIncremental.Current != nullptr);
            const ui32 sub = TChunkHeader::LargeChunkSize - CacheIncremental.CurrentAllocated;

            TChunkHeader* current = CacheIncremental.Current;
            CacheIncremental.Current = nullptr;
            CacheIncremental.CurrentPosition = 0;

            MarkIncrementalRecycle(current, sub);
        }

        if (CacheIncremental.Current == nullptr) {
            if (CacheIncremental.Next != nullptr) {
                CacheIncremental.Current = CacheIncremental.Next;
                CacheIncremental.Next = nullptr;

#if defined NALF_COLLECT_STATS
                RelaxedStore(&IncrementalStats.PagesFromCache, IncrementalStats.PagesFromCache + 1);
#endif
            } else {
                EnsureNumaNode();
                CacheIncremental.Current = Node->ClaimChunk(TChunkHeader::ChunkIncremental);

#if defined NALF_COLLECT_STATS
                RelaxedStore(&IncrementalStats.PagesClaimed, IncrementalStats.PagesClaimed + 1);
#endif
            }

            AtomicUi32Add(&CacheIncremental.Current->Counter, TChunkHeader::LargeChunkSize); // lock page
            CacheIncremental.CurrentPosition = 64;
            CacheIncremental.CurrentAllocated = 0;
        }

        // ok, we prepared page,
        // now allocate continuous chunk and prepend with size.

        const ui32 cposold = CacheIncremental.CurrentPosition;
        ui32 cposnew = (ui32)AlignUp<ui64>(cposold, align);
        while (cposnew - cposold < sizeof(ui16) || (cposnew % TChunkHeader::SmallChunkSize) == 0)
            cposnew += align;

        ui8* const ret = (ui8*)CacheIncremental.Current + cposnew;
        AtomicStore(((ui16*)ret - 1), (ui16)len);

        CacheIncremental.CurrentPosition = (ui32)(cposnew + len);
        ++CacheIncremental.CurrentAllocated;

#if defined NALF_COLLECT_STATS
        RelaxedStore(&IncrementalStats.TotalAllocations, IncrementalStats.TotalAllocations + 1);
#endif

        return ret;
    }

    void TPerThreadAllocator::MarkChunkRecycle(TChunkHeader* header, TCacheStats* stats, ui32 freeChunksToRecycle, ui32 sub) {
        const ui32 cx = AtomicUi32Sub(&header->Counter, sub);
        if (cx <= freeChunksToRecycle && cx + sub > freeChunksToRecycle) {
#if defined NALF_COLLECT_STATS
            RelaxedStore(&stats->PagesReleased, stats->PagesReleased + 1);
#else
            Y_UNUSED(stats);
#endif
            Node->RecycleChunk(header);
        }
    }

    void* TPerThreadAllocator::AllocateChunkedCached(TChunkHeader::EChunkType chunkType) {
        TChunkCache& cache = CacheChunked[chunkType];

#if defined NALF_COLLECT_STATS
        TCacheStats& stats = ChunkedStats[chunkType];
        RelaxedStore(&stats.TotalAllocations, stats.TotalAllocations + 1);
#endif

        if (cache.CurrentUsed == 0) {
            if (cache.Current != nullptr) {
                if (cache.IsCurrentOwner) {
                    if (cache.CurrentAllocated) { // flush allocated counter before next chunk
                        Y_VERIFY_DEBUG(cache.CurrentAllocated < TChunkCache::CurrentCacheLength);
                        const ui32 updatedCounter = AtomicUi32Add(&cache.Current->Counter, cache.CurrentAllocated);
                        Y_VERIFY_DEBUG(updatedCounter >= TChunkHeader::LargeChunkSize);
                        Y_VERIFY_DEBUG(updatedCounter < 2 * TChunkHeader::LargeChunkSize);
                        Y_UNUSED(updatedCounter);
                        cache.CurrentAllocated = 0;
                    }

                    // try to pop something from same page
                    const ui32 received = cache.Current->PopBulk(cache.CurrentCache, TChunkCache::CurrentCacheLength);
                    if (received > 0) {
                        cache.CurrentUsed = received;
                    } else {
                        const ui32 counter = AtomicLoad(&cache.Current->Counter);
                        Y_VERIFY_DEBUG(counter >= TChunkHeader::LargeChunkSize);
                        Y_VERIFY_DEBUG(counter < 2 * TChunkHeader::LargeChunkSize);

#if defined NALF_COLLECT_STATS
                        MarkChunkRecycle(cache.Current, &stats, cache.FreeChunksToRelease, TChunkHeader::LargeChunkSize);
#else
                        MarkChunkRecycle(cache.Current, nullptr, cache.FreeChunksToRelease, TChunkHeader::LargeChunkSize);
#endif

                        cache.Current = nullptr;
                    }
                } else {
                    // we are not owner of current page, release page
                    // and those is all what we can do
                    cache.Current = nullptr;
                }
            }

            // still have no page, acquire one
            if (cache.Current == nullptr) {
                // first - try recycle something from deletion-cache (if long enough chunk from right numa-node is there)
                ui32 lidx = 0;
                ui32 length = 0;
                const ui32 expectedNumaNode = NumaNode();
                for (ui32 i = 0; i < cache.DelCaches; ++i) {
                    if (cache.Used[i] > length && cache.Pages[i]->NumaNode == expectedNumaNode) {
                        length = cache.Used[i];
                        lidx = i;
                    }
                }

                if (length >= cache.LengthToRecycle) {
                    cache.Current = cache.Pages[lidx];
                    cache.Pages[lidx] = nullptr;
                    cache.Used[lidx] = 0;
                    memcpy(cache.CurrentCache, cache.Cache[lidx], length * sizeof(cache.CurrentCache[0]));
                    cache.IsCurrentOwner = false;
                    cache.CurrentAllocated = 0;
                    cache.CurrentUsed = length;

#if defined NALF_COLLECT_STATS
                    RelaxedStore(&stats.PagesFromCache, stats.PagesFromCache + 1);
#endif
                } else {
                    EnsureNumaNode();
                    TChunkHeader* next = Node->ClaimChunk(chunkType);
                    const ui32 updatedCounter = AtomicUi32Add(&next->Counter, TChunkHeader::LargeChunkSize);
                    Y_VERIFY_DEBUG(updatedCounter < 2 * TChunkHeader::LargeChunkSize);
                    Y_UNUSED(updatedCounter);
                    const ui32 received = next->PopBulk(cache.CurrentCache, cache.CurrentCacheLength);
                    Y_VERIFY_DEBUG(received > 0);
                    cache.Current = next;
                    cache.IsCurrentOwner = true;
                    cache.CurrentAllocated = 0;
                    cache.CurrentUsed = received;

#if defined NALF_COLLECT_STATS
                    RelaxedStore(&stats.PagesClaimed, stats.PagesClaimed + 1);
#endif
                }
            }

            Y_VERIFY_DEBUG(cache.CurrentUsed > 0);
        }

        --cache.CurrentUsed;
        ++cache.CurrentAllocated;
        const ui32 sz = cache.ChunkSize;
        const ui16 idx = cache.CurrentCache[cache.CurrentUsed];
        void* const ret = TChunkHeader::PtrFromIdx(cache.Current, sz, idx);

#ifdef NALF_ALLOC_DEBUG
        for (ui64 *x = (ui64*)(ret), *end = x + sz / 8; x != end; ++x)
            Y_VERIFY(*x == FreeMemoryMark, "FreeMemoryMark corruption at: %" PRIx64 " ChunkSize: %" PRIu64,
                     (ui64)(intptr_t)x, (ui64)cache.ChunkSize);
#endif

        return ret;
    }

    void* TPerThreadAllocator::AllocateChunked(ui64 len, ui64 align) {
        // todo: align on 64
        static const ui8 szToIdx[64] = {
            // first 9 values are indexed by sh16 [0-128]
            TChunkHeader::Chunk16, TChunkHeader::Chunk16, // 1
            TChunkHeader::Chunk32,
            TChunkHeader::Chunk48,
            TChunkHeader::Chunk64,
            TChunkHeader::Chunk96, TChunkHeader::Chunk96,   // 6
            TChunkHeader::Chunk128, TChunkHeader::Chunk128, // 8

            // next values are indexed by sh64 [129-3584]
            TChunkHeader::Chunk192, TChunkHeader::Chunk192,                                                     // 10
            TChunkHeader::Chunk256,                                                                             // 11
            TChunkHeader::Chunk384, TChunkHeader::Chunk384,                                                     // 13
            TChunkHeader::Chunk512, TChunkHeader::Chunk512,                                                     // 15
            TChunkHeader::Chunk768, TChunkHeader::Chunk768, TChunkHeader::Chunk768, TChunkHeader::Chunk768,     // 19
            TChunkHeader::Chunk1024, TChunkHeader::Chunk1024, TChunkHeader::Chunk1024, TChunkHeader::Chunk1024, // 23

            TChunkHeader::Chunk1536, TChunkHeader::Chunk1536, TChunkHeader::Chunk1536, TChunkHeader::Chunk1536,
            TChunkHeader::Chunk1536, TChunkHeader::Chunk1536, TChunkHeader::Chunk1536, TChunkHeader::Chunk1536, // 31

            TChunkHeader::Chunk2176, TChunkHeader::Chunk2176, TChunkHeader::Chunk2176, TChunkHeader::Chunk2176,
            TChunkHeader::Chunk2176, TChunkHeader::Chunk2176, TChunkHeader::Chunk2176, TChunkHeader::Chunk2176,
            TChunkHeader::Chunk2176, TChunkHeader::Chunk2176, // 41

            TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584,
            TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584,
            TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584,
            TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584,
            TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584, TChunkHeader::Chunk3584,
            TChunkHeader::Chunk3584, TChunkHeader::Chunk3584 // 63
        };

        if (len > 6528 || ((len % TChunkHeader::SmallChunkSize) == 0))
            return AllocateSystem(len, align);

        const ui64 szx16 = (len + 15) / 16;
        const ui64 szx64 = (len + 448 + 63) / 64;

        if (len <= 128)
            return AllocateChunkedCached((TChunkHeader::EChunkType)szToIdx[szx16]);
        else if (len <= 3584)
            return AllocateChunkedCached((TChunkHeader::EChunkType)szToIdx[szx64]);
        else if (len <= 4096)
            return AllocateSystem(len, align);
        else
            return AllocateChunkedCached(TChunkHeader::Chunk6528);
    }

    void TPerThreadAllocator::FreeIncremental(TChunkHeader* header) {
#if defined NALF_COLLECT_STATS
        RelaxedStore(&IncrementalStats.TotalReclaimed, IncrementalStats.TotalReclaimed + 1);
#endif

        if (CacheIncremental.ToClean == header) {
            ++CacheIncremental.ToCleanCounter;
        } else {
            TChunkHeader* const toClean = CacheIncremental.ToClean;
            const ui32 toCleanCounter = CacheIncremental.ToCleanCounter;

            CacheIncremental.ToClean = header;
            CacheIncremental.ToCleanCounter = 1;

            if (toClean)
                MarkIncrementalRecycle(toClean, toCleanCounter);
        }
    }

    void TPerThreadAllocator::FreeChunked(TChunkHeader* header, void* x) {
        const ui32 chunkType = header->ChunkType;
        TChunkCache& cache = CacheChunked[chunkType];
#if defined NALF_COLLECT_STATS
        TCacheStats& stats = ChunkedStats[chunkType];
        RelaxedStore(&stats.TotalReclaimed, stats.TotalReclaimed + 1);
#endif

        const ui32 sz = cache.ChunkSize;
        const ui16 idx = TChunkHeader::IdxFromPtr(header, sz, x);

#ifdef NALF_ALLOC_DEBUG
        std::fill_n((ui64*)(x), sz / 8, FreeMemoryMark);
#endif

        if (cache.Current == header) {
            // page active right now and we have space in cache, just place there
            Y_VERIFY_DEBUG(cache.CurrentUsed < TChunkCache::CurrentCacheLength);
            cache.CurrentCache[cache.CurrentUsed] = idx;
            ++cache.CurrentUsed;
            --cache.CurrentAllocated;
            return;
        }

        ui32 emptyCacheIdx = Max<ui32>();
        ui32 oldestCacheIdx = 0;
        ui64 oldestRecycleIndex = 0;

        for (ui32 i = 0; i != cache.DelCaches; ++i) {
            if (cache.Pages[i] == header) {
                // page is already in cache, use this entry
                ui32 udx = cache.Used[i];
                cache.Cache[i][udx] = idx;
                ++udx;
                if (udx == TChunkCache::DelCacheLength) {
                    // flush from cache
                    const ui16* dcache = cache.Cache[i];
                    header->PushBulk(dcache, TChunkCache::DelCacheLength);

#if defined NALF_COLLECT_STATS
                    MarkChunkRecycle(header, &stats, cache.FreeChunksToRelease, TChunkCache::DelCacheLength);
#else
                    MarkChunkRecycle(header, nullptr, cache.FreeChunksToRelease, TChunkCache::DelCacheLength);
#endif

                    cache.Pages[i] = nullptr;
                    cache.Used[i] = 0;
                } else {
                    cache.RecycleIndex[i] = ++cache.RecycleCounter;
                    cache.Used[i] = udx;
                }
                return;
            }

            if (cache.Pages[i] == nullptr) {
                emptyCacheIdx = i; // found empty cache entry, mark for later use
            } else if (cache.RecycleIndex[i] > oldestRecycleIndex) {
                oldestRecycleIndex = cache.RecycleIndex[i];
                oldestCacheIdx = i;
            }
        }

        if (emptyCacheIdx == Max<ui32>()) {
            // long path - flush oldest of cached pages
            // todo: some sort of heuristics to keep long tails for potential reuse
            TChunkHeader* pageToRecycle = cache.Pages[oldestCacheIdx];
            const ui32 udx = cache.Used[oldestCacheIdx];
            const ui16* dcache = cache.Cache[oldestCacheIdx];

            pageToRecycle->PushBulk(dcache, udx);

#if defined NALF_COLLECT_STATS
            MarkChunkRecycle(pageToRecycle, &stats, cache.FreeChunksToRelease, udx);
#else
            MarkChunkRecycle(pageToRecycle, nullptr, cache.FreeChunksToRelease, udx);
#endif

            emptyCacheIdx = oldestCacheIdx;
        }

        // short path - move everything on free place
        cache.Pages[emptyCacheIdx] = header;
        cache.Cache[emptyCacheIdx][0] = idx;
        cache.Used[emptyCacheIdx] = 1;
        cache.RecycleIndex[emptyCacheIdx] = ++cache.RecycleCounter;
    }

    void TPerThreadAllocator::Free(void* mem) {
        ui64 x = (ui64)mem;
        if ((x % TChunkHeader::SmallChunkSize) == 0) {
            Node->RecycleSysPage(mem);
        } else {
            Y_VERIFY_DEBUG(ReleasedFlag == nullptr || AtomicLoad(ReleasedFlag) == false);
            TChunkHeader* largeChunkHeader = (TChunkHeader*)(x & ~(TChunkHeader::LargeChunkSize - 1));
            Y_VERIFY_DEBUG(largeChunkHeader->Magic == TChunkHeader::MagicTag);
            const TChunkHeader::EChunkType chunkType = (TChunkHeader::EChunkType)largeChunkHeader->ChunkType;
            if (chunkType == TChunkHeader::ChunkIncremental) {
#ifdef NALF_ALLOC_DEBUG
                {
                    ui16* sz = ((ui16*)mem - 1);
                    Y_VERIFY(*sz != FreeMemoryMarkInc);
                    memset(mem, FreeMemoryMarkInc, *sz);
                    *sz = FreeMemoryMarkInc;
                }
#endif
                FreeIncremental(largeChunkHeader);
            } else if (TChunkHeader::TypeToSubpageSize(chunkType) == TChunkHeader::LargeChunkSize) {
                FreeChunked(largeChunkHeader, mem);
            } else {
                TChunkHeader* smallChunkHeader = (TChunkHeader*)(x & ~(TChunkHeader::SmallChunkSize - 1));
                Y_VERIFY_DEBUG(smallChunkHeader->Magic == TChunkHeader::MagicTag && smallChunkHeader->ChunkType == (ui32)chunkType);
                FreeChunked(smallChunkHeader, mem);
            }
        }
    }

    void* TPerThreadAllocator::Realloc(void* mem, ui64 len) {
        if (mem == nullptr)
            return Allocate(len, TAllocHint::Undefined, 16);

        if (len == 0) {
            Free(mem);
            return nullptr;
        }

        const std::pair<ui64, TAllocHint::EHint> oldsz = MemBlockSize(mem);
        Y_VERIFY(oldsz.first > 0);

        void* newptr = Allocate(len, (TAllocHint::EHint)(oldsz.second + 3), 16);
        memcpy(newptr, mem, ((oldsz.first < len) ? oldsz.first : len));
        Free(mem);
        return newptr;
    }

    ui32 TPerThreadAllocator::NumaNode() const {
        return NumaNodeId;
    }

}

#if defined NALF_DEFINE_GLOBALS
// hooks, mostly copied from lfalloc
#if !defined NALF_DONOT_DEFINE_GLOBALS

extern "C" void* malloc(size_t size) {
    return NNumaAwareLockFreeAllocator::Allocate(size);
}

extern "C" void free(void* data) {
    NNumaAwareLockFreeAllocator::Free(data);
}

extern "C" int posix_memalign(void** memptr, size_t alignment, size_t size) {
    void* x = NNumaAwareLockFreeAllocator::Allocate(size, NNumaAwareLockFreeAllocator::TAllocHint::Undefined, alignment);
    *memptr = x;
    return 0;
}

extern "C" void* memalign(size_t alignment, size_t size) {
    void* ptr;
    int res = posix_memalign(&ptr, alignment, size);
    return res ? 0 : ptr;
}

extern "C" void* valloc(size_t size) {
    return memalign(NNumaAwareLockFreeAllocator::SystemPageSize, size);
}

#if !defined(_MSC_VER) && !defined(_freebsd_)
// Workaround for pthread_create bug in linux.
extern "C" void* __libc_memalign(size_t alignment, size_t size) {
    return memalign(alignment, size);
}
#endif

extern "C" void* calloc(size_t n, size_t elem_size) {
    const size_t size = n * elem_size;
    void* x = NNumaAwareLockFreeAllocator::Allocate(size);
    memset(x, 0, size);
    return x;
}

extern "C" void cfree(void* ptr) {
    NNumaAwareLockFreeAllocator::Free(ptr);
}

extern "C" void* realloc(void* oldPtr, size_t newSize) {
    return NNumaAwareLockFreeAllocator::Realloc(oldPtr, newSize);
}

#if defined(USE_INTELCC) || defined(_darwin_) || defined(_freebsd_) || defined(_STLPORT_VERSION)
#define OP_THROWNOTHING throw ()
#define OP_THROWBADALLOC throw (std::bad_alloc)
#else
#define OP_THROWNOTHING
#define OP_THROWBADALLOC
#endif

#if !defined(YMAKE)
void* operator new(size_t size) OP_THROWBADALLOC {
    return NNumaAwareLockFreeAllocator::Allocate(size);
}

void* operator new(size_t size, const std::nothrow_t&) OP_THROWNOTHING {
    return NNumaAwareLockFreeAllocator::Allocate(size);
}

void operator delete(void* p)OP_THROWNOTHING {
    NNumaAwareLockFreeAllocator::Free(p);
}

void operator delete(void* p, const std::nothrow_t&)OP_THROWNOTHING {
    NNumaAwareLockFreeAllocator::Free(p);
}

void* operator new[](size_t size) OP_THROWBADALLOC {
    return NNumaAwareLockFreeAllocator::Allocate(size);
}

void* operator new[](size_t size, const std::nothrow_t&) OP_THROWNOTHING {
    return NNumaAwareLockFreeAllocator::Allocate(size);
}

void operator delete[](void* p) OP_THROWNOTHING {
    NNumaAwareLockFreeAllocator::Free(p);
}

void operator delete[](void* p, const std::nothrow_t&) OP_THROWNOTHING {
    NNumaAwareLockFreeAllocator::Free(p);
}
#endif

#if defined(_MSC_VER) && !defined(_DEBUG)
extern "C" size_t _msize(void* memblock) {
    return NNumaAwareLockFreeAllocator::MemBlockSize(memblock).first;
}

// MSVC runtime doesn't call memory functions directly.
// It uses functions with _base suffix instead.
extern "C" void* _calloc_base(size_t num, size_t size) {
    return calloc(num, size);
}

extern "C" void* _realloc_base(void* old_ptr, size_t new_size) {
    return realloc(old_ptr, new_size);
}

extern "C" void* _malloc_base(size_t new_size) {
    return SafeMalloc(new_size);
}

extern "C" void _free_base(void* ptr) {
    return LFFree(ptr);
}
#endif

#endif
#endif
//x
