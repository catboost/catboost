#pragma once

// unique tag to fix pragma once gcc glueing: ./kikimr/core/nalf_alloc/defs.h
#include <util/system/defaults.h>
#include <util/generic/noncopyable.h>
#include <util/system/atomic.h>
#include <util/system/align.h>
#include <util/generic/utility.h>
#include <util/generic/ylimits.h>
#include <util/system/spinlock.h>

#if !defined NDEBUG && !defined NALF_ALLOC_DEBUG
#define NALF_ALLOC_DEBUG
#endif

#if !defined NALF_NOT_COLLECT_STATS
#define NALF_COLLECT_STATS
#endif

#ifndef NALF_ALLOC_EXTMAPPINGS_BASE
#define NALF_ALLOC_EXTMAPPINGS_BASE (2 * 1024 * 1024) // 16 mb hashtable gives 2m elements
#endif

#ifndef NALF_ALLOC_NUMANODES
#define NALF_ALLOC_NUMANODES 4
#endif

#ifndef NALF_ALLOC_MAXMAPTRACKED
#define NALF_ALLOC_MAXMAPTRACKED 16
#endif

#ifndef NALF_ALLOC_LINKEDMAPSIZE
#define NALF_ALLOC_LINKEDMAPSIZE 5
#endif

#ifndef NALF_CACHE_4K_PAGES
#define NALF_CACHE_4K_PAGES 16384
#endif

#ifndef NALF_CACHE_4K_POOL
#define NALF_ALLOC_CACHE_4K_POOL 3
#endif

#ifndef NALF_CACHE_8K_PAGES
#define NALF_CACHE_8K_PAGES 8192
#endif

#ifndef NALF_CACHE_8K_POOL
#define NALF_ALLOC_CACHE_8K_POOL 3
#endif

#ifndef NALF_CACHE_12K_PAGES
#define NALF_CACHE_12K_PAGES 2048
#endif

#ifndef NALF_CACHE_16K_PAGES
#define NALF_CACHE_16K_PAGES 2048
#endif

#ifndef NALF_CACHE_16K_POOL
#define NALF_ALLOC_CACHE_16K_POOL 3
#endif

#ifndef NALF_CACHE_24K_PAGES
#define NALF_CACHE_24K_PAGES 2048
#endif

#ifndef NALF_CACHE_32K_PAGES
#define NALF_CACHE_32K_PAGES 2048
#endif

#ifndef NALF_CACHE_32K_POOL
#define NALF_ALLOC_CACHE_32K_POOL 3
#endif

#ifndef NALF_CACHE_64K_PAGES
#define NALF_CACHE_64K_PAGES 512
#endif

#ifndef NALF_ALLOC_CACHE16_POOLS
#define NALF_ALLOC_CACHE16_POOLS 2
#define NALF_ALLOC_CACHE16_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE32_POOLS
#define NALF_ALLOC_CACHE32_POOLS 2
#define NALF_ALLOC_CACHE32_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE48_POOLS
#define NALF_ALLOC_CACHE48_POOLS 2
#define NALF_ALLOC_CACHE48_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE64_POOLS
#define NALF_ALLOC_CACHE64_POOLS 2
#define NALF_ALLOC_CACHE64_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE96_POOLS
#define NALF_ALLOC_CACHE96_POOLS 2
#define NALF_ALLOC_CACHE96_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE128_POOLS
#define NALF_ALLOC_CACHE128_POOLS 2
#define NALF_ALLOC_CACHE128_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE192_POOLS
#define NALF_ALLOC_CACHE192_POOLS 2
#define NALF_ALLOC_CACHE192_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE256_POOLS
#define NALF_ALLOC_CACHE256_POOLS 2
#define NALF_ALLOC_CACHE256_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE384_POOLS
#define NALF_ALLOC_CACHE384_POOLS 2
#define NALF_ALLOC_CACHE384_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE512_POOLS
#define NALF_ALLOC_CACHE512_POOLS 2
#define NALF_ALLOC_CACHE512_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE768_POOLS
#define NALF_ALLOC_CACHE768_POOLS 2
#define NALF_ALLOC_CACHE768_POOLSV \
    { 3, 3 }
#endif

#ifndef NALF_ALLOC_CACHE1024_POOLS
#define NALF_ALLOC_CACHE1024_POOLS 1
#define NALF_ALLOC_CACHE1024_POOLSV \
    { 3 }
#endif

#ifndef NALF_ALLOC_CACHE1536_POOLS
#define NALF_ALLOC_CACHE1536_POOLS 1
#define NALF_ALLOC_CACHE1536_POOLSV \
    { 3 }
#endif

#ifndef NALF_ALLOC_CACHE2176_POOLS
#define NALF_ALLOC_CACHE2176_POOLS 1
#define NALF_ALLOC_CACHE2176_POOLSV \
    { 3 }
#endif

#ifndef NALF_ALLOC_CACHE3584_POOLS
#define NALF_ALLOC_CACHE3584_POOLS 1
#define NALF_ALLOC_CACHE3584_POOLSV \
    { 3 }
#endif

#ifndef NALF_ALLOC_CACHE6528_POOLS
#define NALF_ALLOC_CACHE6528_POOLS 1
#define NALF_ALLOC_CACHE6528_POOLSV \
    { 3 }
#endif

#ifndef NALF_ALLOC_CACHEINC_POOLS
#define NALF_ALLOC_CACHEINC_POOLS 3
#define NALF_ALLOC_CACHEINC_POOLSV \
    { 1, 2, 3 }
#endif

// we need explicit 32 bit operations to keep cache-line friendly packs
// so have to define some atomics additionaly to arcadia one
#ifdef _win_
#pragma intrinsic(_InterlockedCompareExchange)
#pragma intrinsic(_InterlockedExchangeAdd)
#pragma intrinsic(_InterlockedIncrement)
#pragma intrinsic(_InterlockedDecrement)
#endif

namespace NNumaAwareLockFreeAllocator {
    static const ui64 MaxTrackedSysPages = (1ull << NALF_ALLOC_MAXMAPTRACKED);

    inline bool AtomicUi32Cas(volatile ui32* a, ui32 exchange, ui32 compare) {
#ifdef _win_
        return _InterlockedCompareExchange((volatile long*)a, exchange, compare) == (long)compare;
#else
        ui32 expected = compare;
        return __atomic_compare_exchange_n(a, &expected, exchange, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
#endif
    }

    inline ui32 AtomicUi32Add(volatile ui32* a, ui32 add) {
#ifdef _win_
        return _InterlockedExchangeAdd((volatile long*)a, add) + add;
#else
        return __atomic_add_fetch(a, add, __ATOMIC_SEQ_CST);
#endif
    }

    inline ui32 AtomicUi32Sub(volatile ui32* a, ui32 sub) {
#ifdef _win_
        return _InterlockedExchangeAdd((volatile long*)a, -(long)sub) - sub;
#else
        return __atomic_sub_fetch(a, sub, __ATOMIC_SEQ_CST);
#endif
    }

    template <typename T>
    inline void AtomicStore(volatile T* a, T x) {
        static_assert(std::is_integral<T>::value || std::is_pointer<T>::value, "expect std::is_integral<T>::value || std::is_pointer<T>::value");
#ifdef _win_
        *a = x;
#else
        __atomic_store_n(a, x, __ATOMIC_RELEASE);
#endif
    }

    template <typename T>
    inline void RelaxedStore(volatile T* a, T x) {
        static_assert(std::is_integral<T>::value || std::is_pointer<T>::value, "expect std::is_integral<T>::value || std::is_pointer<T>::value");
#ifdef _win_
        *a = x;
#else
        __atomic_store_n(a, x, __ATOMIC_RELAXED);
#endif
    }

    template <typename T>
    inline T AtomicLoad(volatile T* a) {
#ifdef _win_
        return *a;
#else
        return __atomic_load_n(a, __ATOMIC_ACQUIRE);
#endif
    }

    template <typename T>
    inline T RelaxedLoad(volatile T* a) {
#ifdef _win_
        return *a;
#else
        return __atomic_load_n(a, __ATOMIC_RELAXED);
#endif
    }

    // copied from <kikimr/core/util/queue_chunk.h> to eliminate dependency
    template <typename T, ui32 TSize, typename TDerived>
    struct TQueueChunkDerived {
        static const ui32 EntriesCount = (TSize - sizeof(TQueueChunkDerived*)) / sizeof(T);
        static_assert(EntriesCount > 0, "expect EntriesCount > 0");

        volatile T Entries[EntriesCount];
        TDerived* volatile Next;

        TQueueChunkDerived() {
            memset(this, 0, sizeof(TQueueChunkDerived));
        }
    };

    template <typename T, ui32 TSize>
    struct TQueueChunk {
        static const ui32 EntriesCount = (TSize - sizeof(TQueueChunk*)) / sizeof(T);
        static_assert(EntriesCount > 0, "expect EntriesCount > 0");

        volatile T Entries[EntriesCount];
        TQueueChunk* volatile Next;

        TQueueChunk() {
            memset(this, 0, sizeof(TQueueChunk));
        }
    };

    // copied from <kikimr/core/util/unordered_cache.h> to eliminate dependency
    // todo: extend TRelaxedManyManyQueue from <library/cpp/threading/queue.h> with missing features and replace with

    template <typename T, ui32 TSize = 512, ui32 TConcurrencyFactor = 1, typename TChunk = TQueueChunk<T, TSize>>
    class TUnorderedCache : TNonCopyable {
        static_assert(std::is_integral<T>::value || std::is_pointer<T>::value, "expect std::is_integral<T>::value || std::is_pointer<T>::value");

    public:
        static const ui32 Concurrency = 4 * TConcurrencyFactor;

    private:
        // on first cache line we put everything about readers
        ui32 ReadPosition[Concurrency]; // todo: overlap readfrom/readposition and place blocks for different queues on different cache-lines (now every read clash with other reads, and all writes clash with other writes).
        TChunk* volatile ReadFrom[Concurrency];
        ui32 ReaderDummyPadding[2 + TConcurrencyFactor * 2];
        // here goes next cache line and writers stuff
        volatile ui32 WritePosition[Concurrency];
        TChunk* volatile WriteTo[Concurrency]; // todo: overlap write-to/reserved blocks
        TChunk* volatile Reserved[Concurrency];

        static_assert(sizeof(TChunk*) == sizeof(TAtomic), "expect sizeof(TChunk*) == sizeof(TAtomic)");

        void LockWriter(TChunk*& writeTo, ui64& index, ui64 writerRotation) {
            Y_VERIFY_DEBUG(writeTo == nullptr);
            ui64 writerIndex = writerRotation;
            for (;;) {
                index = writerIndex % Concurrency;
                if (AtomicLoad(&WriteTo[index]) != nullptr) {
                    if (writeTo = AtomicSwap(&WriteTo[index], nullptr)) {
                        if (AtomicLoad(&Reserved[index]) != nullptr)
                            return;

                        AtomicStore(&WriteTo[index], writeTo); // unlock to avoid race on reading
                        if (TChunk* overtaking = AtomicSwap(&Reserved[index], new TChunk()))
                            delete overtaking;
                    }
                }
                ++writerIndex;
            } // todo (?): some sort of spinwait
        }

        void UnlockWriter(TChunk* writeTo, ui64 index) {
            AtomicStore(&WriteTo[index], writeTo);
        }

        void WriteOne(TChunk*& writeTo, ui64 index, T x) {
            Y_VERIFY_DEBUG(x != 0);

            const ui32 pos = AtomicLoad(&WritePosition[index]);
            if (pos != TChunk::EntriesCount) {
                AtomicStore(&WritePosition[index], pos + 1);
                AtomicStore(&writeTo->Entries[pos], x);
            } else {
                TChunk* next = AtomicSwap(&Reserved[index], nullptr);
                AtomicStore(&next->Entries[0], x);
                AtomicStore(&WritePosition[index], 1u);
                AtomicStore(&writeTo->Next, next);
                writeTo = next;
            }
        }

    public:
        TUnorderedCache(bool uninitialized = false) {
            if (!uninitialized) {
                for (ui32 i = 0; i != Concurrency; ++i) {
                    ReadPosition[i] = 0;
                    ReadFrom[i] = new TChunk();

                    WritePosition[i] = 0;
                    WriteTo[i] = ReadFrom[i];

                    Reserved[i] = nullptr;
                }
            }
        }

        ~TUnorderedCache() {
            Y_VERIFY(!Pop(0));

            for (ui64 i = 0; i < Concurrency; ++i) {
                if (ReadFrom[i]) {
                    delete ReadFrom[i];
                    ReadFrom[i] = nullptr;
                }
                if (Reserved[i]) {
                    delete Reserved[i];
                    Reserved[i] = nullptr;
                }
            }
        }

        T Pop(ui64 readerRotation) {
            ui64 readerIndex = readerRotation;
            const ui64 endIndex = readerIndex + Concurrency;
            for (; readerIndex != endIndex; ++readerIndex) {
                const ui64 i = readerIndex % Concurrency;
                if (RelaxedLoad(&ReadFrom[i]) != nullptr) {
                    if (TChunk* readFrom = AtomicSwap(&ReadFrom[i], nullptr)) {
                        const ui32 pos = AtomicLoad(&ReadPosition[i]);
                        if (pos != TChunk::EntriesCount) {
                            if (T ret = AtomicLoad(&readFrom->Entries[pos])) {
                                AtomicStore(&ReadPosition[i], pos + 1);
                                AtomicStore(&ReadFrom[i], readFrom); // release lock with same chunk
                                return ret;                          // found, return
                            } else {
                                AtomicStore(&ReadFrom[i], readFrom); // release lock with same chunk
                            }
                        } else if (TChunk* next = AtomicLoad(&readFrom->Next)) {
                            if (T ret = AtomicLoad(&next->Entries[0])) {
                                AtomicStore(&ReadPosition[i], 1u);
                                AtomicStore(&ReadFrom[i], next); // release lock with new chunk
                                delete readFrom;
                                return ret;
                            } else {
                                AtomicStore(&ReadPosition[i], 0u);
                                AtomicStore(&ReadFrom[i], next); // release lock with new chunk
                                delete readFrom;
                            }
                        } else {
                            AtomicStore(&ReadFrom[i], readFrom); // nothing in old chunk and no next chunk, just release lock with old chunk
                        }
                    }
                }
            }

            return 0; // got nothing after full cycle, return
        }

        void Push(T x, ui64 writerRotation) {
            TChunk* writeTo = nullptr;
            ui64 index = 0;

            LockWriter(writeTo, index, writerRotation);
            WriteOne(writeTo, index, x);
            UnlockWriter(writeTo, index);
        }

        void PushBulk(T* x, ui32 xcount, ui64 writerRotation) {
            TChunk* writeTo = nullptr;
            ui64 index = 0;

            for (;;) { // fill no more then one queue chunk per round
                const ui32 xround = Min(xcount, (ui32)TChunk::EntriesCount);

                LockWriter(writeTo, index, writerRotation++);
                for (T* end = x + xround; x != end; ++x)
                    WriteOne(writeTo, index, *x);
                UnlockWriter(writeTo, index);

                if (xcount <= TChunk::EntriesCount)
                    break;

                xcount -= TChunk::EntriesCount;
            }
        }
    };

}
