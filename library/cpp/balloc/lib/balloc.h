#pragma once

#include <sys/mman.h>
#include <pthread.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include <new>
#include <util/system/defaults.h>
#include <library/cpp/malloc/api/malloc.h>
#include <library/cpp/balloc/lib/alloc_stats.h>
#include <library/cpp/balloc/setup/alloc.h>

#ifndef NDEBUG
#define DBG_FILL_MEMORY
#endif

#if defined(Y_COVER_PTR)
#define DBG_FILL_MEMORY
#endif

#if (defined(_i386_) || defined(_x86_64_)) && defined(_linux_)
#define HAVE_VDSO_GETCPU 1

#include <contrib/libs/linuxvdso/interface.h>
#endif

namespace NBalloc {
#if HAVE_VDSO_GETCPU
    // glibc does not provide a wrapper around getcpu, we'll have to load it manually
    static int (*getcpu)(unsigned* cpu, unsigned* node, void* unused) = nullptr;
#endif

    static Y_FORCE_INLINE void* Advance(void* block, size_t size) {
        return (void*)((char*)block + size);
    }

    static constexpr size_t PAGE_CACHE = 16;
#if defined(_ppc64_) || defined(_arm64_)
    static constexpr size_t PAGE_ELEM = 65536;
#else
    static constexpr size_t PAGE_ELEM = 4096;
#endif
    static constexpr size_t SINGLE_ALLOC = (PAGE_ELEM / 2);
    static constexpr size_t ORDERS = 1024;
    static constexpr size_t DUMP_STAT = 0;

    static void* (*LibcMalloc)(size_t) = nullptr;
    static void (*LibcFree)(void*) = nullptr;

    static size_t Y_FORCE_INLINE Align(size_t value, size_t align) {
        return (value + align - 1) & ~(align - 1);
    }

#define RDTSC(eax, edx) __asm__ __volatile__("rdtsc" \
                                             : "=a"(eax), "=d"(edx));
#define CPUID(func, eax, ebx, ecx, edx) __asm__ __volatile__("cpuid"                                      \
                                                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) \
                                                             : "a"(func));

    static int GetNumaNode() {
#if HAVE_VDSO_GETCPU
        if (Y_LIKELY(getcpu)) {
            unsigned node = 0;
            if (getcpu(nullptr, &node, nullptr)) {
                return 0;
            }
            return node;
        }
#endif
#if defined(_i386_) or defined(_x86_64_)
        int a = 0, b = 0, c = 0, d = 0;
        CPUID(0x1, a, b, c, d);
        int acpiID = (b >> 24);
        int numCPU = (b >> 16) & 255;
        if (numCPU == 0)
            return 0;
        int ret = acpiID / numCPU;
        return ret;
#else
        return 0;
#endif
    }

    static void AbortFromSystemError() {
        char buf[512] = {0};
#if defined(_freebsd_) or defined(_darwin_) or defined(_musl_) or defined(_bionic_)
        strerror_r(errno, buf, sizeof(buf));
        const char* msg = buf;
#elif defined(_linux_) or defined(_cygwin_)
        const char* msg = strerror_r(errno, buf, sizeof(buf));
#endif
        NMalloc::AbortFromCorruptedAllocator(msg);
    }

    static pthread_key_t key;
    static volatile long init = 0;
    static unsigned long long counter = 0;

    static void Destructor(void* data);

    template <class T>
    Y_FORCE_INLINE bool DoCas(T* volatile* target, T* exchange, T* compare) {
        return __sync_bool_compare_and_swap(target, compare, exchange);
    }

    class TLFAllocFreeList {
        struct TNode {
            TNode* Next;
        };

        TNode* volatile Head;
        TNode* volatile Pending;
        long long volatile PendingToFreeListCounter;
        TNode* volatile Destroyed;
        long long AllocCount;

        static Y_FORCE_INLINE void Enqueue(TNode* volatile* headPtr, TNode* n) {
            for (;;) {
                TNode* volatile prevHead = *headPtr;
                n->Next = prevHead;
                if (DoCas(headPtr, n, prevHead))
                    break;
            }
        }
        Y_FORCE_INLINE void* DoAlloc() {
            TNode* res;
            for (res = Head; res; res = Head) {
                TNode* keepNext = res->Next;
                if (DoCas(&Head, keepNext, res)) {
                    //Y_ABORT_UNLESS(keepNext == res->Next);
                    break;
                }
            }
            return res;
        }
        void FreeList(TNode* fl) {
            if (!fl)
                return;
            TNode* flTail = fl;
            while (flTail->Next)
                flTail = flTail->Next;
            for (;;) {
                TNode* volatile prevHead = Head;
                flTail->Next = prevHead;
                if (DoCas(&Head, fl, prevHead))
                    break;
            }
        }

    public:
        Y_FORCE_INLINE void Free(void* ptr) {
            TNode* newFree = (TNode*)ptr;
            if (__sync_add_and_fetch(&AllocCount, 0) == 0)
                Enqueue(&Head, newFree);
            else
                Enqueue(&Pending, newFree);
        }
        Y_FORCE_INLINE void Destroy(void* ptr, size_t length) {
            TNode* newFree = (TNode*)ptr;
            TNode* fl = nullptr;
            if (__sync_add_and_fetch(&AllocCount, 1) == 1) {
                fl = Destroyed;
                if (fl && !DoCas(&Destroyed, (TNode*)nullptr, fl)) {
                    fl = nullptr;
                }
                Enqueue(&fl, newFree);
            } else {
                Enqueue(&Destroyed, newFree);
            }
            __sync_sub_and_fetch(&AllocCount, 1);

            // TODO try to merge blocks to minimize number of syscalls
            while (nullptr != fl) {
                TNode* next = fl->Next;
                if (-1 == munmap(fl, length)) {
                    AbortFromSystemError();
                }
                fl = next;
            }
        }
        Y_FORCE_INLINE void* Alloc() {
            long long volatile keepCounter = __sync_add_and_fetch(&PendingToFreeListCounter, 0);
            TNode* fl = Pending;
            if (__sync_add_and_fetch(&AllocCount, 1) == 1) {
                // No other allocs in progress.
                // If (keepCounter == PendingToFreeListCounter) then Pending was not freed by other threads.
                // Hence Pending is not used in any concurrent DoAlloc() atm and can be safely moved to FreeList
                if (fl &&
                    keepCounter == __sync_add_and_fetch(&PendingToFreeListCounter, 0) &&
                    DoCas(&Pending, (TNode*)nullptr, fl))
                {
                    // pick first element from Pending and return it
                    void* res = fl;
                    fl = fl->Next;
                    // if there are other elements in Pending list, add them to main free list
                    FreeList(fl);
                    __sync_sub_and_fetch(&PendingToFreeListCounter, 1);
                    __sync_sub_and_fetch(&AllocCount, 1);
                    return res;
                }
            }
            void* res = DoAlloc();
            if (!res && __sync_add_and_fetch(&Pending, 0)) {
                // live-lock situation: there are no free items in the "Head"
                // list and there are free items in the "Pending" list
                // but the items are forbidden to allocate to prevent ABA
                NAllocStats::IncLiveLockCounter();
            }
            __sync_sub_and_fetch(&AllocCount, 1);
            return res;
        }
    };

    TLFAllocFreeList nodes[2][ORDERS];
    unsigned long long sizesGC[2][16];
    unsigned long long sizeOS, totalOS;

    struct TBlockHeader {
        size_t Size;
        int RefCount;
        unsigned short AllCount;
        unsigned short NumaNode;
    };

    static bool PushPage(void* page, size_t order) {
        if (order < ORDERS) {
            int node = ((TBlockHeader*)page)->NumaNode;
            __sync_add_and_fetch(&sizesGC[node][order % 16], order);
            TBlockHeader* blockHeader = (TBlockHeader*)page;
            if (!__sync_bool_compare_and_swap(&blockHeader->RefCount, 0, -1)) {
                NMalloc::AbortFromCorruptedAllocator();
            }
            nodes[node][order].Free(page);
            return true;
        }
        return false;
    }

    static void* PopPage(size_t order) {
        if (order < ORDERS) {
            int numaNode = GetNumaNode() & 1;
            void* alloc = nodes[numaNode][order].Alloc();
            if (alloc == nullptr) {
                alloc = nodes[1 - numaNode][order].Alloc();
                if (alloc) {
                    __sync_sub_and_fetch(&sizesGC[1 - numaNode][order % 16], order);
                }
            } else {
                __sync_sub_and_fetch(&sizesGC[numaNode][order % 16], order);
            }
            if (alloc) {
                TBlockHeader* blockHeader = (TBlockHeader*)alloc;
                if (!__sync_bool_compare_and_swap(&blockHeader->RefCount, -1, 0)) {
                    NMalloc::AbortFromCorruptedAllocator();
                }
            }
            return alloc;
        }
        return nullptr;
    }

#if DUMP_STAT
    static unsigned long long TickCounter() {
        int lo = 0, hi = 0;
        RDTSC(lo, hi);
        return (((unsigned long long)hi) << 32) + (unsigned long long)lo;
    }

    struct TTimeHold {
        unsigned long long Start;
        unsigned long long Finish;
        const char* Name;
        TTimeHold(const char* name)
            : Start(TickCounter())
            , Name(name)
        {
        }
        ~TTimeHold() {
            Finish = TickCounter();
            double diff = Finish > Start ? (Finish - Start) / 1000000.0 : 0.0;
            if (diff > 20.0) {
                fprintf(stderr, "%s %f mticks\n", diff, Name);
            }
        }
    };
#endif

    long long allocs[ORDERS];

    static void Map(size_t size, void* pages[], size_t num) {
#if DUMP_STAT
        TTimeHold hold("mmap");
        size_t order = size / PAGE_ELEM;
        if (order < ORDERS) {
            __sync_add_and_fetch(&allocs[order], num);
        }
#endif
        if (!NAllocSetup::CanAlloc(__sync_add_and_fetch(&sizeOS, size * num), totalOS)) {
            NMalloc::AbortFromCorruptedAllocator();
        }
        void* map = mmap(nullptr, size * num, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (map == MAP_FAILED) {
            AbortFromSystemError();
        }
        unsigned short numaNode = (GetNumaNode() & 1);
        NAllocStats::IncMmapCounter(size * num / PAGE_ELEM);
        for (size_t i = 0; i < num; ++i) {
            TBlockHeader* blockHeader = static_cast<TBlockHeader*>(map);
            blockHeader->NumaNode = numaNode;
            pages[i] = map;
            map = Advance(map, size);
        }
    }

    static void* SysAlloc(size_t& size) {
        size = Align(size, PAGE_ELEM);
        size_t order = size / PAGE_ELEM;
        void* result = PopPage(order);
        if (result) {
            return result;
        }
        void* pages[1] = {nullptr};
        Map(size, pages, 1);
        return pages[0];
    }

    static void UnMap(void* block, size_t order) {
#if DUMP_STAT
        TTimeHold hold("munmap");
        if (order < ORDERS) {
            __sync_sub_and_fetch(&allocs[order], 1);
        }
#endif
        size_t size = order * PAGE_ELEM;
        __sync_sub_and_fetch(&sizeOS, size);
        TBlockHeader* blockHeader = (TBlockHeader*)block;
        if (!__sync_bool_compare_and_swap(&blockHeader->RefCount, 0, -1)) {
            NMalloc::AbortFromCorruptedAllocator();
        }
        if (order < ORDERS) {
            int node = blockHeader->NumaNode;
            nodes[node][order].Destroy(block, size);
        } else {
            if (-1 == munmap(block, size)) {
                AbortFromSystemError();
            }
        }
    }

    static void SysClear(size_t order) {
        void* page = PopPage(order);
        if (page) {
            UnMap(page, order);
        }
    }

    static void Y_FORCE_INLINE GlobalInit() {
        if (__sync_bool_compare_and_swap(&init, 0, 1)) {
#if HAVE_VDSO_GETCPU
            getcpu = (int (*)(unsigned*, unsigned*, void*))NVdso::Function("__vdso_getcpu", "LINUX_2.6");
#endif
            LibcMalloc = (void* (*)(size_t))dlsym(RTLD_NEXT, "malloc");
            LibcFree = (void (*)(void*))dlsym(RTLD_NEXT, "free");
            pthread_key_create(&key, Destructor);
            __sync_bool_compare_and_swap(&init, 1, 2);
        }
        while (init < 2) {
        };
    }

    enum EMode {
        Empty = 0,
        Born,
        Alive,
        Disabled,
        Dead,
        ToBeEnabled
    };

    struct TLS {
        void* PageCache[PAGE_CACHE];
        size_t Cached;
        void* Chunk;
        size_t Ptr;
        void* Block;
        int Counter;
        EMode Mode;
        unsigned char Count0;
        unsigned long Count1;
        bool NeedGC() {
            if (Count0++ != 0)
                return false;
            __sync_add_and_fetch(&totalOS, 1);
            unsigned long long count = 0;
            for (size_t i = 0; i < 16; ++i) {
                count += sizesGC[0][i];
                count += sizesGC[1][i];
            }
            return NAllocSetup::NeedReclaim(count * PAGE_ELEM, ++Count1);
        }
        void ClearCount() {
            Count1 = 0;
        }
    };

#if defined(_darwin_)

    static Y_FORCE_INLINE TLS* PthreadTls() {
        GlobalInit();
        TLS* ptls = (TLS*)pthread_getspecific(key);
        if (!ptls) {
            ptls = (TLS*)LibcMalloc(sizeof(TLS));
            if (!ptls) {
                NMalloc::AbortFromCorruptedAllocator(); // what do we do here?
            }
            memset(ptls, 0, sizeof(TLS));
            pthread_setspecific(key, ptls);
        }
        return ptls;
    }

#define tls (*PthreadTls())

#else

    __thread TLS tls;

#endif

    static void UnRefHard(void* block, int add, TLS& ltls) {
        TBlockHeader* blockHeader = (TBlockHeader*)block;
        if ((blockHeader->RefCount == add ? (blockHeader->RefCount = 0, true) : false) || __sync_sub_and_fetch(&blockHeader->RefCount, add) == 0) {
            size_t order = blockHeader->Size / PAGE_ELEM;
            if (ltls.Mode == Alive) {
                // page cache has first priority
                if (order == 1 && ltls.Cached < PAGE_CACHE) {
                    ltls.PageCache[ltls.Cached] = block;
                    ++ltls.Cached;
                    return;
                }
                if (ltls.NeedGC()) {
                    ltls.ClearCount();
                    size_t index = __sync_add_and_fetch(&counter, 1);
                    SysClear(index % ORDERS);
                    UnMap(block, order);
                    return;
                }
            }
            if (!PushPage(block, order)) {
                UnMap(block, order);
            }
        }
    }

    static void Init(TLS& ltls) {
        bool ShouldEnable = (NAllocSetup::IsEnabledByDefault() || ltls.Mode == ToBeEnabled);
        ltls.Mode = Born;
        GlobalInit();
        pthread_setspecific(key, (void*)&ltls);
        if (ShouldEnable) {
            ltls.Mode = Alive;
        } else {
            ltls.Mode = Disabled;
        }
    }

    static void Y_FORCE_INLINE UnRef(void* block, int counter, TLS& ltls) {
        if (ltls.Mode != Alive) {
            UnRefHard(block, counter, ltls);
            return;
        }
        if (ltls.Block == block) {
            ltls.Counter += counter;
        } else {
            if (ltls.Block) {
                UnRefHard(ltls.Block, ltls.Counter, ltls);
            }
            ltls.Block = block;
            ltls.Counter = counter;
        }
    }

    static void Destructor(void* data) {
        TLS& ltls = *(TLS*)data;
        ltls.Mode = Dead;
        if (ltls.Chunk) {
            TBlockHeader* blockHeader = (TBlockHeader*)ltls.Chunk;
            UnRef(ltls.Chunk, PAGE_ELEM - blockHeader->AllCount, ltls);
        }
        if (ltls.Block) {
            UnRef(ltls.Block, ltls.Counter, ltls);
        }
        for (size_t i = 0; i < ltls.Cached; ++i) {
            PushPage(ltls.PageCache[i], 1);
        }
#if defined(_darwin_)
        LibcFree(data);
#endif
    }

    using TAllocHeader = NMalloc::TAllocHeader;

    static Y_FORCE_INLINE TAllocHeader* AllocateRaw(size_t size, size_t signature) {
        TLS& ltls = tls;
        size = Align(size, sizeof(TAllocHeader));
        if (Y_UNLIKELY(ltls.Mode == Empty || ltls.Mode == ToBeEnabled)) {
            Init(ltls);
        }
        size_t extsize = size + sizeof(TAllocHeader) + sizeof(TBlockHeader);
        if (extsize > SINGLE_ALLOC || ltls.Mode != Alive) {
            // The dlsym() function in GlobalInit() may call malloc() resulting in recursive call
            // of the NBalloc::Malloc(). We have to serve such allocation request via balloc even
            // when (IsEnabledByDefault() == false) because at this point we don't know where the
            // libc malloc is.
            if (extsize > 64 * PAGE_ELEM) {
                extsize = Align(extsize, 16 * PAGE_ELEM);
            }
            NAllocSetup::ThrowOnError(extsize);
            void* block = SysAlloc(extsize);
            TBlockHeader* blockHeader = (TBlockHeader*)block;
            blockHeader->RefCount = 1;
            blockHeader->Size = extsize;
            blockHeader->AllCount = 0;
            TAllocHeader* allocHeader = (TAllocHeader*)Advance(block, sizeof(TBlockHeader));
            allocHeader->Encode(blockHeader, size, signature);
            if (NAllocStats::IsEnabled()) {
                NAllocStats::IncThreadAllocStats(size);
            }
#ifdef DBG_FILL_MEMORY
            memset(allocHeader + 1, 0xec, size);
#endif
            return allocHeader;
        }

        size_t ptr = ltls.Ptr;
        void* chunk = ltls.Chunk;

        if (ptr < extsize) {
            NAllocSetup::ThrowOnError(PAGE_ELEM);
            if (chunk) {
                TBlockHeader* blockHeader = (TBlockHeader*)chunk;
                UnRef(chunk, PAGE_ELEM - blockHeader->AllCount, ltls);
            }
            void* block = nullptr;
            while (1) {
                if (ltls.Cached > 0) {
                    --ltls.Cached;
                    block = ltls.PageCache[ltls.Cached];
                    break;
                }
                block = PopPage(1);
                if (block) {
                    break;
                }
                Map(PAGE_ELEM, ltls.PageCache, PAGE_CACHE);
                ltls.Cached = PAGE_CACHE;
            }
            TBlockHeader* blockHeader = (TBlockHeader*)block;
            blockHeader->RefCount = PAGE_ELEM;
            blockHeader->Size = PAGE_ELEM;
            blockHeader->AllCount = 0;
            ltls.Ptr = PAGE_ELEM;
            ltls.Chunk = block;
            ptr = ltls.Ptr;
            chunk = ltls.Chunk;
        }
        ptr = ptr - size - sizeof(TAllocHeader);
        TAllocHeader* allocHeader = (TAllocHeader*)Advance(chunk, ptr);
        allocHeader->Encode(chunk, size, signature);
        TBlockHeader* blockHeader = (TBlockHeader*)chunk;
        ++blockHeader->AllCount;
        ltls.Ptr = ptr;
        if (NAllocStats::IsEnabled()) {
            NAllocStats::IncThreadAllocStats(size);
        }
#ifdef DBG_FILL_MEMORY
        memset(allocHeader + 1, 0xec, size);
#endif
        return allocHeader;
    }

    static void Y_FORCE_INLINE FreeRaw(void* ptr) {
        UnRef(ptr, 1, tls);
    }
}
