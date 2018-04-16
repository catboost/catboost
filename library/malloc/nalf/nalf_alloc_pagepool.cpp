#include "nalf_alloc_pagepool.h"
#include "nalf_alloc_impl.h"

namespace NNumaAwareLockFreeAllocator {
    TPagePool::TPagePool()
        : PoolType(ClassUndefined)
        , NotAllocatedPagesLeft(0)
    {
    }

    TPagePool::~TPagePool() {
        // must be called only on allocator shutdown
        // could return pages back to system (but who cares?)
    }

    void TPagePool::Init(EType poolType, ui32 maxPages) {
        Y_VERIFY(PoolType == ClassUndefined && poolType != ClassUndefined);
        PoolType = poolType;
        NotAllocatedPagesLeft = maxPages;
    }

    void* TPagePool::Pop() {
        return PopInt(FirstPage);
    }

    static ui32 LastChunkIdx(TPagePool::EType pageType) {
        switch (pageType) {
            case TPagePool::Class1Gb:
                return (1024 * 1024 * 1024) / TChunkHeader::SmallChunkSize;
            case TPagePool::Class2Mb:
            case TPagePool::Class4k:
                return (2 * 1024 * 1024) / TChunkHeader::SmallChunkSize;
            default:
                Y_FAIL("broken TPagePool::EType");
        }
    }

    static void* AllocatePageMemory(TPagePool::EType pageType) {
        switch (pageType) {
            case TPagePool::Class1Gb:
                return SystemAllocation(1024 * 1024 * 1024); // TODO: allocate 1gb page
                break;
            case TPagePool::Class2Mb:
                return SystemAllocation(2 * 1024 * 1024); // TODO: allocate huge pages
                break;
            case TPagePool::Class4k:
                return SystemAllocation(2 * 1024 * 1024);
                break;
            default:
                Y_FAIL("broken TPagePool::EType");
        }
    }

    void* TPagePool::PopInt(TPageListElement& pageList) {
        const ui32 lastChunkIdx = LastChunkIdx(PoolType);
        // todo: remove recursion

        const ui32 activeChunk = AtomicLoad(&pageList.ActiveChunk);

        if (activeChunk == TPageListElement::PagesPerList) { // current page-list is done
            if (TPageListElement* const next = AtomicLoad(&pageList.Next)) {
                return PopInt(*next);
            } else if (AtomicLoad(&NotAllocatedPagesLeft) > 0) {
                AllocatePage(pageList);
                return PopInt(pageList);
            } else {
                if (TPageListElement* const next2 = AtomicLoad(&pageList.Next))
                    return PopInt(*next2);
                else
                    return nullptr; // all hope is gone
            }
        }

        void* const chunkmem = AtomicLoad(&pageList.PageMemory[activeChunk]);

        if (chunkmem == nullptr) { // page not allocated, we are first one trying?
            if (AtomicLoad(&NotAllocatedPagesLeft) > 0) {
                AllocatePage(pageList); // must be called under lock
                return PopInt(pageList);
            } else {
                if (AtomicLoad(&pageList.PageMemory[activeChunk]))
                    return PopInt(pageList);
                else
                    return nullptr;
            }
        }

        const ui32 chunkidx = AtomicLoad(&pageList.AllocatedChunk[activeChunk]);

        if (chunkidx >= lastChunkIdx) { // chunk overcommit, try to select next and retry
            AtomicUi32Cas(&pageList.ActiveChunk, activeChunk + 1, activeChunk);
            return PopInt(pageList);
        }

        const ui32 myidxend = AtomicUi32Add(&pageList.AllocatedChunk[activeChunk], TChunkHeader::SmallChunksPerLarge);
        if (myidxend >= lastChunkIdx)
            return PopInt(pageList);

        const ui32 myidx = myidxend - TChunkHeader::SmallChunksPerLarge;
        // ok, we got 32k block for us!
        void* const retmem = (ui8*)chunkmem + myidx * TChunkHeader::SmallChunkSize;

        // now check possibilty to allocate new page speculatively to never hang on page allocation
        const ui32 barrierChunkIdx = lastChunkIdx * 112 / 128;
        if (myidx < barrierChunkIdx && myidxend >= barrierChunkIdx)
            AllocatePage(pageList);

        // got it!
        return retmem;
    }

    void TPagePool::TPageListElement::Init(TPagePool::EType poolType, ui32 idx) {
        ui8* pagemem = (ui8*)AllocatePageMemory(poolType);
        Y_VERIFY(pagemem, "system allocation must never fails, we don't handle such conditions");
        ui8* alignedmem = AlignUp(pagemem, TChunkHeader::LargeChunkSize);
        ui32 startidx = (ui32)((alignedmem - pagemem) / TChunkHeader::SmallChunkSize);
        AtomicStore(&AllocatedChunk[idx], startidx);
        AtomicStore<void*>(&PageMemory[idx], pagemem);
    }

    void TPagePool::AllocatePage(TPageListElement& pageList) {
        TGuard<TAdaptiveLock> guard(&Lock);

        if (NotAllocatedPagesLeft == 0)
            return;

        ui32 activeChunk = AtomicLoad(&pageList.ActiveChunk);
        if (activeChunk != TPageListElement::PagesPerList && AtomicLoad(&pageList.PageMemory[activeChunk]) != nullptr)
            ++activeChunk;

        if (activeChunk == TPageListElement::PagesPerList) {
            if (pageList.Next == nullptr) {
                // allocate page-list, then allocate page, then setup page and finally - attach as next and finally - decrement notallocatedpagesleft
                THolder<TPageListElement> nlist(new TPageListElement());
                nlist->Init(PoolType, 0);
                AtomicStore(&pageList.Next, nlist.Release());
                AtomicStore(&NotAllocatedPagesLeft, NotAllocatedPagesLeft - 1);
            }
            return;
        }

        if (AtomicLoad(&pageList.PageMemory[activeChunk]) == nullptr) {
            pageList.Init(PoolType, activeChunk);
            AtomicStore(&NotAllocatedPagesLeft, NotAllocatedPagesLeft - 1);
        }
    }

}
