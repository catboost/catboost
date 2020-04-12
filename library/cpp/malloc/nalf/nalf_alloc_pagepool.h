#pragma once

#include "defs.h"
#include "nalf_alloc.h"
#include "alloc_helpers.h"

#include <util/system/spinlock.h>

namespace NNumaAwareLockFreeAllocator {
    class TPagePool : TNonCopyable {
    public:
        enum EType {
            ClassUndefined,
            Class1Gb,
            Class2Mb,
            Class4k,
        };

    private:
        // first element must be place with zero allocation (as we must place it before any allocation is possible)
        // (not true as on first step we use bootstrap page pool)
        struct TPageListElement : TSystemAllocHelper<TPageListElement> {
            static const ui32 PagesPerList = 340; // for 4k page
            void* PageMemory[PagesPerList];
            ui32 AllocatedChunk[PagesPerList];
            ui32 ActiveChunk;
            TPageListElement* Next;

            TPageListElement() {
                memset(this, 0, sizeof(*this));
            }

            void Init(TPagePool::EType poolType, ui32 idx);
        };

        EType PoolType;
        ui32 NotAllocatedPagesLeft;

        TPageListElement FirstPage;
        TAdaptiveLock Lock; // used for new page allocation

        void AllocatePage(TPageListElement& pageList); // must be called under lock
        void* PopInt(TPageListElement& pageList);

    public:
        TPagePool();
        ~TPagePool();

        void Init(EType poolType, ui32 maxPages);
        void* Pop(); // returns pointer to 32k block
    };

}
