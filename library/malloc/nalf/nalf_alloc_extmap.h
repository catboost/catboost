#pragma once

#include "defs.h"
#include "nalf_alloc.h"

namespace NNumaAwareLockFreeAllocator {
    class TExtMappingsMap : TNonCopyable {
        static const ui64 LocalPlacementFlag = 1ull << 63;
        static const ui64 MaxNumaNode = NALF_ALLOC_NUMANODES - 1;
        static const ui64 NumaFieldShift = 63 - NALF_ALLOC_NUMANODES;
        static const ui64 NumaFieldMask = ((0x1ull << NALF_ALLOC_NUMANODES) - 1) << NumaFieldShift;
        static const ui64 MaxFieldPages = (0x1ull << NALF_ALLOC_MAXMAPTRACKED) - 1;
        static const ui64 SizeFieldShift = NumaFieldShift - NALF_ALLOC_MAXMAPTRACKED;
        static const ui64 SizeFieldMask = ((0x1ull << NALF_ALLOC_MAXMAPTRACKED) - 1) << SizeFieldShift;
        static const ui64 AddressMask = (0x1ull << SizeFieldShift) - 1;

        struct TLinkedChunk : TNonCopyable {
            ui64 Payload[NALF_ALLOC_LINKEDMAPSIZE];
            TLinkedChunk* Next;

            TLinkedChunk() {
                memset(this, 0, sizeof(TLinkedChunk));
            }
        };

        ui64 Base[NALF_ALLOC_EXTMAPPINGS_BASE];

        // see http://arxiv.org/pdf/1202.4961.pdf for explanation
        static ui32 Hash(void* mem) noexcept {
            const ui64 mx = (ui64)mem;
            const ui32 p[2] = {(ui32)mx, (ui32)(mx >> 32ull)};

            const ui64 x1 = p[0] * 0x001DFF3D8DC48F5Dull;
            const ui64 x2 = p[1] * 0x179CA10C9242235Dull;
            const ui64 sum = 0x0F530CAD458B0FB1 + x1 + x2;

            return (sum >> 32);
        }

        bool PlaceLocal(ui64 mx, ui32 idx) noexcept {
            const ui64 mxl = mx | LocalPlacementFlag;
            if (AtomicCas(&Base[idx], mxl, 0))
                return true;
            else
                return false;
        }

        bool ReplaceLocal(ui64 mx, ui32 idx, ui64 lx) noexcept {
            TLinkedChunk* x = ::new (Allocate(sizeof(TLinkedChunk))) TLinkedChunk();
            x->Payload[0] = lx & ~LocalPlacementFlag;
            x->Payload[1] = mx;

            if (AtomicCas(&Base[idx], (TAtomicBase)x, lx))
                return true;

            Free(x);
            return false;
        }

        void PlaceLinked(ui64 mx, ui64 lx) noexcept {
            TLinkedChunk* chunk = (TLinkedChunk*)lx;

            ui64* p = chunk->Payload;
            ui64* pend = chunk->Payload + NALF_ALLOC_LINKEDMAPSIZE;

            for (;;) {
                if (AtomicLoad(p) == 0 && AtomicCas(p, mx, 0))
                    return;

                if (++p == pend) {
                    TLinkedChunk* next = AtomicLoad(&chunk->Next);
                    if (next != nullptr) {
                        chunk = next;
                        p = chunk->Payload;
                        pend = chunk->Payload + NALF_ALLOC_LINKEDMAPSIZE;
                    } else {
                        TLinkedChunk* x = ::new (Allocate(sizeof(TLinkedChunk))) TLinkedChunk();
                        x->Payload[0] = mx;
                        if (AtomicCas(&chunk->Next, x, nullptr))
                            return;

                        Free(x);

                        chunk = AtomicLoad(&chunk->Next);
                        p = chunk->Payload;
                        pend = chunk->Payload + NALF_ALLOC_LINKEDMAPSIZE;
                    }
                }
            }
        }

        // mx - prepared address, lx - stored address.
        // compares address-part of values and extracts size on success
        static ui32 CompareAddresses(ui64 mx, ui64 lx) noexcept {
            if (mx == (lx & AddressMask))
                return (lx & SizeFieldMask) >> SizeFieldShift;
            else
                return 0;
        }

        std::pair<ui32, ui32> ReclaimLocal(ui64 mx, ui32 idx, ui64 lx, bool pop) {
            const ui32 sz = CompareAddresses(mx, lx);
            if (sz != 0) {
                if (pop) {
                    if (AtomicCas(&Base[idx], 0, lx))
                        return std::pair<ui32, ui32>(sz, (lx & NumaFieldMask) >> NumaFieldShift);
                    else {
                        const ui64 lxx = AtomicLoad(&Base[idx]);
                        Y_VERIFY(lxx != 0 && (lxx & LocalPlacementFlag) == 0);
                        return ReclaimLinked(mx, lxx, true);
                    }
                } else {
                    return std::pair<ui32, ui32>(sz, (lx & NumaFieldMask) >> NumaFieldShift);
                }
            } else {
                return std::pair<ui32, ui32>(0, 0);
            }
        }

        std::pair<ui32, ui32> ReclaimLinked(ui64 mx, ui64 lx, bool pop) {
            TLinkedChunk* chunk = (TLinkedChunk*)lx;
            do {
                for (ui64 *p = chunk->Payload, *pend = chunk->Payload + NALF_ALLOC_LINKEDMAPSIZE; p != pend; ++p) {
                    const ui64 lxx = AtomicLoad(p);
                    const ui32 sz = CompareAddresses(mx, lxx);
                    if (sz != 0) {
                        if (pop)
                            AtomicStore(p, ui64(0));
                        return std::pair<ui32, ui32>(sz, (lxx & NumaFieldMask) >> NumaFieldShift);
                    }
                }
                chunk = AtomicLoad(&chunk->Next);
            } while (chunk != nullptr);
            return std::pair<ui32, ui32>(0, 0);
        }

    public:
        TExtMappingsMap() {
        }

        ~TExtMappingsMap() {
            // we are destroying, as only one instance is instance in GlobalAllocator singleton - we are in shutdown process
            // so not much must be done at this step and no much sense in accurate unmapping all tracked memory
        }

        void Push(void* mem, ui32 pages, ui32 numaNode) {
            const ui32 hash = Hash(mem);
            const ui32 idx = hash % (NALF_ALLOC_EXTMAPPINGS_BASE);

            ui64 mx = (ui64)mem;
            Y_VERIFY((mx % SystemPageSize) == 0, "page address: %" PRIx64 ", pages: %" PRIu32 " must be aligned on page boundary", mx, pages);

            mx = mx >> 12; // get rid of low space

            Y_VERIFY(numaNode <= MaxNumaNode, "broken numa node id");
            mx = (ui64(numaNode) << NumaFieldShift) | (mx & ~NumaFieldMask);

            Y_VERIFY(pages <= MaxFieldPages, "large mappings are not tracked, instead [-1] page used");
            mx = (ui64(pages) << SizeFieldShift) | (mx & ~SizeFieldMask);

            for (;;) {
                const ui64 lx = AtomicLoad(&Base[idx]);
                if (lx == 0) {
                    if (PlaceLocal(mx, idx))
                        return;
                } else if (lx & LocalPlacementFlag) {
                    if (ReplaceLocal(mx, idx, lx))
                        return;
                } else {
                    PlaceLinked(mx, lx);
                    return;
                }
            }
        }

        std::pair<ui32, ui32> Lookup(void* mem, bool pop) {
            const ui32 hash = Hash(mem);
            const ui32 idx = hash % NALF_ALLOC_EXTMAPPINGS_BASE;

            ui64 mx = (ui64)mem;
            Y_VERIFY((mx % SystemPageSize) == 0, "page address must be aligned on page boundary");

            mx = (mx >> 12) & (AddressMask);

            const ui64 lx = AtomicLoad(&Base[idx]);
            if (lx == 0)
                return std::pair<ui32, ui32>(0, 0);
            else if (lx & LocalPlacementFlag)
                return ReclaimLocal(mx, idx, lx, pop);
            else
                return ReclaimLinked(mx, lx, pop);
        }
    };

}
