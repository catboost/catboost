#pragma once

#include "defs.h"
#include "nalf_alloc.h"
#include "alloc_helpers.h"

namespace NNumaAwareLockFreeAllocator {
    static const ui64 FreeMemoryMark = 0xDEDEDEDEDEDEDEDE;
    static const ui16 FreeMemoryMarkInc = 0xECEC;
    // interleaved header for line-chunk-smallchunk
    struct TChunkHeader : TNonCopyable {
        static const ui32 MagicTag = 0x9A1FA11C;
        static const ui32 HiddenMagicTag = 0xA9F11AC1;
        static const ui64 SmallChunkSize = SystemPageSize;
        static const ui64 SmallChunksPerLarge = 8;
        static const ui64 LargeChunkSize = SmallChunksPerLarge * SmallChunkSize;

        static const ui64 MaxIncrementalAllocation = LargeChunkSize / 2 - 64;

        struct TChunkQueue : TWithNalfForceChunkedAlloc, TQueueChunk<ui16, 16> {};
        static_assert(sizeof(TChunkQueue) == sizeof(TQueueChunk<ui16, 16>), "expect sizeof(TChunkQueue) == sizeof(TQueueChunk<ui8, 16>)");

        const static int g = sizeof(TChunkQueue);

        enum EChunkType {
            ChunkUnknown,
            // 4k pages, entries aligned on 16 bytes
            Chunk16,
            Chunk32,
            Chunk48,
            Chunk64,
            Chunk96,
            // 32k pages, entries aligned on 64 bytes
            Chunk128,
            Chunk192,
            Chunk256,
            Chunk384,
            Chunk512,
            Chunk768,
            Chunk1024,
            Chunk1536,
            Chunk2176,
            Chunk3584,
            Chunk4096,
            Chunk6528,
            // 32k page with incremental allocation, 16 bytes alignment and page-wide ref-counting
            ChunkIncremental,
        };

        static ui32 TypeToSize(EChunkType chunkType) {
            switch (chunkType) {
                case ChunkUnknown:
                    return 0;
                case Chunk16:
                    return 16;
                case Chunk32:
                    return 32;
                case Chunk48:
                    return 48;
                case Chunk64:
                    return 64;
                case Chunk96:
                    return 96;
                case Chunk128:
                    return 128;
                case Chunk192:
                    return 192;
                case Chunk256:
                    return 256;
                case Chunk384:
                    return 384;
                case Chunk512:
                    return 512;
                case Chunk768:
                    return 768;
                case Chunk1024:
                    return 1024;
                case Chunk1536:
                    return 1536;
                case Chunk2176:
                    return 2176;
                case Chunk3584:
                    return 3584;
                case Chunk4096:
                    return 4096;
                case Chunk6528:
                    return 6528;
                    // 32k page with incremental allocation, 16 bytes alignment and page-wide ref-counting
                case ChunkIncremental:
                    return 0;
                default:
                    Y_FAIL();
            }
        }

        static ui32 TypeToSubpageSize(EChunkType chunkType) {
            switch (chunkType) {
                case ChunkUnknown:
                    return 0;
                case Chunk16:
                case Chunk32:
                case Chunk48:
                case Chunk64:
                case Chunk96:
                    return SmallChunkSize;
                case Chunk128:
                case Chunk192:
                case Chunk256:
                case Chunk384:
                case Chunk512:
                case Chunk768:
                case Chunk1024:
                case Chunk1536:
                case Chunk2176:
                case Chunk3584:
                case Chunk4096:
                case Chunk6528:
                case ChunkIncremental:
                    return LargeChunkSize;
                default:
                    Y_FAIL();
            }
        }

        static ui32 ChunksPerPage(EChunkType chunkType) {
            switch (chunkType) {
                case ChunkUnknown:
                    Y_VERIFY_DEBUG(false);
                    return 0;
                case Chunk16:
                    return 251;
                case Chunk32:
                    return 125;
                case Chunk48:
                    return 83;
                case Chunk64:
                    return 62;
                case Chunk96:
                    return 41;
                case Chunk128:
                    return 255;
                case Chunk192:
                    return 170;
                case Chunk256:
                    return 127;
                case Chunk384:
                    return 85;
                case Chunk512:
                    return 63;
                case Chunk768:
                    return 42;
                case Chunk1024:
                    return 31;
                case Chunk1536:
                    return 21;
                case Chunk2176:
                    return 15;
                case Chunk3584:
                    return 9;
                case Chunk4096:
                    return 0;
                case Chunk6528:
                    return 5;
                case ChunkIncremental:
                    Y_VERIFY_DEBUG(false);
                    return 0;
                default:
                    Y_FAIL();
            }
        }

        static ui32 FreeChunksToReleasePage(EChunkType chunkType) {
            return ChunksPerPage(chunkType) * 3 / 4;
        }

        // header stuff
        const ui32 Magic;     // 4 bytes
        const ui32 ChunkType; // 4 bytes

        // stuff
        // 2 writers (acts also as locks)
        TChunkQueue* QueueChunkWriter1; // 8 bytes (16)
        TChunkQueue* QueueChunkWriter2; // 8 bytes (24)
        // 2 non-locked readers
        TChunkQueue* QueueChunkReader1; // 8 bytes (32)
        TChunkQueue* QueueChunkReader2; // 8 bytes (40)

        ui8 Writer1Position;
        ui8 Writer2Position;
        ui8 Reader1Position;
        ui8 Reader2Position; // 4 bytes (44)

        // technicaly external queues could be replaced with chunk-embedded lists (or even bit-masks), but it's hard to argue about
        // so experiments left for future fine-tuning

        // ref-counter for incremental allocator
        ui32 Counter; // 4 bytes (48)

        // debug stuff
        ui32 Locked; // 4 bytes (52)

        // line header stuff
        const ui16 NumaNode;   // 2 bytes (54)
        const ui16 PoolIndex;  // 2 bytes (56)
        void* const LineStart; // 8 bytes (64)

        TChunkHeader(EChunkType chunkType, void* lineStart, ui16 numaNode, ui16 poolIndex)
            : Magic(MagicTag)
            , ChunkType(chunkType)
            , QueueChunkWriter1(nullptr)
            , QueueChunkWriter2(nullptr)
            , QueueChunkReader1(nullptr)
            , QueueChunkReader2(nullptr)
            , Writer1Position(0)
            , Writer2Position(0)
            , Reader1Position(0)
            , Reader2Position(0)
            , Counter(0)
            , Locked(0)
            , NumaNode(numaNode)
            , PoolIndex(poolIndex)
            , LineStart(lineStart)
        {
            const ui32 chunkSize = TypeToSize(chunkType);
            const ui32 subpageSize = TypeToSubpageSize(chunkType);
            ui8* thismem = (ui8*)this;

#ifdef NALF_ALLOC_DEBUG
            std::fill_n((ui64*)(this + 1), (subpageSize - 64) / 8, FreeMemoryMark);
#endif

            if (chunkType == ChunkIncremental) {
                // just do nothing, basic initialization is ok
            } else if (chunkType == Chunk16) {
                // for 16 byte chunks we allocate TChunkQueue parts inplace
                ui32 ms = 64;
                ui32 x = 1;
                QueueChunkReader1 = ::new (thismem + ms) TChunkQueue();
                ms += chunkSize;
                ++x;
                QueueChunkReader2 = ::new (thismem + ms) TChunkQueue();
                ms += chunkSize;
                ++x;
                QueueChunkWriter2 = QueueChunkReader2;

                TChunkQueue* writer = QueueChunkReader1;
                ui8 pos = 0;

                ui32 counter = 0;
                for (; ms + chunkSize < subpageSize; ++x, ms += chunkSize) {
                    if (pos != TChunkQueue::EntriesCount) {
                        writer->Entries[pos] = ms;
                        ++pos;
                    } else {
                        TChunkQueue* next = ::new (thismem + ms) TChunkQueue();
                        writer->Next = next;
                        writer = next;
                        pos = 0;
                        ++counter;
                    }
                }

                QueueChunkWriter1 = writer;
                Writer1Position = pos;
                Counter = counter;
            } else {
                QueueChunkReader1 = new TChunkQueue();
                QueueChunkReader2 = new TChunkQueue();
                QueueChunkWriter2 = QueueChunkReader2;

                TChunkQueue* writer = QueueChunkReader1;
                ui8 pos = 0;

                for (ui32 x = 1, ms = 64; ms + chunkSize < subpageSize; ++x, ms += chunkSize) {
                    if ((ms % SmallChunkSize) != 0) {
                        if (pos != TChunkQueue::EntriesCount) {
                            writer->Entries[pos] = ms;
                            ++pos;
                        } else {
                            TChunkQueue* next = new TChunkQueue();
                            next->Entries[0] = ms;
                            writer->Next = next;
                            writer = next;
                            pos = 1;
                        }
                    }
                }

                QueueChunkWriter1 = writer;
                Writer1Position = pos;
            }
        }

        ui32 PopBulkImpl(ui16* dest, ui32 count, TChunkQueue*& reader, ui8& readerPosition) {
            ui32 ret = 0;
            ui32 pos = AtomicLoad(&readerPosition);
            TChunkQueue* head = AtomicLoad(&reader);
            while (ret < count) {
                ui16 x = 0;
                if (pos != TChunkQueue::EntriesCount) {
                    x = AtomicLoad(&head->Entries[pos]);
                } else if (TChunkQueue* next = (TChunkQueue*)AtomicLoad(&head->Next)) {
                    if (ChunkType != Chunk16) {
                        x = AtomicLoad(&next->Entries[0]);
                        delete head;
                        head = next;
                        pos = 0;
                    } else {
                        Y_VERIFY_DEBUG((ui64(head) & ~SmallChunkSize) != (ui64)this);
                        // from self page
                        x = IdxFromPtr(this, 16, head);
#ifdef NALF_ALLOC_DEBUG
                        std::fill_n((ui64*)head, 2, FreeMemoryMark);
#endif
                        head = next;
                        pos = ui32(-1);
                    }
                }

                if (x != 0) {
                    *dest++ = x;
                    ++pos;
                    ++ret;
                } else
                    break;
            }
            if (ret) {
                AtomicStore(&readerPosition, (ui8)pos);
                AtomicStore(&reader, head);
            }
            return ret;
        }

        ui32 PopBulk(ui16* dest, ui32 count) {
            ui32 ret = 0;
            ret += PopBulkImpl(dest, count, QueueChunkReader1, Reader1Position);
            ret += PopBulkImpl(dest + ret, count - ret, QueueChunkReader2, Reader2Position);
            return ret;
        }

        void PushBulkImpl(const ui16* src, ui32 count, TChunkQueue*& writer, ui8& writerPosition) {
            // for chunk16 some part of returned objects could be used to construct queue-chunks
            ui8 pos = AtomicLoad(&writerPosition);
            for (const ui16* srcend = src + count; src != srcend; ++src) {
                const ui16 x = *src;
                if (pos != TChunkQueue::EntriesCount) {
                    AtomicStore(&writer->Entries[pos++], x);
                } else {
                    if (ChunkType != Chunk16) {
                        TChunkQueue* next = new TChunkQueue();
                        next->Entries[0] = x;
                        AtomicStore((void**)&writer->Next, (void*)next);
                        writer = next;
                        pos = 1;
                    } else {
                        TChunkQueue* next = ::new (PtrFromIdx(this, 16, x)) TChunkQueue();
                        AtomicStore((void**)&writer->Next, (void*)next);
                        writer = next;
                        pos = 0;
                    }
                }
            }
            AtomicStore(&writerPosition, pos);
        }

        // returns number of reclaimed chunks (as some part could be used for internal needs)
        void PushBulk(const ui16* src, ui32 count) {
            for (;;) {
                if (TChunkQueue* writer = AtomicLoad(&QueueChunkWriter1)) {
                    if (AtomicCas(&QueueChunkWriter1, nullptr, writer)) {
                        PushBulkImpl(src, count, writer, Writer1Position);
                        AtomicStore(&QueueChunkWriter1, writer);
                        return;
                    }
                }
                if (TChunkQueue* writer = AtomicLoad(&QueueChunkWriter2)) {
                    if (AtomicCas(&QueueChunkWriter2, nullptr, writer)) {
                        PushBulkImpl(src, count, writer, Writer2Position);
                        AtomicStore(&QueueChunkWriter2, writer);
                        return;
                    }
                }
                SpinLockPause();
            }
        }

        static void* PtrFromIdx(TChunkHeader* header, ui64 sz, ui16 idx) {
            Y_VERIFY_DEBUG(sz == TypeToSize((EChunkType)header->ChunkType));
            const ui64 shift = idx;
            void* const ret = (ui8*)header + shift;
            return ret;
        }

        static ui16 IdxFromPtr(TChunkHeader* header, ui32 sz, void* x) {
            Y_VERIFY_DEBUG(sz == TypeToSize((EChunkType)header->ChunkType));
            const ui64 shift = ui64((ui8*)x - (ui8*)header);
            const ui16 idx = shift;
            Y_VERIFY_DEBUG(x == PtrFromIdx(header, sz, idx));
            return idx;
        }

        // effective payload is 32704 bytes for 32k pages and 4032 for 4k pages
    };

    static_assert(sizeof(TChunkHeader) == 64, "expect sizeof(TChunkHeader) == 64");

}
