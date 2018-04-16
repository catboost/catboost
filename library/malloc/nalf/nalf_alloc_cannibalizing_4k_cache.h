#pragma once

#include "defs.h"
#include "nalf_alloc.h"

namespace NNumaAwareLockFreeAllocator {
    class TCannibalizing4kCache {
    public:
        static const ui32 Concurrency = 4;

    private:
        using TChunk = TQueueChunk<void*, 4096>;

        ui32 ReadPosition[Concurrency];
        TChunk* volatile ReadFrom[Concurrency];

        volatile ui32 WritePosition[Concurrency];
        TChunk* volatile WriteTo[Concurrency];

        static_assert(sizeof(TChunk*) == sizeof(TAtomic), "expect sizeof(TChunk*) == sizeof(TAtomic)");

        void LockWriter(TChunk*& writeTo, ui64& index, ui64 writerRotation) {
            Y_VERIFY_DEBUG(writeTo == nullptr);
            ui64 writerIndex = writerRotation;
            for (;;) {
                index = writerIndex % Concurrency;
                if (RelaxedLoad(&WriteTo[index]) != nullptr) {
                    if (writeTo = AtomicSwap(&WriteTo[index], nullptr))
                        return;
                }
                ++writerIndex;
            }
        }

        void UnlockWriter(TChunk* writeTo, ui64 index) {
            AtomicStore(&WriteTo[index], writeTo);
        }

        void WriteOne(TChunk*& writeTo, ui64 index, void* x) {
            Y_VERIFY_DEBUG(x != nullptr);
            const ui32 pos = AtomicLoad(&WritePosition[index]);
            if (pos != TChunk::EntriesCount) {
                AtomicStore(&WritePosition[index], pos + 1);
                AtomicStore(&writeTo->Entries[pos], x);
            } else {
                TChunk* next = ::new (x) TChunk();
                AtomicStore(&WritePosition[index], 0u);
                AtomicStore(&writeTo->Next, next);
                writeTo = next;
            }
        }

    public:
        TCannibalizing4kCache() {
            for (ui32 i = 0; i != Concurrency; ++i) {
                ReadPosition[i] = 0;
                ReadFrom[i] = ::new (Allocate(sizeof(TChunk))) TChunk();

                WritePosition[i] = 0;
                WriteTo[i] = ReadFrom[i];
            }
        }

        ~TCannibalizing4kCache() {
            Y_FAIL("must not be destructed");
        }

        void* Pop(ui64 readerRotation) {
            ui64 readerIndex = readerRotation;
            const ui64 endIndex = readerIndex + Concurrency;
            for (; readerIndex != endIndex; ++readerIndex) {
                const ui64 i = readerIndex % Concurrency;
                if (ReadFrom[i] != nullptr) {
                    if (TChunk* readFrom = AtomicSwap(&ReadFrom[i], nullptr)) {
                        const ui32 pos = AtomicLoad(&ReadPosition[i]);
                        if (pos != TChunk::EntriesCount) {
                            if (void* ret = AtomicLoad(&readFrom->Entries[pos])) {
                                AtomicStore(&ReadPosition[i], pos + 1);
                                AtomicStore(&ReadFrom[i], readFrom); // release lock with same chunk
                                return ret;                          // found, return
                            } else {
                                AtomicStore(&ReadFrom[i], readFrom);
                            }
                        } else if (TChunk* next = AtomicLoad(&readFrom->Next)) {
                            // got next page entry, could consume current 'read-from' as chunk
                            AtomicStore(&ReadPosition[i], 0u);
                            AtomicStore(&ReadFrom[i], next); // release lock with new chunk
                            if (((ui64)readFrom % SystemPageSize) == 0) {
                                return (void*)readFrom;
                            } else {
                                Free(readFrom);
                            }
                        } else {
                            AtomicStore(&ReadFrom[i], readFrom); // nothing in old chunk and no next chunk (so we could not consume old one), just release lock with old chunk
                        }
                    }
                }
            }

            return nullptr; // got nothing after full cycle, return
        }

        void Push(void* x, ui64 writerRotation) {
            TChunk* writeTo = nullptr;
            ui64 index = 0;

            LockWriter(writeTo, index, writerRotation);
            WriteOne(writeTo, index, x);
            UnlockWriter(writeTo, index);
        }
    };

}
