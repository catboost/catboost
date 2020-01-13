#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>
#include <util/generic/vector.h>
#include <util/string/builder.h>
#include <util/datetime/base.h>
/*
 * For most GPU task we don't need sophisticated allocators.
 * We just need to allocated big bunch of memory, do some job and deallocate it.
 * Moreover, memory copy is pretty fast operation in GPU memory.
 * So for buffers we use simple stack-based scheme for memory allocation. We greedy allocate memory in last free block while we can.
 * If we don't have enough memory - we simply defragment it
 */

namespace NCudaLib {
    template <EPtrType PtrType>
    class TStackLikeMemoryPool: private TNonCopyable {
    private:
        static constexpr ui32 MEMORY_ALIGMENT_BYTES = 256;
        static constexpr double MB = 1024.0 * 1024.0;
        static constexpr ui64 MINIMUM_FREE_MEMORY_TO_DEFRAGMENTATION = (ui64)(16 * MB);
        static constexpr ui64 MEMORY_REQUEST_ADJUSTMENT = MEMORY_ALIGMENT_BYTES * 32;

        struct TAllocatedBlock: public TSimpleRefCount<TAllocatedBlock>, public TNonCopyable {
            char* Ptr;
            ui64 Size;
            bool IsFree;

            //Most part of memory is static (module unknown dataset size). Heuristic for fast search of dynamic memory blocks start.
            TIntrusivePtr<TAllocatedBlock> Next = nullptr;
            TIntrusivePtr<TAllocatedBlock> Prev = nullptr;

            void UpdateNeighboursLinks() {
                if (Next && (Next->Prev) != this) {
                    Next->Prev = this;
                }
                if (Prev && ((Prev->Next) != this)) {
                    Prev->Next = this;
                }
            }

            TAllocatedBlock(char* ptr,
                            ui64 size,
                            bool isFree)
                : Ptr(ptr)
                , Size(size)
                , IsFree(isFree)
            {
            }
        };

        TIntrusivePtr<TAllocatedBlock> FirstFreeBlock;

        TIntrusivePtr<TAllocatedBlock> FindFirstFreeBlock(TAllocatedBlock* cursor) {
            while (cursor != nullptr && !cursor->IsFree) {
                cursor = cursor->Next.Get();
            }
            Y_ASSERT(cursor);
            Y_ASSERT(cursor->Ptr <= LastBlock->Ptr);

            return TIntrusivePtr<TAllocatedBlock>(cursor);
        }

        //splits free block to allocated and free
        TIntrusivePtr<TAllocatedBlock> SplitFreeBlock(TIntrusivePtr<TAllocatedBlock> block,
                                                      ui64 size) {
            CB_ENSURE(block->IsFree, "Error: block is not free");
            CB_ENSURE(block->Size >= size, "Error: can't split block: " << block->Size << "-" << size);
            TIntrusivePtr<TAllocatedBlock> leftBlock;

            if (size == block->Size) {
                leftBlock = block;
                leftBlock->IsFree = false;
                Y_ASSERT(FirstFreeBlock->Ptr <= LastBlock->Ptr);
            } else {
                CB_ENSURE(block->IsFree);
                leftBlock = new TAllocatedBlock(block->Ptr, size, false);

                block->Ptr += size;
                block->Size -= size;

                leftBlock->Prev = block->Prev;
                leftBlock->Next = block;
                leftBlock->UpdateNeighboursLinks();
                Y_ASSERT(FirstFreeBlock->Ptr <= LastBlock->Ptr);
            }

            if (FirstFreeBlock->Ptr == leftBlock->Ptr) {
                FirstFreeBlock = FindFirstFreeBlock(leftBlock.Get());
                Y_ASSERT(FirstFreeBlock);
            }

            return leftBlock;
        }

        void MergeFreeBlocks(TIntrusivePtr<TAllocatedBlock> block) {
            CB_ENSURE(block->IsFree, "Block should be free");

            auto cursor = block;

            while (cursor->Prev != nullptr && cursor->Prev->IsFree) {
                auto toRemove = cursor;
                cursor = cursor->Prev;
                cursor->Next = toRemove->Next;
                cursor->Size += toRemove->Size;
                cursor->UpdateNeighboursLinks();
            }

            while (cursor->Next != nullptr && cursor->Next->IsFree) {
                auto toRemove = cursor;

                cursor = cursor->Next;
                cursor->Prev = toRemove->Prev;
                cursor->Ptr = toRemove->Ptr;
                cursor->Size += toRemove->Size;
                cursor->UpdateNeighboursLinks();
            }

            if (FirstFreeBlock->Ptr >= cursor->Ptr) {
                Y_ASSERT(cursor && cursor->IsFree);
                FirstFreeBlock = cursor;
            }
            Y_ASSERT(FirstFreeBlock->Ptr <= LastBlock->Ptr);
        }

        char* Memory = nullptr;
        ui64 TotalMemory;
        ui64 FreeMemory;
        TIntrusivePtr<TAllocatedBlock> LastBlock;

        ui64 CalculateFragmentedMemorySize() const {
            ui64 fragmentedMemory = 0;
            TAllocatedBlock* cursor = FirstFreeBlock.Get();

            while (cursor != nullptr) {
                if (cursor->IsFree && cursor->Next.Get() != nullptr) {
                    fragmentedMemory += cursor->Size;
                }
                cursor = cursor->Next.Get();
            }
            return fragmentedMemory;
        }

        void MemoryDefragmentation() {
            GetDefaultStream().Synchronize();
            auto startTime = Now();

            auto cursor = FirstFreeBlock;
            CB_ENSURE(cursor != nullptr);
            auto last = LastBlock;

            char* const startPtr = cursor->Ptr;

            ui64 writeOffset = 0;
            ui64 newBlockOffset = 0;

            char* const temp = last->Ptr;
            const ui64 tempBufferSize = (last->Size / 4096) * 4096;
            ui64 tempBufferUsed = 0;

            while (cursor != last) {
                if (!cursor->IsFree) {
                    ui64 movedSize = 0;

                    while (movedSize < cursor->Size) {
                        const ui64 tempBufferFreeSpace = tempBufferSize - tempBufferUsed;
                        ui64 copySize = std::min(tempBufferFreeSpace, cursor->Size - movedSize);
                        Y_ASSERT((temp + tempBufferUsed - cursor->Ptr - movedSize - copySize) >= 0);
                        Y_ASSERT((temp + tempBufferUsed + copySize) <= (temp + tempBufferSize));
                        TMemoryCopier<PtrType, PtrType>::CopyMemorySync(cursor->Ptr + movedSize, temp + tempBufferUsed, copySize);

                        movedSize += copySize;
                        tempBufferUsed += copySize;
                        Y_ASSERT(tempBufferUsed <= tempBufferSize);

                        if (tempBufferUsed == tempBufferSize) {
                            TMemoryCopier<PtrType, PtrType>::CopyMemorySync(temp, startPtr + writeOffset, tempBufferUsed);
                            GetDefaultStream().Synchronize();
                            CheckLastError();
                            writeOffset += tempBufferUsed;
                            tempBufferUsed = 0;
                        }
                    }
                    Y_ASSERT(movedSize == cursor->Size);

                    cursor->Ptr = startPtr + newBlockOffset;
                    newBlockOffset += cursor->Size;
                } else {
                    cursor->Prev->Next = cursor->Next;
                    cursor->Prev->UpdateNeighboursLinks();
                }
                cursor = cursor->Next;
            }

            if (tempBufferUsed) {
                TMemoryCopier<PtrType, PtrType>::CopyMemorySync(temp, startPtr + writeOffset, tempBufferUsed);
                writeOffset += tempBufferUsed;
                tempBufferUsed = 0;
            }

            const ui64 defragmentedMemory = (temp - (startPtr + writeOffset));
            GetDefaultStream().Synchronize();
            CATBOOST_DEBUG_LOG << "Defragment " << defragmentedMemory * 1.0 / 1024 / 1024 << " memory"
                               << " in " << (Now() - startTime).SecondsFloat() << " seconds " << Endl;
            LastBlock->Size += defragmentedMemory;
            LastBlock->Ptr = startPtr + writeOffset;

            CB_ENSURE(LastBlock == cursor);
            FirstFreeBlock = LastBlock;
        }

        template <class T>
        static constexpr ui64 GetBlockSize(ui64 elemCount) {
            return ((sizeof(T) * elemCount + MEMORY_ALIGMENT_BYTES - 1) / MEMORY_ALIGMENT_BYTES) *
                   MEMORY_ALIGMENT_BYTES;
        }

    public:
        template <typename T>
        class TMemoryBlock: private TNonCopyable {
        private:
            TIntrusivePtr<TAllocatedBlock> Block;
            TStackLikeMemoryPool& Owner;

        private:
            TMemoryBlock(TIntrusivePtr<TAllocatedBlock> block,
                         TStackLikeMemoryPool& owner)
                : Block(block)
                , Owner(owner)
            {
            }

        public:
            ~TMemoryBlock() {
                if (Block) {
                    Block->IsFree = true;
                    Owner.FreeMemory += Block->Size;
                    Owner.MergeFreeBlocks(Block);
                }
            }

            T* Get() {
                return Block->Ptr;
            }

            const T* Get() const {
                return Block->Ptr;
            }

            ui64 MaxSize() const {
                return Block->Size;
            }

            template<EPtrType PType>
            friend class TStackLikeMemoryPool;
        };

    public:
        explicit TStackLikeMemoryPool(ui64 memorySize) {
            Memory = TCudaMemoryAllocation<PtrType>::template Allocate<char>(memorySize);
            LastBlock = new TAllocatedBlock(Memory, memorySize, true);
            FirstFreeBlock = LastBlock;
            TotalMemory = memorySize;
            FreeMemory = TotalMemory;
        }

        ~TStackLikeMemoryPool() noexcept(false) {
            TAllocatedBlock* block = LastBlock.Get();

            while (block != nullptr) {
                if (!block->IsFree) {
                    ythrow yexception() << "Error: can't deallocate memory. It's still used in some buffer";
                }
                block = block->Prev.Get();
            }

            TCudaMemoryAllocation<PtrType>::FreeMemory(Memory);
        }

        template <class T>
        bool NeedSyncForAllocation(ui64 size) const {
            const ui64 requestedBlockSize = GetBlockSize<T>(size) + MEMORY_REQUEST_ADJUSTMENT;
            const bool canUseFirstFreeBlock = FirstFreeBlock != LastBlock && (FirstFreeBlock->Size >= requestedBlockSize);
            return (LastBlock->Size < requestedBlockSize || ((LastBlock->Size - requestedBlockSize) <= MINIMUM_FREE_MEMORY_TO_DEFRAGMENTATION)) && !canUseFirstFreeBlock;
        }

        template <typename T = char>
        TMemoryBlock<T>* Create(ui64 size) {
            ui64 requestedBlockSize = GetBlockSize<T>(size);

            TIntrusivePtr<TAllocatedBlock> block = nullptr;

            if (FirstFreeBlock->Size >= requestedBlockSize && (FirstFreeBlock != LastBlock)) {
                Y_ASSERT(FirstFreeBlock->IsFree);
                block = SplitFreeBlock(FirstFreeBlock, requestedBlockSize);
            } else {
                const ui64 adjustedMemoryRequestSize = (requestedBlockSize + MEMORY_REQUEST_ADJUSTMENT);
                const bool needDefragment = (LastBlock->Size < adjustedMemoryRequestSize || ((LastBlock->Size - requestedBlockSize) <= MINIMUM_FREE_MEMORY_TO_DEFRAGMENTATION));
                if (needDefragment) {
                    TryDefragment();
                }
                if (LastBlock->Size < adjustedMemoryRequestSize) {
                    ythrow TOutOfMemoryError() << "Error: Out of memory. Requested " << requestedBlockSize / MB << " MB; Free "
                                               << (LastBlock->Size) / MB << " MB";
                }
                block = SplitFreeBlock(LastBlock, requestedBlockSize);
                Y_ASSERT(FirstFreeBlock->Ptr <= LastBlock->Ptr);
            }
            Y_ASSERT(FirstFreeBlock->Ptr <= LastBlock->Ptr);

            FreeMemory -= block->Size;
            return new TMemoryBlock<T>(block, *this);
        }

        void TryDefragment() {
            const ui64 memoryToDefragment = CalculateFragmentedMemorySize();

            if (memoryToDefragment == 0) {
                return;
            }

            CATBOOST_DEBUG_LOG << "Starting memory defragmentation" << Endl;
            CATBOOST_DEBUG_LOG << "Fragmented memory " << memoryToDefragment * 1.0 / 1024 / 1024 << Endl;
            CATBOOST_DEBUG_LOG << "Free memory in last block " << LastBlock->Size * 1.0 / 1024 / 1024 << Endl;

            if ((memoryToDefragment > LastBlock->Size) && (LastBlock->Size < MINIMUM_FREE_MEMORY_TO_DEFRAGMENTATION)) {
                ythrow TOutOfMemoryError() << "Error: We don't have enough memory for defragmentation";
            } else {
                //this algorithm copies everything in last block and then copies back.
                MemoryDefragmentation();
            }
        }

        ui64 GetRequestedRamSize() const {
            return TotalMemory;
        }

        ui64 GetFreeMemorySize() const {
            return FreeMemory;
        }
    };

    extern template class TStackLikeMemoryPool<EPtrType::CudaDevice>;
    extern template class TStackLikeMemoryPool<EPtrType::CudaHost>;
}
