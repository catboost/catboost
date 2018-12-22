#pragma once

#include <catboost/cuda/cuda_lib/cuda_base.h>

namespace NCudaLib {
    //use it only for cuda-memcheck runs, otherwise our custom solution should be faster
    template <EPtrType PtrType>
    class TCudaMallocWrapper {
    public:
        explicit TCudaMallocWrapper(ui64 memorySize) {
            Y_UNUSED(memorySize);
            CATBOOST_WARNING_LOG << "We don't support maxMemorySize for CUDA-malloc wrapper" << Endl;
        }
        template <typename T>
        class TMemoryBlock: private TNonCopyable {
        private:
            T* Data;
            ui64 DataSize;

        public:
            TMemoryBlock(ui64 dataSize)
                : DataSize(dataSize)
            {
                Data = TCudaMemoryAllocation<PtrType>::template Allocate<T>(dataSize);
            }

            ~TMemoryBlock() {
                TCudaMemoryAllocation<PtrType>::FreeMemory(Data);
            }

            T* Get() {
                return Data;
            }

            const T* Get() const {
                return Data;
            }

            ui64 Size() const {
                return DataSize;
            }
        };

        ui64 GetRequestedRamSize() const {
            size_t freeBytes;
            size_t totalBytes;
            CUDA_SAFE_CALL(cudaMemGetInfo(&freeBytes, &totalBytes));
            return totalBytes;
        }

        ui64 GetFreeMemorySize() const {
            size_t freeBytes;
            size_t totalBytes;
            CUDA_SAFE_CALL(cudaMemGetInfo(&freeBytes, &totalBytes));
            return freeBytes;
        }

        void TryDefragment() {
        }

        template <typename T = char>
        TMemoryBlock<T>* Create(ui64 size) {
            return new TMemoryBlock<T>(size);
        }

        //cudaMalloc implicity synchronize, so we'll do it manually also
        template <class T>
        bool NeedSyncForAllocation(ui64 size) const {
            Y_UNUSED(size);
            return true;
        }
    };
}
