#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {



    struct TSegmentedRadixSortContext : public IKernelContext {

        TSegmentedRadixSortContext(ui32 firstBit, ui32 lastBit,
                                   bool descending) {
            FirstBit = firstBit;
            LastBit = lastBit;
            Descending = descending;
        }

        ui32 FirstBit = 0;
        ui32 LastBit = 0;
        ui64 TempStorageSize = 0;

        TDevicePointer<char> TempStorage;

        bool Descending = false;


        TSegmentedRadixSortContext() = default;
    };


    template <typename K, typename V>
    cudaError_t SegmentedRadixSort(K *keys, V *values, K *tmpKeys, V *tmpValues, int size,
                                   const ui32* segmentStarts,  const ui32* segmentEnds, int segmentCount,
                                   TSegmentedRadixSortContext& context, TCudaStream stream);
}
