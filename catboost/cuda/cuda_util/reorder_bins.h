#pragma once

#include "sort.h"
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_util/kernel/sort.cuh>
#include <catboost/libs/helpers/exception.h>

template <class TMapping>
inline void ReorderBins(TCudaBuffer<ui32, TMapping>& bins,
                        TCudaBuffer<ui32, TMapping>& indices,
                        ui32 offset,
                        ui32 bits,
                        ui64 stream = 0) {
    using TKernel = NKernelHost::TRadixSortKernel<ui32, ui32>;
    CB_ENSURE((offset + bits) <= (sizeof(ui32) * 8));
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, indices, false, offset, offset + bits);
}

template <class TMapping>
inline void ReorderBins(TCudaBuffer<ui32, TMapping>& bins,
                        TCudaBuffer<ui32, TMapping>& indices,
                        ui32 offset,
                        ui32 bits,
                        TCudaBuffer<ui32, TMapping>& tmpBins,
                        TCudaBuffer<ui32, TMapping>& tmpIndices,
                        ui64 stream = 0) {
    using TKernel = NKernelHost::TRadixSortKernel<ui32, ui32>;
    CB_ENSURE((offset + bits) <= (sizeof(ui32) * 8));
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, indices, false, offset, offset + bits, tmpBins, tmpIndices);
}
