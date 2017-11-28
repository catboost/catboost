#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/methods/kernel/pointwise_hist2.cuh>
#include <catboost/cuda/methods/kernel/pointwise_scores.cuh>
#include <catboost/cuda/cuda_util/compression_helpers.h>
#include <catboost/cuda/gpu_data/binarized_dataset.h>
#include <catboost/cuda/models/kernel/add_model_value.cuh>

namespace NKernelHost {
    class TAddBinModelValueKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> BinValues;
        TCudaBufferPtr<const ui32> Bins;
        TCudaBufferPtr<const ui32> ReadIndices;
        TCudaBufferPtr<const ui32> WriteIndices;
        TCudaBufferPtr<float> Cursor;

    public:
        TAddBinModelValueKernel() = default;

        TAddBinModelValueKernel(TCudaBufferPtr<const float> binValues,
                                TCudaBufferPtr<const ui32> bins,
                                TCudaBufferPtr<float> cursor,
                                TCudaBufferPtr<const ui32> readIndices = TCudaBufferPtr<const ui32>(),
                                TCudaBufferPtr<const ui32> writeIndices = TCudaBufferPtr<const ui32>())
            : BinValues(binValues)
            , Bins(bins)
            , ReadIndices(readIndices)
            , WriteIndices(writeIndices)
            , Cursor(cursor)
        {
        }

        SAVELOAD(BinValues, Bins, ReadIndices, WriteIndices, Cursor);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Cursor.Size() < (1ULL << 32));

            NKernel::AddBinModelValue(BinValues.Get(), BinValues.Size(),
                                      Bins.Get(),
                                      ReadIndices.Get(),
                                      WriteIndices.Get(),
                                      Cursor.Get(),
                                      Cursor.Size(),
                                      stream.GetStream());
        }
    };
}

template <class TMapping, class Uint = ui32>
inline void AddBinModelValues(TCudaBuffer<float, TMapping>& cursor,
                              const TCudaBuffer<float, NCudaLib::TMirrorMapping>& leafValues,
                              const TCudaBuffer<ui32, TMapping>& bins,
                              const TCudaBuffer<Uint, TMapping>& readIndices,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TAddBinModelValueKernel;
    LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, leafValues, bins, cursor, readIndices);
}

template <class TMapping, class Uint = ui32>
inline void AddBinModelValues(TCudaBuffer<float, TMapping>& cursor,
                              const TCudaBuffer<float, NCudaLib::TMirrorMapping>& leafValues,
                              const TCudaBuffer<ui32, TMapping>& bins,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TAddBinModelValueKernel;
    LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, leafValues, bins, cursor);
}
