#pragma once

#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/methods/kernel/pointwise_hist2.cuh>
#include <catboost/cuda/methods/kernel/pointwise_scores.cuh>
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

        Y_SAVELOAD_DEFINE(BinValues, Bins, ReadIndices, WriteIndices, Cursor);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Cursor.Size() < (1ULL << 32));

            const ui32 cursorDim = Cursor.GetColumnCount();
            CB_ENSURE(BinValues.Size() % cursorDim == 0);
            const ui32 leavesCount = BinValues.Size() / cursorDim;
            NKernel::AddBinModelValue(BinValues.Get(), leavesCount,
                                      Bins.Get(),
                                      ReadIndices.Get(),
                                      WriteIndices.Get(),
                                      Cursor.Size(),
                                      Cursor.Get(),
                                      cursorDim,
                                      Cursor.AlignedColumnSize(),
                                      stream.GetStream());
        }
    };

    class TAddObliviousTreeKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> Features;
        TCudaBufferPtr<const ui8> Bins;
        TCudaBufferPtr<const float> Leaves;
        TCudaBufferPtr<const ui32> DataSet;
        TCudaBufferPtr<float> Cursor;
        TCudaBufferPtr<const ui32> ReadIndices;
        TCudaBufferPtr<const ui32> WriteIndices;

    public:
        TAddObliviousTreeKernel() = default;

        TAddObliviousTreeKernel(TCudaBufferPtr<const TCFeature> features,
                                TCudaBufferPtr<const ui8> bins,
                                TCudaBufferPtr<const float> leaves,
                                TCudaBufferPtr<const ui32> index,
                                TCudaBufferPtr<float> cursor,
                                TCudaBufferPtr<const ui32> readIndices = TCudaBufferPtr<const ui32>(),
                                TCudaBufferPtr<const ui32> writeIndices = TCudaBufferPtr<const ui32>())
            : Features(features)
            , Bins(bins)
            , Leaves(leaves)
            , DataSet(index)
            , Cursor(cursor)
            , ReadIndices(readIndices)
            , WriteIndices(writeIndices)
        {
        }

        Y_SAVELOAD_DEFINE(Features, Bins, Leaves, DataSet, Cursor, ReadIndices, WriteIndices);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Cursor.Size() < (1ULL << 32));
            CB_ENSURE(Bins.Size() == Features.Size());
            NKernel::AddObliviousTree(Features.Get(), Bins.Get(), Leaves.Get(), (ui32)Bins.Size(), DataSet.Get(),
                                      ReadIndices.Get(), WriteIndices.Get(), Cursor.Size(), Cursor.Get(),
                                      Cursor.GetColumnCount(), Cursor.AlignedColumnSize(),
                                      stream.GetStream());
        }
    };

    class TComputeObliviousTreeLeaveIndicesKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const TCFeature> Features;
        TCudaBufferPtr<const ui8> Bins;
        TCudaBufferPtr<const ui32> DataSet;
        TCudaBufferPtr<ui32> Cursor;
        TCudaBufferPtr<const ui32> ReadIndices;
        TCudaBufferPtr<const ui32> WriteIndices;

    public:
        TComputeObliviousTreeLeaveIndicesKernel() = default;

        TComputeObliviousTreeLeaveIndicesKernel(TCudaBufferPtr<const TCFeature> features,
                                                TCudaBufferPtr<const ui8> bins,
                                                TCudaBufferPtr<const ui32> index,
                                                TCudaBufferPtr<ui32> cursor,
                                                TCudaBufferPtr<const ui32> readIndices = TCudaBufferPtr<const ui32>(),
                                                TCudaBufferPtr<const ui32> writeIndices = TCudaBufferPtr<const ui32>())
            : Features(features)
            , Bins(bins)
            , DataSet(index)
            , Cursor(cursor)
            , ReadIndices(readIndices)
            , WriteIndices(writeIndices)
        {
        }

        Y_SAVELOAD_DEFINE(Features, Bins, DataSet, Cursor, ReadIndices, WriteIndices);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Cursor.Size() < (1ULL << 32));
            CB_ENSURE(Bins.Size() == Features.Size());
            CB_ENSURE(Bins.Size() < 32, "Tree depth " << Bins.Size() << " is too large, should be < 32");
            NKernel::ComputeObliviousTreeBins(Features.Get(),
                                              Bins.Get(),
                                              (ui32)Bins.Size(), // depth
                                              DataSet.Get(),
                                              ReadIndices.Get(),
                                              WriteIndices.Get(),
                                              Cursor.Get(),
                                              Cursor.Size(),
                                              stream.GetStream());
        }
    };
}

template <class TMapping, class Uint = ui32>
inline void AddBinModelValues(const TCudaBuffer<float, NCudaLib::TMirrorMapping>& leafValues,
                              const TCudaBuffer<ui32, TMapping>& bins,
                              const TCudaBuffer<Uint, TMapping>& readIndices,
                              TCudaBuffer<float, TMapping>& cursor,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TAddBinModelValueKernel;
    LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, leafValues, bins, cursor, readIndices);
}

template <class TMapping, class Uint = ui32>
inline void AddBinModelValues(const TCudaBuffer<float, NCudaLib::TMirrorMapping>& leafValues,
                              const TCudaBuffer<Uint, TMapping>& bins,
                              TCudaBuffer<float, TMapping>& cursor,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TAddBinModelValueKernel;
    LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, leafValues, bins, cursor);
}

template <class TUi32>
inline void AddObliviousTree(const TCudaBuffer<TUi32, NCudaLib::TStripeMapping>& dataSet,
                             const TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping>& features,
                             const TCudaBuffer<ui8, NCudaLib::TMirrorMapping>& bins,
                             const TCudaBuffer<float, NCudaLib::TMirrorMapping>& leaves,
                             TCudaBuffer<float, NCudaLib::TStripeMapping>& cursor,
                             ui32 stream = 0) {
    using TKernel = NKernelHost::TAddObliviousTreeKernel;
    LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, features, bins, leaves, dataSet, cursor);
}

template <class TUi32>
inline void ComputeObliviousTreeLeaves(const TCudaBuffer<TUi32, NCudaLib::TStripeMapping>& dataSet,
                                       const TCudaBuffer<const TCFeature, NCudaLib::TStripeMapping>& features,
                                       const TCudaBuffer<ui8, NCudaLib::TMirrorMapping>& bins,
                                       TCudaBuffer<ui32, NCudaLib::TStripeMapping>& cursor,
                                       ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeObliviousTreeLeaveIndicesKernel;
    LaunchKernels<TKernel>(cursor.NonEmptyDevices(), stream, features, bins, dataSet, cursor);
}
