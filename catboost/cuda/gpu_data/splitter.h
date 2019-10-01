#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/cuda/data/binarizations_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/kernel/split.cuh>
#include <catboost/cuda/cuda_util/compression_helpers_gpu.h>

namespace NKernelHost {
    class TWriteCompressedSplitKernel: public TStatelessKernel {
    private:
        TCFeature Feature;
        ui32 BinIdx;
        TCudaBufferPtr<const ui32> CompressedIndex;
        ui32 SampleCount;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<ui64> CompressedBits;

    public:
        TWriteCompressedSplitKernel() = default;

        TWriteCompressedSplitKernel(TCFeature feature,
                                    ui32 binIdx,
                                    TCudaBufferPtr<const ui32> compressedIndex,
                                    ui32 dataSetSize,
                                    TCudaBufferPtr<ui64> compressedBits,
                                    TCudaBufferPtr<const ui32> indices)
            : Feature(feature)
            , BinIdx(binIdx)
            , CompressedIndex(compressedIndex)
            , SampleCount(dataSetSize)
            , Indices(indices)
            , CompressedBits(compressedBits)
        {
        }

        Y_SAVELOAD_DEFINE(Feature, BinIdx, CompressedIndex, SampleCount, Indices, CompressedBits);

        void Run(const TCudaStream& stream) const {
            NKernel::WriteCompressedSplit(Feature,
                                          BinIdx,
                                          CompressedIndex.Get(),
                                          Indices.Get(),
                                          SampleCount,
                                          CompressedBits.Get(),
                                          stream.GetStream());
        }
    };

    class TWriteCompressedSplitFloatKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Values;
        float Border;
        TCudaBufferPtr<const ui32> Indices;
        TCudaBufferPtr<ui64> CompressedBits;

    public:
        TWriteCompressedSplitFloatKernel() = default;

        TWriteCompressedSplitFloatKernel(TCudaBufferPtr<const float> values,
                                         float border,
                                         TCudaBufferPtr<ui64> compressedBits, TCudaBufferPtr<const ui32> indices)
            : Values(values)
            , Border(border)
            , Indices(indices)
            , CompressedBits(compressedBits)
        {
        }

        Y_SAVELOAD_DEFINE(Values, Border, Indices, CompressedBits);

        void Run(const TCudaStream& stream) const {
            NKernel::WriteCompressedSplitFloat(Values.Get(), Border, Indices.Get(), static_cast<int>(Values.Size()),
                                               CompressedBits.Get(), stream.GetStream());
        }
    };

    class TUpdateBinsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui64> CompressedBits;
        ui32 Depth;
        TCudaBufferPtr<ui32> Bins;

    public:
        TUpdateBinsKernel() = default;

        TUpdateBinsKernel(TCudaBufferPtr<const ui64> compressedBits, ui32 depth, TCudaBufferPtr<ui32> bins)
            : CompressedBits(compressedBits)
            , Depth(depth)
            , Bins(bins)
        {
        }

        Y_SAVELOAD_DEFINE(CompressedBits, Depth, Bins);

        void Run(const TCudaStream& stream) const {
            NKernel::UpdateBins(CompressedBits.Get(), Depth, Bins.Get(),
                                static_cast<int>(Bins.Size()), stream.GetStream());
        }
    };

    class TUpdateBinsFromCompressedIndexKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Index;
        TCudaBufferPtr<const ui32> Indices;
        TCFeature Feature;
        ui32 BinIdx;
        ui32 Depth;
        TCudaBufferPtr<ui32> Bins;

    public:
        TUpdateBinsFromCompressedIndexKernel() = default;

        TUpdateBinsFromCompressedIndexKernel(TCudaBufferPtr<const ui32> cindex,
                                             TCudaBufferPtr<const ui32> indices,
                                             TCFeature feature, ui32 bin,
                                             ui32 depth,
                                             TCudaBufferPtr<ui32> bins)
            : Index(cindex)
            , Indices(indices)
            , Feature(feature)
            , BinIdx(bin)
            , Depth(depth)
            , Bins(bins)
        {
        }

        Y_SAVELOAD_DEFINE(Index, Indices, Feature, BinIdx, Depth, Bins);

        void Run(const TCudaStream& stream) const {
            NKernel::UpdateBinsFromCompressedIndex(Index.Get(), Indices.Get(), Indices.Size(),
                                                   Feature, BinIdx, Depth,
                                                   Bins.Get(), stream.GetStream());
        }
    };
}

template <class TDataSet,
          class TMapping>
inline void CreateCompressedSplit(const TDataSet& dataSet,
                                  const NCudaLib::TDistributedObject<TCFeature>& feature,
                                  ui32 bin,
                                  TCudaBuffer<ui64, TMapping>& bits,
                                  const TCudaBuffer<const ui32, TMapping>* readIndices = nullptr,
                                  ui32 stream = 0) {
    using TKernel = NKernelHost::TWriteCompressedSplitKernel;

    LaunchKernels<TKernel>(bits.NonEmptyDevices(), stream, feature, bin,
                           dataSet.GetCompressedIndex(),
                           dataSet.GetSampleCount(),
                           bits,
                           readIndices);
}

template <class TMapping, class TFloat>
inline void CreateCompressedSplitFloat(const TCudaBuffer<TFloat, TMapping>& feature, float border,
                                       TCudaBuffer<ui64, TMapping>& bits,
                                       const TCudaBuffer<const ui32, TMapping>* readIndices = nullptr,
                                       ui32 stream = 0) {
    using TKernel = NKernelHost::TWriteCompressedSplitFloatKernel;
    LaunchKernels<TKernel>(bits.NonEmptyDevices(), stream, feature, border, bits, readIndices);
}

template <class TMapping>
inline void UpdateBinFromCompressedBits(const TCudaBuffer<ui64, TMapping>& compressedBits,
                                        TCudaBuffer<ui32, TMapping>& dst,
                                        ui32 depth,
                                        ui32 streamId = 0) {
    using TKernel = NKernelHost::TUpdateBinsKernel;
    LaunchKernels<TKernel>(dst.NonEmptyDevices(), streamId, compressedBits, depth, dst);
}

inline void UpdateBinFromCompressedIndex(const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& cindex,
                                         const NCudaLib::TDistributedObject<TCFeature>& feature, ui32 bin,
                                         const TCudaBuffer<ui32, NCudaLib::TStripeMapping>& docs,
                                         ui32 depth,
                                         TCudaBuffer<ui32, NCudaLib::TStripeMapping>& bins,
                                         ui32 streamId = 0) {
    using TKernel = NKernelHost::TUpdateBinsFromCompressedIndexKernel;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), streamId, cindex, docs, feature, bin, depth, bins);
}
