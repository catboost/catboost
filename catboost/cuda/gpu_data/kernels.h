#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/kernel/binarize.cuh>
#include <catboost/cuda/cuda_util/compression_helpers_gpu.h>
#include <catboost/cuda/data/grid_creator.h>
#include <catboost/libs/options/binarization_options.h>

namespace NKernelHost {
    class TFindBordersKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Feature;
        EBorderSelectionType BorderType;
        ui32 BorderCount;
        TCudaBufferPtr<float> Dst;

    public:
        TFindBordersKernel() = default;

        TFindBordersKernel(TCudaBufferPtr<const float> feature,
                           EBorderSelectionType borderType,
                           ui32 borderCount,
                           TCudaBufferPtr<float> dst)
            : Feature(feature)
            , BorderType(borderType)
            , BorderCount(borderCount)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(Feature, Dst, BorderCount, BorderType);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Dst.Size() > BorderCount);

            if (BorderType == EBorderSelectionType::Median) {
                NKernel::FastGpuBorders(Feature.Get(), Feature.Size(), Dst.Get(),
                                        BorderCount, stream.GetStream());
            } else if (BorderType == EBorderSelectionType::Uniform) {
                NKernel::ComputeUniformBorders(Feature.Get(), static_cast<ui32>(Feature.Size()),
                                               Dst.Get(), BorderCount,
                                               stream.GetStream());
            } else {
                ythrow TCatboostException() << "Error: unsupported binarization for combinations ctrs "
                                            << BorderType;
            }
        }
    };

    class TBinarizeFloatFeatureKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Values;
        TCudaBufferPtr<const float> Borders;
        TCFeature Feature;
        TCudaBufferPtr<ui32> Dst;
        TCudaBufferPtr<const ui32> GatherIndex;
        bool AtomicUpdate;

    public:
        TBinarizeFloatFeatureKernel() = default;

        TBinarizeFloatFeatureKernel(TCudaBufferPtr<const float> values, TCudaBufferPtr<const float> borders, TCFeature feature, TCudaBufferPtr<const ui32> gatherIndex, TCudaBufferPtr<ui32> dst, bool atomicUpdate)
            : Values(values)
            , Borders(borders)
            , Feature(feature)
            , Dst(dst)
            , GatherIndex(gatherIndex)
            , AtomicUpdate(atomicUpdate)
        {
        }

        Y_SAVELOAD_DEFINE(Values, Borders, Feature, Dst, GatherIndex, AtomicUpdate);

        void Run(const TCudaStream& stream) const {
            NKernel::BinarizeFloatFeature(Values.Get(), static_cast<ui32>(Values.Size()),
                                          Borders.Get(), Feature,
                                          Dst.Get(),
                                          GatherIndex.Get(),
                                          AtomicUpdate,
                                          stream.GetStream());
        }
    };

    template <NCudaLib::EPtrType Type>
    class TWriteCompressedIndexKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui8, Type> Bins;
        TCFeature Feature;
        TCudaBufferPtr<ui32> Dst;

    public:
        TWriteCompressedIndexKernel() = default;

        TWriteCompressedIndexKernel(TCudaBufferPtr<const ui8, Type> bins,
                                    TCFeature feature,
                                    TCudaBufferPtr<ui32> cindex)
            : Bins(bins)
            , Feature(feature)
            , Dst(cindex)
        {
        }

        Y_SAVELOAD_DEFINE(Bins, Feature, Dst);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Feature.Mask != 0);
            CB_ENSURE(Feature.Offset != (ui64)(-1));
            NKernel::WriteCompressedIndex(Feature,
                                          Bins.Get(),
                                          Bins.Size(),
                                          Dst.Get(),
                                          stream.GetStream());
        }
    };
}

template <class TFloat, class TMapping>
inline void ComputeBordersOnDevice(const TCudaBuffer<TFloat, TMapping>& feature,
                                   const NCatboostOptions::TBinarizationOptions& description,
                                   TCudaBuffer<float, TMapping>& dst,
                                   ui32 stream = 0) {
    LaunchKernels<NKernelHost::TFindBordersKernel>(feature.NonEmptyDevices(), stream, feature, description.BorderSelectionType, description.BorderCount, dst);
}

template <class TValuesFloatType, class TBordersFloatType, class TUi32, class TMapping>
inline void BinarizeOnDevice(const TCudaBuffer<TValuesFloatType, TMapping>& featureValues,
                             const TCudaBuffer<TBordersFloatType, TMapping>& borders,
                             const NCudaLib::TDistributedObject<TCFeature>& feature,
                             TCudaBuffer<ui32, TMapping>& dst,
                             bool atomicUpdate,
                             const TCudaBuffer<TUi32, TMapping>* gatherIndices,
                             ui32 stream = 0) {
    using TKernel = NKernelHost::TBinarizeFloatFeatureKernel;
    LaunchKernels<TKernel>(featureValues.NonEmptyDevices(), stream, featureValues, borders, feature, gatherIndices, dst, atomicUpdate);
};

template <class TUi32,
          class TBinsBuffer,
          class TMapping>
inline void WriteCompressedFeature(const NCudaLib::TDistributedObject<TCFeature>& feature,
                                   const TBinsBuffer& bins,
                                   TCudaBuffer<TUi32, TMapping>& cindex,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TWriteCompressedIndexKernel<TBinsBuffer::PtrType()>;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, feature, cindex);
};
