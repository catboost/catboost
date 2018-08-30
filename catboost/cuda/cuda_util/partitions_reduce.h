#pragma once


#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_kernel_buffer.h>
#include <catboost/cuda/cuda_util/kernel/update_part_props.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>

namespace NKernelHost {

    class TReducePartitionsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Input;
        TCudaBufferPtr<const TDataPartition> Partitions;
        TCudaBufferPtr<const ui32> PartIds;
        TCudaBufferPtr<double> Output;
    public:
        TReducePartitionsKernel() = default;

        TReducePartitionsKernel(TCudaBufferPtr<const float> input,
                                TCudaBufferPtr<const TDataPartition> partitions,
                                TCudaBufferPtr<const ui32> partIds,
                                TCudaBufferPtr<double> output)
            : Input(input)
            , Partitions(partitions)
            , PartIds(partIds)
            , Output(output)
        {
        }

        Y_SAVELOAD_DEFINE(Input, Partitions, PartIds, Output);

        void Run(const TCudaStream& stream) const {
            NKernel::UpdatePartitionsProps(Partitions.Get(),
                                           PartIds.Get(),
                                           PartIds.Size(),
                                           Input.Get(),
                                           Input.GetColumnCount(),
                                           Input.AlignedColumnSize(),
                                           Output.Get(),
                                           stream.GetStream()
            );
        }
    };



    class TReducePartitionsWithOffsetsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Input;
        TCudaBufferPtr<const ui32> Offsets;
        TCudaBufferPtr<double> Output;
    public:
        TReducePartitionsWithOffsetsKernel() = default;

        TReducePartitionsWithOffsetsKernel(TCudaBufferPtr<const float> input,
                                           TCudaBufferPtr<const ui32> offsets,
                                           TCudaBufferPtr<double> output)
                : Input(input)
                , Offsets(offsets)
                , Output(output)
        {
        }

        Y_SAVELOAD_DEFINE(Input, Offsets, Output);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Input.GetColumnCount());
            CB_ENSURE(Offsets.Size() > 1);
            NKernel::UpdatePartitionsPropsForOffsets(Offsets.Get(), Offsets.Size() - 1, Input.Get(), Input.GetColumnCount(), Input.AlignedColumnSize(), Output.Get(), stream.GetStream());
        }
    };


    class TCastCopyKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Input;
        TCudaBufferPtr<double> Output;
    public:
        TCastCopyKernel() = default;

        TCastCopyKernel(TCudaBufferPtr<const float> input,
                           TCudaBufferPtr<double> output)
                : Input(input)
                  , Output(output)
        {
        }

        Y_SAVELOAD_DEFINE(Input, Output);

        void Run(const TCudaStream& stream) const {
            NKernel::CopyFloatToDouble(Input.Get(), Input.Size(), Output.Get(), stream.GetStream());
        }
    };




}

template <class TMapping>
inline void CastCopy(const TCudaBuffer<float, TMapping>& input,
                     TCudaBuffer<double, TMapping>* output,
                     ui32 streamId = 0) {
    using TKernel = NKernelHost::TCastCopyKernel;
    LaunchKernels<TKernel>(output->NonEmptyDevices(), streamId, input, output);
}




template <class TMapping>
inline void ComputePartitionStats(const TCudaBuffer<float, TMapping>& input,
                                  const TCudaBuffer<TDataPartition, TMapping>& parts,
                                  const TCudaBuffer<ui32, NCudaLib::TMirrorMapping>& partIds,
                                  TCudaBuffer<double, TMapping>* output,
                                  ui32 streamId = 0) {
    using TKernel = NKernelHost::TReducePartitionsKernel;
    LaunchKernels<TKernel>(output->NonEmptyDevices(), streamId, input, parts, partIds, output);
}



template <class TMapping, class TFloat>
inline void ComputePartitionStats(const TCudaBuffer<TFloat, TMapping>& input,
                                  const TCudaBuffer<ui32, TMapping>& offsets,
                                  TCudaBuffer<double, TMapping>* output,
                                  ui32 streamId = 0) {
    using TKernel = NKernelHost::TReducePartitionsWithOffsetsKernel;
    LaunchKernels<TKernel>(output->NonEmptyDevices(), streamId, input, offsets, output);
}
