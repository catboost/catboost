#include "partitions.cuh"
#include "fill.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>



namespace NKernel {


    __global__ void UpdatePartitionSizes(TDataPartition* parts, ui32 partCount,
                                         const ui32* sortedBins, ui32 size) {

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < size) {
            ui32 bin0 = __ldg(sortedBins + i);
            ui32 bin1 = i ? __ldg(sortedBins + i - 1) : 0;

            if (bin0 != bin1) {
                ui32 b = bin1;
                while (b < bin0) {
                    parts[b].Size = i - parts[b].Offset;
                    b++;
                }
            }
            if ((i + 1) == size) {
                parts[bin0].Size = size - parts[bin0].Offset;
                ui32 b = bin0 + 1;
                while (b < partCount) {
                    parts[b].Size = 0;
                    b++;
                }
            }
            i += blockDim.x * gridDim.x;
        }
    }


    __global__ void ComputeSizes(ui32* beginOffsets, ui32* endOffsets, ui32 count, float* dst) {

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) {
            dst[i] = static_cast<float>(endOffsets[i] - beginOffsets[i]);
        }
    }

    struct TPartitionOffsetWriter {
        using TStorageType = TDataPartition;
        TDataPartition* Parts;

        __device__ TPartitionOffsetWriter(TDataPartition* parts)
                : Parts(parts) {

        }

        __device__ void Write(ui32 bin, ui32 offset) {
            Parts[bin].Offset = offset;
        }
    };

    struct TVecOffsetWriter {
        using TStorageType = ui32;
        ui32* BinOffsets;


        __device__ TVecOffsetWriter(ui32* offsets)
                : BinOffsets(offsets) {

        }

        __device__ void Write(ui32 bin, ui32 offset) {
            BinOffsets[bin] = offset;
        }
    };




    template <class TWriter, bool DONT_WRITE_EMPTY_SUFFIX>
    __global__ void UpdatePartitionOffsets(typename TWriter::TStorageType* parts, ui32 partCount,
                                           const ui32* sortedBins, ui32 size) {

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        ui32 lastBin = DONT_WRITE_EMPTY_SUFFIX ? LdgWithFallback(sortedBins + size - 1, 0) : UINT32_MAX;
        TWriter writer(parts);

        while (i < size) {
            ui32 bin0 = __ldg(sortedBins + i);
            ui32 bin1 = i ? __ldg(sortedBins + i - 1) : UINT32_MAX;
            if (bin0 != bin1) {
                ui32 b = bin0;
                while (b != bin1) {
                    writer.Write(b, i);
                    b--;
                }
            }
            if (i + 1 == size) {
                ui32 b = bin0 + 1;
                while (b < min(lastBin, partCount)) {
                    writer.Write(b, size);
                    b++;
                }
            }
            i += blockDim.x * gridDim.x;
        }
    }


    __global__ void ZeroPartitions(TDataPartition* __restrict parts, ui32 partCount)
    {
        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        while (i < partCount) {
            parts[i].Size = 0;
            parts[i].Offset = 0;
            i += blockDim.x * gridDim.x;
        }
    }

    void UpdatePartitionDimensions(TDataPartition* parts, ui32 partCount,
                                   const ui32* sortedBins, ui32 size,
                                   TCudaStream stream)
    {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize, (ui32)TArchProps::MaxBlockCount());
        if (numBlocks)
        {
            UpdatePartitionOffsets<TPartitionOffsetWriter, false> << < numBlocks, blockSize, 0, stream >> > (parts, partCount, sortedBins, size);
            UpdatePartitionSizes << < numBlocks, blockSize, 0, stream >> > (parts, partCount, sortedBins, size);
        } else {
            const ui32 numBlocksClear = (partCount + blockSize - 1) / blockSize;
            ZeroPartitions<<<numBlocksClear, blockSize, 0, stream>>>(parts, partCount);
        }
    }

    __global__ void ComputeSegmentSizesImpl(const ui32* beginOffsets, const ui32* endOffsets, ui32 count, float* dst) {

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < count) {
            dst[i] = static_cast<float>(endOffsets[i] - beginOffsets[i]);
        }
    }

    void ComputeSegmentSizes(const ui32* offsets, ui32 size,
                             float* dst, TCudaStream stream) {
        size -= 1;
        const ui32* begin = offsets;
        const ui32* end = offsets + 1;

        const ui32 blockSize = 256;
        const ui32 numBlocks = (size + blockSize - 1) / blockSize;
        ComputeSegmentSizesImpl <<< numBlocks, blockSize, 0, stream >>> (begin, end, size, dst);
    }

    void UpdatePartitionOffsets(ui32* partOffsets, ui32 partCount,
                                const ui32* sortedBins, ui32 size, TCudaStream stream)
    {
        const ui32 blockSize = 256;
        const ui32 numBlocks = min((size + blockSize - 1) / blockSize, (ui32)TArchProps::MaxBlockCount());
        //partOffsets are copyMapping of bins, usually with empty tail
        bool skipSuffixBins = false;

        if (numBlocks)
        {
            if (partCount == size) {
                FillBuffer(partOffsets, size, size, stream);
                skipSuffixBins = true;
            }
            if (skipSuffixBins)
            {
                UpdatePartitionOffsets<TVecOffsetWriter, true> << < numBlocks, blockSize, 0, stream >>>(partOffsets, partCount, sortedBins, size);
            } else {
                UpdatePartitionOffsets<TVecOffsetWriter, false> << < numBlocks, blockSize, 0, stream >>>(partOffsets, partCount, sortedBins, size);
            }
        } else {
            FillBuffer(partOffsets, static_cast<ui32>(0), partCount, stream);
        }
    }


}
