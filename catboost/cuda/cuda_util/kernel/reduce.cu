#include "reduce.cuh"
#include "kernel_helpers.cuh"
#include <contrib/libs/cub/cub/device/device_reduce.cuh>
#include <contrib/libs/cub/cub/device/device_segmented_reduce.cuh>


namespace NKernel {


    //current cub segmented reduce sucks on small segments problems
    //LINE_SIZE should be leq 32
    //TODO(noxoomo): special version for by-thread reduction in case of 1-4 elements per segment
    //TODO(noxoomo): Fallback to block-reduce if one of segments is too big (e.g. loopSize > 256)
    template <typename T, int BLOCK_SIZE, int LINE_SIZE>
    __global__ void SegmentedReduceWarpPartPerSegmentImpl(const T* src,
                                                          const int* segmentStarts,
                                                          const int* segmentEnds,
                                                          ui32 segmentsCount,
                                                          T* reducedSegments) {
        __shared__  T localBufferStorage[BLOCK_SIZE];
        const int tid = threadIdx.x;
        localBufferStorage[tid] = 0;

        const int mask = LINE_SIZE - 1;

        const int segmentsPerBlock = BLOCK_SIZE / LINE_SIZE;
        const int warpId = tid / LINE_SIZE;
        const int segmentId = blockIdx.x * segmentsPerBlock + warpId;

        T* localBuffer = &localBufferStorage[warpId * LINE_SIZE];

        int segmentStart = segmentId < segmentsCount ? segmentStarts[segmentId] : 0;
        int segmentEnd = segmentId < segmentsCount ? segmentEnds[segmentId] : 0;
        int segmentSize = segmentEnd - segmentStart;

        src += segmentStart;
        
        const int localId = tid & mask;
        const auto loopSize = LINE_SIZE * CeilDivide(segmentSize, LINE_SIZE);

        for (int i = localId; i < loopSize; i += LINE_SIZE) {
            localBuffer[localId] += i < segmentSize ? StreamLoad(src + i) : 0;
        }

        const T warpResult = WarpReduce(localId, localBuffer, LINE_SIZE);

        __syncthreads();

        if (localId == 0) {
            localBufferStorage[warpId] = warpResult;
        }
        __syncthreads();


        if (tid < segmentsPerBlock && (blockIdx.x * segmentsPerBlock + tid < segmentsCount)) {
            reducedSegments[blockIdx.x * segmentsPerBlock + tid] = localBufferStorage[tid];
        }
    }


    template <typename T, int BLOCK_SIZE>
    __global__ void SegmentedReduceBlockPerSegmentImpl(const T* src,
                                                       const int* segmentStarts,
                                                       const int* segmentEnds,
                                                       ui32 segmentsCount,
                                                       T* reducedSegments) {
        __shared__  T localBuffer[BLOCK_SIZE];
        const int tid = threadIdx.x;
        localBuffer[tid] = 0;

        const int segmentId = blockIdx.x;
        int segmentStart = segmentStarts[segmentId];
        int segmentEnd = segmentEnds[segmentId];
        int segmentSize = segmentEnd - segmentStart;

        src += segmentStart;

        const auto loopSize = BLOCK_SIZE * CeilDivide(segmentSize, BLOCK_SIZE);

        for (int i = tid; i < loopSize; i += BLOCK_SIZE) {
            localBuffer[tid] += i < segmentSize ? StreamLoad(src + i) : 0;
        }
        __syncthreads();

        T result = FastInBlockReduce(tid, localBuffer, BLOCK_SIZE);

        if (tid == 0) {
            reducedSegments[blockIdx.x] = result;
        }
    }


    template<typename T>
    cudaError_t Reduce(const T* input, T* output, ui32 size,
                       EOperatorType type,
                       TCubKernelContext& context, TCudaStream stream) {
        using TKernelContext = TCubKernelContext;

        switch (type) {
            case EOperatorType::Sum: {
                return cub::DeviceReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                 input, output, size,
                                                 cub::Sum(),
                                                 T(),
                                                 stream);
            }
            case EOperatorType::Max: {
                return cub::DeviceReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                 input, output, size,
                                                 cub::Max(),
                                                 T(),
                                                 stream);
            }
            case EOperatorType::Min: {
                return cub::DeviceReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                 input, output, size,
                                                 cub::Min(),
                                                 T(),
                                                 stream);
            }
            default: {
                return cudaErrorNotYetImplemented;
            }
        }
    }

    template<typename T, typename K>
    cudaError_t ReduceByKey(const T* input, const K* keys, ui32 size,
                            T* output, K* outKeys, ui32* outputSize,
                            EOperatorType type,
                            TCubKernelContext& context,
                            TCudaStream stream) {

        using TKernelContext = TCubKernelContext;

        switch (type) {
            case EOperatorType::Sum: {
                return cub::DeviceReduce::ReduceByKey(context.TempStorage, context.TempStorageSize,
                                                     keys, outKeys,
                                                     input, output,
                                                     outputSize,
                                                     cub::Sum(),
                                                     size,
                                                     stream);
            }
            case EOperatorType::Max: {
                return cub::DeviceReduce::ReduceByKey(context.TempStorage, context.TempStorageSize,
                                                      keys, outKeys,
                                                      input, output,
                                                      outputSize,
                                                      cub::Max(),
                                                      size,
                                                      stream);
            }
            case EOperatorType::Min: {
                return cub::DeviceReduce::ReduceByKey(context.TempStorage, context.TempStorageSize,
                                                      keys, outKeys,
                                                      input, output,
                                                      outputSize,
                                                      cub::Min(),
                                                      size,
                                                      stream);
            }
            default: {
                return cudaErrorNotYetImplemented;
            }
        }
    }


    template<typename T>
    cudaError_t SegmentedReduce(const T* input, ui32 size, const ui32* offsets, ui32 numSegments, T* output,
                                EOperatorType type,
                                TCubKernelContext& context,
                                TCudaStream stream) {
        using TKernelContext = TCubKernelContext;

        //WTF: in cub kernel interface aren't const, but test shows, that they effectively const type
        int* beginOffsets = const_cast<int*>((const int*) offsets);
        int* endOffsets = const_cast<int*>((const int*) (offsets + 1));

        const double meanSize = size * 1.0 /  numSegments;
        if (meanSize < 2048) {
            if (!context.Initialized) {
                return cudaSuccess;
            }
            switch (type) {
                case EOperatorType::Sum: {
                    if (meanSize <= 2) {
                        const ui32 lineSize = 2;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < numBlocks, blockSize, 0, stream >> >
                                                                                                                    (input, beginOffsets, endOffsets, numSegments, output);

                    } else if (meanSize <= 4) {
                        const ui32 lineSize = 4;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < numBlocks, blockSize, 0, stream >> >
                                (input, beginOffsets, endOffsets, numSegments, output);

                    } else if (meanSize <= 8) {
                        const ui32 lineSize = 8;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < numBlocks, blockSize, 0, stream >> >
                                                                                                                (input, beginOffsets, endOffsets, numSegments, output);

                    } else if (meanSize <= 16) {
                        const ui32 lineSize = 16;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < numBlocks, blockSize, 0, stream >> >
                                                                                                                (input, beginOffsets, endOffsets, numSegments, output);
                    } else if (meanSize <= 256) {
                        const ui32 lineSize = 32;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < numBlocks, blockSize, 0, stream >> >(input, beginOffsets, endOffsets, numSegments, output);
                    } else {
                        const ui32 blockSize = 256;
                        const ui32 numBlocks = numSegments;
                        SegmentedReduceBlockPerSegmentImpl<T, blockSize> << < numBlocks, blockSize, 0, stream >> >(input, beginOffsets, endOffsets, numSegments, output);
                    }
                    return cudaSuccess;
                }
                default: {
                    return cudaErrorNotYetImplemented;
                }
            }
        } else {
            switch (type) {

                case EOperatorType::Sum: {
                    return cub::DeviceSegmentedReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                              input, output, numSegments,
                                                              beginOffsets, endOffsets,
                                                              cub::Sum(),
                                                              T(),
                                                              stream);
                }
                case EOperatorType::Max: {
                    return cub::DeviceSegmentedReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                              input, output,
                                                              numSegments,
                                                              beginOffsets, endOffsets,
                                                              cub::Max(),
                                                              T(),
                                                              stream);
                }
                case EOperatorType::Min: {
                    return cub::DeviceSegmentedReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                              input, output,
                                                              numSegments,
                                                              beginOffsets, endOffsets,
                                                              cub::Min(),
                                                              T(),
                                                              stream);
                }
                default: {
                    return cudaErrorNotYetImplemented;
                }
            }
        }
    }


    template  cudaError_t Reduce<float>(const float* input, float* output, ui32 size, EOperatorType type, TCubKernelContext& context, TCudaStream stream);

    template  cudaError_t SegmentedReduce<float>(const float* input, ui32 size, const ui32* offsets, ui32 numSegments, float* output,
                                                 EOperatorType type,
                                                 TCubKernelContext& context,
                                                 TCudaStream stream);

    template  cudaError_t ReduceByKey<float, ui32>(const float* input, const ui32* keys, ui32 size,
                                                   float* output, ui32* outKeys, ui32* outputSize,
                                                   EOperatorType type,
                                                   TCubKernelContext& context,
                                                   TCudaStream stream);

}