#include "reduce.cuh"
#include "fill.cuh"
#include "kernel_helpers.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>


namespace NKernel {

    /**
    * \brief Default sum functor
    */
    struct L1Sum
    {
        /// Boolean sum operator, returns <tt>|a| + |b|</tt>
        __host__ __device__ __forceinline__ float operator()(const float &a, const float &b) const
        {
            return fabs(a) + fabs(b);
        }
    };

    //current cub segmented reduce sucks on small segments problems
    //LINE_SIZE should be leq 32
    //TODO(noxoomo): special version for by-thread reduction in case of 1-4 elements per segment
    //TODO(noxoomo): Fallback to block-reduce if one of segments is too big (e.g. loopSize > 256)
    template <typename T, int BLOCK_SIZE, int LINE_SIZE>
    __launch_bounds__(BLOCK_SIZE, CUDA_MAX_THREADS_PER_SM / BLOCK_SIZE)
    __global__ void SegmentedReduceWarpPartPerSegmentImpl(const T* src,
                                                          const int* segmentStarts,
                                                          const int* segmentEnds,
                                                          ui32 segmentsCount,
                                                          T* reducedSegments,
                                                          int blockCount
    ) {
        __shared__  T localBufferStorage[BLOCK_SIZE];
        const int tid = threadIdx.x;
        int blockId = blockIdx.x;

        while (blockId < blockCount) {
            __syncthreads();

            localBufferStorage[tid] = 0;

            const int mask = LINE_SIZE - 1;

            const int segmentsPerBlock = BLOCK_SIZE / LINE_SIZE;
            const int warpId = tid / LINE_SIZE;
            const int segmentId = blockId * segmentsPerBlock + warpId;

            T* localBuffer = &localBufferStorage[warpId * LINE_SIZE];

            int segmentStart = segmentId < segmentsCount ? segmentStarts[segmentId] : 0;
            int segmentEnd = segmentId < segmentsCount ? segmentEnds[segmentId] : 0;
            int segmentSize = segmentEnd - segmentStart;

            src += segmentStart;

            const int localId = tid & mask;
            const auto loopSize = LINE_SIZE * CeilDivide(segmentSize, LINE_SIZE);

            {
                float tmp = 0;
                for (int i = localId; i < loopSize; i += LINE_SIZE) {
                    tmp += i < segmentSize ? StreamLoad(src + i) : 0;
                }
                localBuffer[localId] = tmp;
            }

            const T warpResult = WarpReduce(localId, localBuffer, LINE_SIZE);

            __syncthreads();

            if (localId == 0) {
                localBufferStorage[warpId] = warpResult;
            }
            __syncthreads();


            if (tid < segmentsPerBlock && (blockId * segmentsPerBlock + tid < segmentsCount)) {
                reducedSegments[blockId * segmentsPerBlock + tid] = localBufferStorage[tid];
            }
            blockId += gridDim.x;
        }
    }


    template <typename T, int BLOCK_SIZE>
    __global__ void SegmentedReduceBlockPerSegmentImpl(const T* src,
                                                       const int* segmentStarts,
                                                       const int* segmentEnds,
                                                       ui32 segmentsCount,
                                                       T* reducedSegments,
                                                       int numBlocks
    ) {
        __shared__  T localBuffer[BLOCK_SIZE];
        int blockId = blockIdx.x;
        while (blockId < numBlocks) {
            __syncthreads();

            const int tid = threadIdx.x;
            localBuffer[tid] = 0;

            const int segmentId = blockId;
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
                reducedSegments[blockId] = result;
            }
            blockId += gridDim.x;
        }
    }


    template <typename T>
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
                                                 -std::numeric_limits<T>::infinity(),
                                                 stream);
            }
            case EOperatorType::Min: {
                return cub::DeviceReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                 input, output, size,
                                                 cub::Min(),
                                                 std::numeric_limits<T>::infinity(),
                                                 stream);
            }
            case EOperatorType::L1Sum: {
                return cub::DeviceReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                 input, output, size,
                                                 L1Sum(),
                                                 T(),
                                                 stream);
            }
            default: {
                return cudaErrorNotYetImplemented;
            }
        }
    }

    template <typename T, typename K>
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
            case EOperatorType::L1Sum: {
                return cub::DeviceReduce::ReduceByKey(context.TempStorage, context.TempStorageSize,
                                                      keys, outKeys,
                                                      input, output,
                                                      outputSize,
                                                      L1Sum(),
                                                      size,
                                                      stream);
            }
            default: {
                return cudaErrorNotYetImplemented;
            }
        }
    }


    template <typename T>
    cudaError_t SegmentedReduce(const T* input, ui32 size, const ui32* offsets, ui32 numSegments, T* output,
                                EOperatorType type,
                                TCubKernelContext& context,
                                TCudaStream stream) {
        using TKernelContext = TCubKernelContext;

        //WTF: in cub kernel interface aren't const, but test shows, that they effectively const type
        int* beginOffsets = const_cast<int*>((const int*) offsets);
        int* endOffsets = const_cast<int*>((const int*) (offsets + 1));

        if (size == 0) {
            FillBuffer(output, (T)(0), numSegments,  stream);
            return cudaSuccess;
        }

        const double meanSize = size * 1.0 /  numSegments;
        if (meanSize < 600) {
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

                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < min(numBlocks, (ui32)TArchProps::MaxBlockCount()), blockSize, 0, stream >> >
                                (input, beginOffsets, endOffsets, numSegments, output, numBlocks);

                    } else if (meanSize <= 4) {
                        const ui32 lineSize = 4;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < min(numBlocks, (ui32)TArchProps::MaxBlockCount()), blockSize, 0, stream >> >
                                (input, beginOffsets, endOffsets, numSegments, output, numBlocks);

                    } else if (meanSize <= 8) {
                        const ui32 lineSize = 8;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < min(numBlocks, (ui32)TArchProps::MaxBlockCount()), blockSize, 0, stream >> >
                                (input, beginOffsets, endOffsets, numSegments, output, numBlocks);

                    } else if (meanSize <= 16) {
                        const ui32 lineSize = 16;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < min(numBlocks, (ui32)TArchProps::MaxBlockCount()), blockSize, 0, stream >> >
                                (input, beginOffsets, endOffsets, numSegments, output, numBlocks);
                    } else if (meanSize <= 256) {
                        const ui32 lineSize = 32;
                        const ui32 blockSize = 256;
                        const ui32 segmentsPerBlock = blockSize / lineSize;
                        const ui32 numBlocks = CeilDivide(numSegments, segmentsPerBlock);
                        SegmentedReduceWarpPartPerSegmentImpl<T, blockSize, lineSize> << < min(numBlocks, (ui32)TArchProps::MaxBlockCount()), blockSize, 0, stream >> >(input, beginOffsets, endOffsets, numSegments, output, numBlocks);
                    } else {
                        const ui32 blockSize = 512;
                        const ui32 numBlocks = numSegments;
                        SegmentedReduceBlockPerSegmentImpl<T, blockSize> << < min(numBlocks, (ui32)TArchProps::MaxBlockCount()), blockSize, 0, stream >> >(input, beginOffsets, endOffsets, numSegments, output, numBlocks);
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
                case EOperatorType::L1Sum: {
                    return cub::DeviceSegmentedReduce::Reduce(context.TempStorage, context.TempStorageSize,
                                                              input, output,
                                                              numSegments,
                                                              beginOffsets, endOffsets,
                                                              L1Sum(),
                                                              T(),
                                                              stream);
                }
                default: {
                    return cudaErrorNotYetImplemented;
                }
            }
        }
    }

    #define REDUCE(Type) \
    template  cudaError_t Reduce<Type>(const Type* input, Type* output, ui32 size, EOperatorType type, TCubKernelContext& context, TCudaStream stream);

    REDUCE(float)
    REDUCE(ui32)
    REDUCE(int)
    REDUCE(ui64)


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
