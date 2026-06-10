#include <catboost/cuda/methods/kernel/exact_estimation.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/kernel_helpers.cuh>

namespace NKernel {

    template <int BLOCK_SIZE>
    __global__ void ComputeWeightedQuantileWithBinarySearchImpl(const float* targets,
                                                                const float* weightsPrefixSum,
                                                                ui32 objectsCount,
                                                                const float* needWeights,
                                                                const ui32* beginOffsets,
                                                                const ui32* endOffsets,
                                                                float* point,
                                                                float alpha,
                                                                ui32 binarySearchIterations) {
        const ui32 i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        // One thread per leaf. The host passes the per-leaf count (binCount) into
        // this bound on HIP; on NVIDIA it stays the object count (byte-identical).
        // Bounding by the object count leaves high-index leaves uncomputed when
        // binCount > objectCount (a populated leaf reads back 0) and otherwise
        // runs threads past the binCount-sized arrays -- a pre-existing all-platform
        // bug that catboost's TExactLeavesEstimationTest exercises.
        if (i >= objectsCount) {
            return;
        }

        ui32 left = beginOffsets[i];
#if defined(USE_HIP)
        const ui32 end = endOffsets[i];

        // Empty leaf (begin == end) must yield 0 to match the CPU optimum-const.
        // The base `right = end==0?0:end-1` then `left>right` guard caught empty
        // interior bins but not a leading run of empties (begin==end==0 -> right=0,
        // guard false, returns targets[0]). HIP-only; NVIDIA keeps the base guard.
        if (left >= end) {
            point[i] = 0;
            return;
        }
        ui32 right = end - 1;
#else
        ui32 right = endOffsets[i] == 0 ? 0 : endOffsets[i] - 1;

        if (left > right) {
            point[i] = 0;
            return;
        }
#endif

        const float eps = std::numeric_limits<float>::epsilon();
        for (ui32 index = 0; index < binarySearchIterations; ++index) {
            ui32 middle = left + (right - left) / 2;

            if (weightsPrefixSum[middle] < needWeights[i] - eps) {
                left = middle;
            } else {
                right = middle;
            }
        }

        point[i] = targets[right];
        return;
    }

    template <int BLOCK_SIZE>
    __global__ void ComputeNeedWeightsImpl(const float* targets,
                                           const float* weights,
                                           const ui32* beginOffsets,
                                           const ui32* endOffsets,
                                           float* needWeights,
                                           float alpha,
                                           ui32 elementsPerThreads) {
        const ui32 begin = beginOffsets[blockIdx.x] + threadIdx.x;
        const ui32 end = endOffsets[blockIdx.x];

        __shared__ float localBuffer[BLOCK_SIZE];
        localBuffer[threadIdx.x] = 0;

        if (begin >= end) {
            return;
        }

        float totalSum = 0;
        for (ui32 idx = begin; idx < end; idx += BLOCK_SIZE) {
            totalSum += weights[idx];
        }

        localBuffer[threadIdx.x] = totalSum;
        __syncthreads();

        float blocksSum = FastInBlockReduce<float>(threadIdx.x, localBuffer, BLOCK_SIZE);
        if (threadIdx.x == 0) {
            needWeights[blockIdx.x] = blocksSum * alpha;
        }
    }

    __global__ void ComputeWeightsWithTargetsImpl(const float* targets,
                                                  const float* weights,
                                                  float* weightsWithTargets,
                                                  ui32 objectsCount) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i >= objectsCount) {
            return;
        }

        const float delta = max(1.0f, abs(targets[i]));;
        weightsWithTargets[i] = weights[i] / delta;
    }

    __global__ void MakeEndOfBinsFlagsImpl(const ui32* beginOffsets,
                                           const ui32* endOffsets,
                                           ui32* flags) {
        const ui32 begin = beginOffsets[blockIdx.x];
        const ui32 end = endOffsets[blockIdx.x];

        if (begin == end) {
            return;
        }

        flags[begin] = 1;
    }

    void ComputeNeedWeights(const float* targets,
                            const float* weights,
                            ui32 objectsCount,
                            ui32 binCount,
                            const ui32* beginOffsets,
                            const ui32* endOffsets,
                            float* needWeights,
                            float alpha,
                            TCudaStream stream) {
        const ui32 blockSize = 1024;
        const ui32 blocksNum = binCount;
        const ui32 elementsPerThreads = CeilDivide(objectsCount, blockSize * blocksNum);

        ComputeNeedWeightsImpl<blockSize> <<< blocksNum, blockSize, 0, stream >>> (targets,
                                                                                     weights,
                                                                                     beginOffsets,
                                                                                     endOffsets,
                                                                                     needWeights,
                                                                                     alpha,
                                                                                     elementsPerThreads);
    }

    void ComputeWeightsWithTargets(const float* targets,
                                   const float* weights,
                                   float* weightsWithTargets,
                                   ui32 objectsCount,
                                   TCudaStream stream) {
        const ui32 blockSize = 512;
        const ui32 blocksNum = CeilDivide(objectsCount, blockSize);

        ComputeWeightsWithTargetsImpl <<< blocksNum, blockSize, 0, stream >>> (targets,
                                                                                 weights,
                                                                                 weightsWithTargets,
                                                                                 objectsCount);
    }

    void ComputeWeightedQuantileWithBinarySearch(const float* targets,
                                                 const float* weightsPrefixSum,
                                                 ui32 objectsCount,
                                                 const float* needWeights,
                                                 const ui32* beginOffsets,
                                                 const ui32* endOffsets,
                                                 ui32 binCount,
                                                 float* point,
                                                 float alpha,
                                                 ui32 binarySearchIterations,
                                                 TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 blocksNum = CeilDivide(binCount, blockSize);

#if defined(USE_HIP)
        // The kernel is one-thread-per-leaf; bound it by binCount (the grid already
        // covers binCount). NVIDIA passes objectsCount below, byte-identical to base.
        ComputeWeightedQuantileWithBinarySearchImpl<blockSize> <<< blocksNum, blockSize, 0, stream >>> (targets,
                                                                                                          weightsPrefixSum,
                                                                                                          binCount,
                                                                                                          needWeights,
                                                                                                          beginOffsets,
                                                                                                          endOffsets,
                                                                                                          point,
                                                                                                          alpha,
                                                                                                          binarySearchIterations);
#else
        ComputeWeightedQuantileWithBinarySearchImpl<blockSize> <<< blocksNum, blockSize, 0, stream >>> (targets,
                                                                                                          weightsPrefixSum,
                                                                                                          objectsCount,
                                                                                                          needWeights,
                                                                                                          beginOffsets,
                                                                                                          endOffsets,
                                                                                                          point,
                                                                                                          alpha,
                                                                                                          binarySearchIterations);
#endif
    }

    void MakeEndOfBinsFlags(const ui32* beginOffsets,
                            const ui32* endOffsets,
                            ui32 binCount,
                            ui32* flags,
                            TCudaStream stream) {
        const ui32 blockSize = 128;
        const ui32 blocksNum = binCount;

        MakeEndOfBinsFlagsImpl <<< blocksNum, blockSize, 0, stream >>> (beginOffsets,
                                                                          endOffsets,
                                                                          flags);
    }
}
