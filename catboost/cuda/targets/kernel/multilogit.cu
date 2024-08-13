#include "multilogit.cuh"
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

namespace NKernel {


    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiLogitValAndFirstDerImpl(const float* targetClasses, int numClasses, ui32 size,
                                                 const float* weights,
                                                 const float* predictions,
                                                 const ui32* loadPredictionsIndices,
                                                 ui64 predictionsAlignSize,
                                                 float* functionValue,
                                                 float* der,
                                                 ui64 derAlignSize) {

        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;
        const int effectiveClassCount = numClasses - 1;

        float tmpScore = 0;

        float classApprox[ElementsPerThread];
        ui16 targetClass[ElementsPerThread];
        float sumExpApproxForAllClasses[ElementsPerThread];

        float weight[ElementsPerThread];
        float maxApprox[ElementsPerThread];

        ui32 loadApproxIndex[ElementsPerThread];
        {

            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const int idx = tid + j * BlockSize;
                loadApproxIndex[j] = loadPredictionsIndices && idx < size ? __ldg(loadPredictionsIndices + idx) : idx;
                targetClass[j] = idx < size ? static_cast<ui16>(__ldg(targetClasses + idx)) : 0;


                maxApprox[j] = 0;
                for (int k = 0; k < effectiveClassCount; ++k) {
                    maxApprox[j] = idx < size ? max(maxApprox[j], __ldg(predictions + loadApproxIndex[j] + k * predictionsAlignSize)) : 0;
                }

                const float tmp =  targetClass[j] < effectiveClassCount  && idx < size ? __ldg(predictions + loadApproxIndex[j] + targetClass[j] * predictionsAlignSize)  : 0.0f;
                classApprox[j] = tmp - maxApprox[j];

                sumExpApproxForAllClasses[j] = 0.0f;
                for (int k = 0; k < effectiveClassCount; ++k) {
                    sumExpApproxForAllClasses[j] += idx < size ? __expf(__ldg(predictions + loadApproxIndex[j] + k * predictionsAlignSize) - maxApprox[j]) : 0.0f;
                }

                sumExpApproxForAllClasses[j] += __expf(0.0f - maxApprox[j]);
            }
        }


        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const int idx = tid + j * BlockSize;
            weight[j] = (weights && (idx < size)) ? weights[idx] : 1.0f;
        }



        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const int idx = tid + j * BlockSize;

            if (der && idx < size) {
                for (int k = 0; k < effectiveClassCount; ++k) {
                    const float pk = __expf(__ldg(predictions + loadApproxIndex[j] + k * predictionsAlignSize) - maxApprox[j]) / sumExpApproxForAllClasses[j];

                    der[idx + k * derAlignSize] = weight[j] * ((targetClass[j] == k ? 1.0f : 0.0f) - pk);
                }
            }


            if (functionValue) {
                const float logDenum = __logf(sumExpApproxForAllClasses[j]);
                tmpScore += (idx < size) ? weight[j] * (classApprox[j] - logDenum) : 0;
            }
        }


        if (functionValue) {
            __shared__ float tmpScores[BlockSize];
            tmpScores[threadIdx.x] = tmpScore;
            __syncthreads();

            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BlockSize);

            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }



    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiLogitSecondDerRowImpl(const float* targetClasses, int numClasses, ui32 size,
                                               const float* weights,
                                               const float* predictions,
                                               ui64 predictionsAlignSize,
                                               int der2Row,
                                               ui64 der2AlignSize,
                                               float* der2) {

        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;
        const int effectiveClassCount = numClasses - 1;

        float sumExpApproxForAllClasses[ElementsPerThread];

        float weight[ElementsPerThread];
        float maxApprox[ElementsPerThread];

        {

            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const int idx = tid + j * BlockSize;

                maxApprox[j] = 0;
                for (int k = 0; k < effectiveClassCount; ++k) {
                    maxApprox[j] = idx < size ? max(maxApprox[j], __ldg(predictions + idx + k * predictionsAlignSize)) : 0;
                }


                sumExpApproxForAllClasses[j] = 0.0f;
                for (int k = 0; k < effectiveClassCount; ++k) {
                    sumExpApproxForAllClasses[j] += idx < size ? __expf(__ldg(predictions + idx + k * predictionsAlignSize) - maxApprox[j]) : 0;
                }

                sumExpApproxForAllClasses[j] += __expf(0.0f - maxApprox[j]);
            }
        }


        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const int idx = tid + j * BlockSize;
            weight[j] = (weights && (idx < size)) ? weights[idx] : 1.0f;
        }


        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const int idx = tid + j * BlockSize;
            if (idx < size) {
                float pRow = 0;
                if (der2Row < effectiveClassCount) {
                    pRow = __expf(__ldg(predictions + idx + der2Row * predictionsAlignSize) - maxApprox[j]) / sumExpApproxForAllClasses[j];
                } else {
                    pRow = __expf(-maxApprox[j]) / sumExpApproxForAllClasses[j];
                }

                for (int k = 0; k < der2Row; ++k) {
                    const float pk = __expf(__ldg(predictions + idx + k * predictionsAlignSize) - maxApprox[j]) / sumExpApproxForAllClasses[j];

                    der2[idx + k * der2AlignSize] = -weight[j] * pk * pRow;
                }
                der2[idx + der2Row * der2AlignSize] = weight[j] * (1.0 - pRow) * pRow;
            }
        }
    }


    void MultiLogitValueAndDer(const float* targetClasses, int numClasses,
                               const float* targetWeights,
                               ui32 size,
                               const float* predictions, ui32 predictionsAlignSize,
                               const ui32* loadPredictionsIndices,
                               float* functionValue,
                               float* der, ui32 derAlignSize,
                               TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 1;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);

        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }

        if (numBlocks) {
            MultiLogitValAndFirstDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, numClasses, size, targetWeights, predictions, loadPredictionsIndices, predictionsAlignSize,  functionValue, der, derAlignSize);
        }
    }


    void MultiLogitSecondDer(const float* targetClasses, int numClasses,
                             const float* targetWeights,
                             ui32 size,
                             const float* predictions, ui32 predictionsAlignSize,
                             float* der2,
                             int der2Row, ui32 der2AlignSize,
                             TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 1;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);


        if (numBlocks) {
            MultiLogitSecondDerRowImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, numClasses, size, targetWeights, predictions, predictionsAlignSize, der2Row, der2AlignSize, der2);
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void RMSEWithUncertaintyValAndFirstDerImpl(
        const float* target, ui32 size,
        const float* weights,
        const float* predictions,
        const ui32* loadPredictionsIndices,
        ui64 predictionsAlignSize,
        float* functionValue,
        float* der,
        ui64 derAlignSize
    ) {
        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;

        float tmpScore = 0;

        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const int idx = tid + j * BlockSize;
            if (idx >= size) {
                continue;
            }
            const ui32 loadApproxIndex = loadPredictionsIndices ? __ldg(loadPredictionsIndices + idx) : idx;
            const float weight = weights ? __ldg(weights + idx) : 1.0f;

            const float approx0 = __ldg(predictions + loadApproxIndex);
            const float approx1 = __ldg(predictions + loadApproxIndex + predictionsAlignSize);
            const float direction = __ldg(target + idx) - approx0;
            const float expApprox1 = __expf(min(-2 * approx1, 70.0f));

            if (der) { // -gradient
                der[idx] = weight * direction;
                der[idx + derAlignSize] = weight * (direction * direction * expApprox1 - 1);
            }

            if (functionValue) {
                // np.log(2 * np.pi) / 2.0 = 0.9189385332046
                tmpScore += -weight * (0.9189385332046 + approx1 + 0.5 * expApprox1 * direction * direction);
            }
        }

        if (functionValue) {
            __shared__ float tmpScores[BlockSize];
            tmpScores[threadIdx.x] = tmpScore;
            __syncthreads();

            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BlockSize);

            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void RMSEWithUncertaintySecondDerRowImpl(
        const float* target, ui32 size,
        const float* weights,
        const float* predictions,
        ui64 predictionsAlignSize,
        int der2Row,
        ui64 der2AlignSize,
        float* der2
    ) {
        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;
        if (der2Row == 0) {
            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const ui32 idx = tid + j * BlockSize;
                if (idx < size) {
                    der2[idx] = weights ? __ldg(weights + idx) : 1.0f;
                }
            }
        } else if (der2Row == 1) {
            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const ui32 idx = tid + j * BlockSize;
                if (idx < size) {
                    const float approx0 = __ldg(predictions + idx);
                    const float approx1 = __ldg(predictions + idx + predictionsAlignSize);
                    const float weight = weights ? __ldg(weights + idx) : 1.0f;
                    const float miss = __ldg(target + idx) - approx0;
                    const float expApprox1 = __expf(min(-2 * approx1, 70.0f));
                    der2[idx] = 0.0f;
                    der2[idx + der2AlignSize] = 2 * weight * miss * miss * expApprox1;
                }
            }
        } else {
            // unreachable
            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const int idx = tid + j * BlockSize;
                if (idx < size) {
                    for (int k = 0; k < der2Row; ++k) {
                        der2[idx + k * der2AlignSize] = 0.0;
                    }
                }
            }
        }
    }


    void RMSEWithUncertaintyValueAndDer(
        const float* target,
        const float* weights,
        ui32 size,
        const float* predictions, ui32 predictionsAlignSize,
        const ui32* loadPredictionsIndices,
        float* functionValue,
        float* der, ui32 derAlignSize,
        TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }
        if (numBlocks) {
            RMSEWithUncertaintyValAndFirstDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(target, size, weights, predictions, loadPredictionsIndices, predictionsAlignSize,  functionValue, der, derAlignSize);
        }
    }

    void RMSEWithUncertaintySecondDer(
        const float* target,
        const float* weights,
        ui32 size,
        const float* predictions, ui32 predictionsAlignSize,
        float* der2,
        int der2Row, ui32 der2AlignSize,
        TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        if (numBlocks) {
            RMSEWithUncertaintySecondDerRowImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(target, size, weights, predictions, predictionsAlignSize, der2Row, der2AlignSize, der2);
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiCrossEntropyValueAndDerImpl(
        ui32 targetCount,
        ui32 size,
        const float* targets, ui32 targetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        const ui32* loadPredictionsIndices,
        float* functionValue,
        float* der, ui32 derAlignSize
    ) {
        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;

        float sumDimErrors = 0;
        for (int dim = 0; dim < targetCount; ++dim) {
            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const int idx = tid + j * BlockSize;
                if (idx >= size) {
                    continue;
                }
                const ui32 loadApproxIndex = loadPredictionsIndices ? __ldg(loadPredictionsIndices + idx) : idx;
                const float weight = weights ? __ldg(weights + idx) : 1.0f;
                const float target = __ldg(targets + idx + dim * targetAlignSize);
                const float approx = __ldg(predictions + loadApproxIndex + dim * predictionsAlignSize);
                const float expApprox = __expf(approx);
                if (functionValue) {
                    sumDimErrors += -(isfinite(expApprox) ? __logf(1.0f + expApprox) : approx) * weight;
                    sumDimErrors += (target * approx) * weight;
                }
                if (der) { // -gradient
                    const float sigmoid = isfinite(expApprox) ? expApprox / (1.0f + expApprox) : 1.0f;
                    der[idx + dim * derAlignSize] = (-sigmoid + target) * weight;
                }

            }
        }
        if (functionValue) {
            __shared__ float tmpScores[BlockSize];
            tmpScores[threadIdx.x] = sumDimErrors / targetCount;
            __syncthreads();

            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BlockSize);

            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiCrossEntropySecondDerImpl(
        ui32 targetCount,
        ui32 size,
        const float* targets, ui32 targetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        float* der2,
        int der2Row, ui32 der2AlignSize
    ) {
        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;
        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const ui32 idx = tid + j * BlockSize;
            if (idx < size) {
                for (int k = 0; k < der2Row; ++k) {
                    der2[idx + k * der2AlignSize] = 0.0f;
                }
                const float approx = __ldg(predictions + idx + der2Row * predictionsAlignSize);
                const float expApprox = __expf(approx);
                const float weight = weights ? __ldg(weights + idx) : 1.0f;
                const float target = __ldg(targets + idx + der2Row * targetAlignSize);
                const float negSigmoid = isfinite(expApprox) ? -expApprox / (1.0f + expApprox) : -1.0f;
                der2[idx + der2Row * der2AlignSize] = -negSigmoid * (1.0f + negSigmoid) * weight;
            }
        }
    }


    void MultiCrossEntropyValueAndDer(
        ui32 targetCount,
        ui32 size,
        const float* target, ui32 tragetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        const ui32* loadPredictionsIndices,
        float* functionValue,
        float* der, ui32 derAlignSize,
        TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }
        if (numBlocks) {
            MultiCrossEntropyValueAndDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(
                targetCount,
                size,
                target, tragetAlignSize,
                weights,
                predictions, predictionsAlignSize,
                loadPredictionsIndices,
                functionValue,
                der, derAlignSize);
        }
    }

    void MultiCrossEntropySecondDer(
        ui32 targetCount,
        ui32 size,
        const float* target, ui32 tragetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        float* der2,
        ui32 der2Row, ui32 der2AlignSize,
        TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        if (numBlocks) {
            MultiCrossEntropySecondDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(
                targetCount,
                size,
                target, tragetAlignSize,
                weights,
                predictions, predictionsAlignSize,
                der2,
                der2Row, der2AlignSize);
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiRMSEValueAndDerImpl(
        ui32 targetCount,
        ui32 size,
        const float* targets, ui32 targetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        const ui32* loadPredictionsIndices,
        float* functionValue,
        float* der, ui32 derAlignSize
    ) {
        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;

        float sumDimErrors = 0;
        for (int dim = 0; dim < targetCount; ++dim) {
            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const int idx = tid + j * BlockSize;
                if (idx >= size) {
                    continue;
                }
                const ui32 loadApproxIndex = loadPredictionsIndices ? __ldg(loadPredictionsIndices + idx) : idx;
                const float weight = weights ? __ldg(weights + idx) : 1.0f;
                const float target = __ldg(targets + idx + dim * targetAlignSize);
                const float approx = __ldg(predictions + loadApproxIndex + dim * predictionsAlignSize);
                const float diff = target - approx;
                if (functionValue) {
                    sumDimErrors += diff * diff * weight;
                }
                if (der) { // -gradient
                    der[idx + dim * derAlignSize] = diff * weight;
                }

            }
        }
        if (functionValue) {
            __shared__ float tmpScores[BlockSize];
            tmpScores[threadIdx.x] = -sumDimErrors;
            __syncthreads();

            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BlockSize);

            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiRMSESecondDerImpl(
        ui32 size,
        const float* weights,
        float* der2,
        int der2Row, ui32 der2AlignSize
    ) {
        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;
        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const ui32 idx = tid + j * BlockSize;
            if (idx < size) {
                for (int k = 0; k < der2Row; ++k) {
                    der2[idx + k * der2AlignSize] = 0.0f;
                }
                const float weight = weights ? __ldg(weights + idx) : 1.0f;
                der2[idx + der2Row * der2AlignSize] = weight;
            }
        }
    }


    void MultiRMSEValueAndDer(
        ui32 targetCount,
        ui32 size,
        const float* target, ui32 tragetAlignSize,
        const float* weights,
        const float* predictions, ui32 predictionsAlignSize,
        const ui32* loadPredictionsIndices,
        float* functionValue,
        float* der, ui32 derAlignSize,
        TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }
        if (numBlocks) {
            MultiRMSEValueAndDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(
                targetCount,
                size,
                target, tragetAlignSize,
                weights,
                predictions, predictionsAlignSize,
                loadPredictionsIndices,
                functionValue,
                der, derAlignSize);
        }
    }

    void MultiRMSESecondDer(
        ui32 size,
        const float* weights,
        float* der2,
        ui32 der2Row, ui32 der2AlignSize,
        TCudaStream stream
    ) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        if (numBlocks) {
            MultiRMSESecondDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(
                size,
                weights,
                der2,
                der2Row, der2AlignSize);
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiClassOneVsAllValAndFirstDerImpl(const float* targetClasses, int numClasses, ui32 size,
                                                         const float* weights,
                                                         const float* predictions,
                                                         const ui32* loadPredictionsIndices,
                                                         ui64 predictionsAlignSize,
                                                         float* functionValue,
                                                         float* der,
                                                         ui64 derAlignSize) {
        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;

        float tmpScore = 0;

        ui16 targetClass[ElementsPerThread];
        float weight[ElementsPerThread];
        ui32 loadPredictionIndex[ElementsPerThread];

        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const int idx = tid + j * BlockSize;
            loadPredictionIndex[j] = loadPredictionsIndices && idx < size ? __ldg(loadPredictionsIndices + idx) : idx;
            targetClass[j] = idx < size ? static_cast<ui16>(__ldg(targetClasses + idx)) : 0;
            weight[j] = (weights && (idx < size)) ? weights[idx] : 1.0f;
        }


        for (int clazz = 0; clazz < numClasses; ++clazz) {
            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
               const int idx = tid + j * BlockSize;
               const float val = idx < size ? __ldg(predictions + loadPredictionIndex[j] + clazz * predictionsAlignSize) : 0.0f;
               const float expVal = __expf(val);
               const float p = ClipProb(expVal / (1.0f + expVal));
               const float c = clazz == targetClass[j] ? 1.0f : 0.0f;
               const float direction = c - p;

                if (der && idx < size) {
                    der[idx + clazz * derAlignSize] = weight[j] * direction;
               }

               if (functionValue) {
                   const float logExpValPlusOne = isfinite(expVal) ? __logf(1 + expVal) : val;
                   tmpScore += (idx < size) ? weight[j] * (c * val - logExpValPlusOne) / numClasses : 0;
               }
            }
        }


        if (functionValue) {
            __shared__ float tmpScores[BlockSize];
            tmpScores[threadIdx.x] = tmpScore;
            __syncthreads();

            float val = FastInBlockReduce<float>(threadIdx.x, tmpScores, BlockSize);

            if (threadIdx.x == 0) {
                atomicAdd(functionValue, val);
            }
        }
    }

    template <int BlockSize, int ElementsPerThread>
    __launch_bounds__(BlockSize, CUDA_MAX_THREADS_PER_SM / BlockSize)
    __global__ void MultiClassOneVsAllSecondDerImpl(const float* targetClasses, int numClasses, ui32 size,
                                                    const float* weights,
                                                    const float* predictions,
                                                    ui64 predictionsAlignSize,
                                                    ui64 der2AlignSize,
                                                    float* der2) {

        ui32 tid = blockIdx.x * BlockSize * ElementsPerThread + threadIdx.x;

        float weight[ElementsPerThread];

        #pragma unroll
        for (int j = 0; j < ElementsPerThread; ++j) {
            const int idx = tid + j * BlockSize;
            weight[j] = (weights && (idx < size)) ? weights[idx] : 1.0f;
        }


        for (int clazz = 0; clazz < numClasses; ++clazz) {
            #pragma unroll
            for (int j = 0; j < ElementsPerThread; ++j) {
                const int idx = tid + j * BlockSize;
                const float val = idx < size ? __ldg(predictions + idx + clazz * predictionsAlignSize) : 0.0f;
                const float expVal = __expf(val);
                const float p = ClipProb(expVal / (1.0f + expVal));
                if (der2 && idx < size) {
                    der2[idx + clazz * der2AlignSize] = weight[j] * p * (1.0f - p);
                }
            }
        }
    }



    void MultiClassOneVsAllValueAndDer(const float* targetClasses, int numClasses,
                               const float* targetWeights,
                               ui32 size,
                               const float* predictions, ui32 predictionsAlignSize,
                               const ui32* loadPredictionsIndices,
                               float* functionValue,
                               float* der, ui32 derAlignSize,
                               TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 1;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        CB_ENSURE(numClasses <= 65536);

        //TODO: get rid of this
        if (functionValue) {
            FillBuffer(functionValue, 0.0f, 1, stream);
        }

        if (numBlocks) {
            MultiClassOneVsAllValAndFirstDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, numClasses, size, targetWeights, predictions, loadPredictionsIndices, predictionsAlignSize,  functionValue, der, derAlignSize);
        }
    }


    void MultiClassOneVsAllSecondDer(const float* targetClasses, int numClasses,
                                     const float* targetWeights,
                                     ui32 size,
                                     const float* predictions, ui32 predictionsAlignSize,
                                     float* der2,
                                     ui32 der2AlignSize,
                                     TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 2;
        const ui32 numBlocks = CeilDivide<ui32>(size, elementsPerThreads * blockSize);
        CB_ENSURE(numClasses <= 65536);


        if (numBlocks) {
            MultiClassOneVsAllSecondDerImpl < blockSize, elementsPerThreads ><<<numBlocks, blockSize, 0, stream>>>(targetClasses, numClasses, size, targetWeights, predictions, predictionsAlignSize, der2AlignSize, der2);
        }
    }


    __global__ void BuildConfusionMatrixBinsImpl(const float* targetClasses, int numClasses, ui32 size,
                                                 const float* predictions, ui32 predictionsDim,
                                                 ui64 predictionsAlignSize,
                                                 bool isBinClass,
                                                 float binTargetProbabilityThreshold,
                                                 ui32* bins) {

        ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            ui32 targetClass;
            float bestApprox = NegativeInfty();
            int bestClass = -1;

            predictions += i;

            if (isBinClass) {
                targetClass = __ldg(targetClasses + i) > binTargetProbabilityThreshold;
                bestClass = __ldg(predictions) > 0;
            } else {
                targetClass = static_cast<ui16>(__ldg(targetClasses + i));
                for (int clazz = 0; clazz < numClasses; ++clazz) {
                    const float approx = clazz < predictionsDim ? __ldg(predictions + clazz * predictionsAlignSize) : 0.0f;
                    if (approx > bestApprox) {
                        bestApprox = approx;
                        bestClass = clazz;
                    }
                }
            }
            bins[i] = bestClass * numClasses + targetClass;
        }
    }

    void BuildConfusionMatrixBins(const float* targetClasses, int numClasses, ui32 size,
                                  const float* predictions, int predictionsDim, ui32 predictionsAlignSize,
                                  bool isBinClass,
                                  float binTargetProbabilityThreshold,
                                  ui32* bins,
                                  TCudaStream stream) {
        const int blockSize = 256;
        const int numBlocks = (size + blockSize - 1) / blockSize;
        CB_ENSURE(numClasses < 65536);
        if (numBlocks) {
            BuildConfusionMatrixBinsImpl << < numBlocks, blockSize, 0, stream >> >(targetClasses, numClasses, size, predictions, predictionsDim, predictionsAlignSize, isBinClass, binTargetProbabilityThreshold, bins);
        }
    }
}
