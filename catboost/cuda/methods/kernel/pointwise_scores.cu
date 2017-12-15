#include "pointwise_scores.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

#include <cmath>
#include <exception>
#include <cfloat>


namespace NKernel {

    template <int BLOCK_SIZE>
    __global__ void FindOptimalSplitSolarImpl(const TCBinFeature* bf,
                                              int binFeatureCount,
                                              const float* binSums,
                                              const TPartitionStatistics* parts,
                                              int pCount, int foldCount,
                                              TBestSplitProperties* result)
    {
        float bestScore = FLT_MAX;
        int bestIndex = 0;
        int tid = threadIdx.x;
        result += blockIdx.x;

        TPartOffsetsHelper helper(foldCount);

        for (int i = blockIdx.x * BLOCK_SIZE; i < binFeatureCount; i += BLOCK_SIZE * gridDim.x) {
            if (i + tid >= binFeatureCount) {
                break;
            }

            const float* current = binSums + 2 * (i + tid);

            float score = 0;

            for (int leaf = 0; leaf < pCount; leaf++) {

                float leftTotalWeight = 0;
                float rightTotalWeight = 0;

                float leftScore = 0;
                float rightScore = 0;

                #pragma unroll 4
                for (int fold = 0; fold < foldCount; fold += 2) {

                    TPartitionStatistics partLearn = LdgWithFallback(parts, helper.GetDataPartitionOffset(leaf, fold));
                    TPartitionStatistics partTest = LdgWithFallback(parts, helper.GetDataPartitionOffset(leaf, fold + 1));


                    float weightEstimateLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold) * 2];
                    float weightEstimateRight = partLearn.Weight - weightEstimateLeft;

                    float sumEstimateLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold) * 2 + 1];
                    float sumEstimateRight = partLearn.Sum - sumEstimateLeft;


                    float weightTestLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold + 1) * 2];
                    float weightTestRight = partTest.Weight - weightTestLeft;

                    float sumTestLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold + 1) * 2 + 1];
                    float sumTestRight = partTest.Sum - sumTestLeft;


                    {
                        const float mu = weightEstimateLeft > 0.0f ? (sumEstimateLeft / (weightEstimateLeft + 1e-15f)) : 0;
                        leftScore += -2 * mu * sumTestLeft + weightTestLeft * mu * mu;
                        leftTotalWeight += weightTestLeft;
                    }

                    {
                        const float mu =  weightEstimateRight > 0.0f ? (sumEstimateRight / (weightEstimateRight + 1e-15f)) : 0;
                        rightTotalWeight += weightTestRight;
                        rightScore += -2 * mu * sumTestRight + weightTestRight * mu * mu;
                    }
                }

                score += leftTotalWeight > 2 ? leftScore * (1 + 2 * log(leftTotalWeight + 1)) : 0;
                score += rightTotalWeight > 2 ? rightScore * (1 + 2 * log(rightTotalWeight + 1)) : 0;
            }

            if (score < bestScore) {
                bestScore = score;
                bestIndex = i + tid;
            }
        }

        __shared__ float scores[BLOCK_SIZE];
        scores[tid] = bestScore;
        __shared__ int indices[BLOCK_SIZE];
        indices[tid] = bestIndex;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
            if ( scores[tid] > scores[tid + s] ||
                (scores[tid] == scores[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            result->FeatureId = bf[indices[0]].FeatureId;
            result->BinId = bf[indices[0]].BinId;
            result->Score = scores[0];
        }
    }


    __forceinline__ __device__ double SolarScore(double sum, double weight) {
        return  weight > 0 ? (-sum * sum) *  (1 + 2 * log(weight + 1.0)) / (weight + 1e-9f)  : 0;
    }


    template <int BLOCK_SIZE>
    __global__ void FindOptimalSplitSolarSingleFoldImpl(const TCBinFeature* bf,
                                                        int binFeatureCount,
                                                        const float* binSums,
                                                        const TPartitionStatistics* parts,
                                                        int pCount,
                                                        TBestSplitProperties* result)
    {
        float bestScore = FLT_MAX;
        int bestIndex = 0;
        int tid = threadIdx.x;
        result += blockIdx.x;

        TPartOffsetsHelper helper(1);

        for (int i = blockIdx.x * BLOCK_SIZE; i < binFeatureCount; i += BLOCK_SIZE * gridDim.x) {
            if (i + tid >= binFeatureCount) {
                break;
            }

            const float* current = binSums + 2 * (i + tid);

            float score = 0;

            for (int leaf = 0; leaf < pCount; leaf++) {
                TPartitionStatistics part = LdgWithFallback(parts, helper.GetDataPartitionOffset(leaf, 0));

                float weightLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, 0) * 2];
                float weightRight = part.Weight < weightLeft ? 0 : part.Weight - weightLeft;

                float sumLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, 0) * 2 + 1];
                float sumRight = part.Sum - sumLeft;

                score += SolarScore(sumLeft, weightLeft) + SolarScore(sumRight, weightRight);
            }

            if (score < bestScore) {
                bestScore = score;
                bestIndex = i + tid;
            }
        }

        __shared__ float scores[BLOCK_SIZE];
        scores[tid] = bestScore;
        __shared__ int indices[BLOCK_SIZE];
        indices[tid] = bestIndex;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if ( scores[tid] > scores[tid + s] ||
                     (scores[tid] == scores[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            result->FeatureId = bf[indices[0]].FeatureId;
            result->BinId = bf[indices[0]].BinId;
            result->Score = scores[0];
        }
    }


    template <int BLOCK_SIZE>
    __global__ void FindOptimalSplitCorrelationSingleFoldImpl(const TCBinFeature* bf,
                                                              int binFeatureCount,
                                                              const float* binSums,
                                                              const TPartitionStatistics* parts,
                                                              int pCount,
                                                              double l2, bool normalize,
                                                              double scoreStdDev, ui64 globalSeed,
                                                              TBestSplitProperties* result)
    {
        float bestScore = FLT_MAX;
        int bestIndex = 0;
        int tid = threadIdx.x;
        result += blockIdx.x;

        TPartOffsetsHelper helper(1);

        for (int i = blockIdx.x * BLOCK_SIZE; i < binFeatureCount; i += BLOCK_SIZE * gridDim.x) {
            if (i + tid >= binFeatureCount) {
                break;
            }

            const float* current = binSums + 2 * (i + tid);

            float score = 0;
            float denumSqr = 0;

            for (int leaf = 0; leaf < pCount; leaf++) {
                TPartitionStatistics part = LdgWithFallback(parts, helper.GetDataPartitionOffset(leaf, 0));

                float weightLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, 0) * 2];
                float weightRight = part.Weight < weightLeft ? 0 : part.Weight - weightLeft;

                float sumLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, 0) * 2 + 1];
                float sumRight = part.Sum - sumLeft;

                {
                    double lambda = normalize ? l2 * weightLeft : l2;

                    const float mu =  weightLeft > 0 ? (sumLeft / (weightLeft + lambda)) : 0;
                    score +=  sumLeft * mu;
                    denumSqr += weightLeft * mu * mu;
                }

                {
                    double lambda = normalize ? l2 * weightRight : l2;

                    const float mu =  weightRight > 0 ? (sumRight / (weightRight + lambda)) : 0;
                    score += sumRight * mu;
                    denumSqr += weightRight * mu * mu;
                }
            }

            score = denumSqr > 0 ? -score / sqrt(denumSqr) : FLT_MAX;
            if (scoreStdDev) {
                ui64 seed = globalSeed + bf[i + tid].FeatureId;
                AdvanceSeed(&seed, 4);
                score += NextNormal(&seed) * scoreStdDev;
            }
            if (score < bestScore) {
                bestScore = score;
                bestIndex = i + tid;
            }
        }

        __shared__ float scores[BLOCK_SIZE];
        scores[tid] = bestScore;
        __shared__ int indices[BLOCK_SIZE];
        indices[tid] = bestIndex;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if ( scores[tid] > scores[tid + s] ||
                     (scores[tid] == scores[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            result->FeatureId = bf[indices[0]].FeatureId;
            result->BinId = bf[indices[0]].BinId;
            result->Score = scores[0];
        }
    }



    template <int BLOCK_SIZE>
    __global__ void FindOptimalSplitCorrelationImpl(const TCBinFeature* bf, int binFeatureCount, const float* binSums,
                                                    const TPartitionStatistics* parts, int pCount, int foldCount,
                                                    double l2, bool normalize,
                                                    double scoreStdDev, ui64 globalSeed,
                                                    TBestSplitProperties* result)
    {
        float bestScore = FLT_MAX;
        int bestIndex = 0;
        int tid = threadIdx.x;
        result += blockIdx.x;
        TPartOffsetsHelper helper(foldCount);



        for (int i = blockIdx.x * BLOCK_SIZE; i < binFeatureCount; i += BLOCK_SIZE * gridDim.x) {
            if (i + tid >= binFeatureCount) {
                break;
            }

            float score = 0;
            float denumSqr = 1e-20f;
            const float* current = binSums + 2 * (i + tid);

            for (int leaf = 0; leaf < pCount; leaf++) {

                #pragma unroll 4
                for (int fold = 0; fold < foldCount; fold += 2) {

                    TPartitionStatistics partLearn = LdgWithFallback(parts, helper.GetDataPartitionOffset(leaf, fold));
                    TPartitionStatistics partTest = LdgWithFallback(parts, helper.GetDataPartitionOffset(leaf, fold + 1));


                    float weightEstimateLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold) * 2];
                    float weightEstimateRight = max(partLearn.Weight - weightEstimateLeft, 0.0f);

                    float sumEstimateLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold) * 2 + 1];
                    float sumEstimateRight = partLearn.Sum - sumEstimateLeft;


                    float weightTestLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold + 1) * 2];
                    float weightTestRight = max(partTest.Weight - weightTestLeft, 0.0f);

                    float sumTestLeft = current[(size_t)binFeatureCount * helper.GetHistogramOffset(leaf, fold + 1) * 2 + 1];
                    float sumTestRight = partTest.Sum - sumTestLeft;


                    {
                        double lambda = normalize ? l2 * weightEstimateLeft : l2;

                        const float mu =  weightEstimateLeft > 0 ? (sumEstimateLeft / (weightEstimateLeft + lambda)) : 0;
                        score += sumTestLeft * mu;
                        denumSqr += weightTestLeft * mu * mu;
                    }

                    {
                        double lambda = normalize ? l2 * weightEstimateRight : l2;

                        const float mu =  weightEstimateRight > 0 ? (sumEstimateRight / (weightEstimateRight + lambda)) : 0;
                        score += sumTestRight * mu;
                        denumSqr += weightTestRight * mu * mu;
                    }
                }
            }

            score = denumSqr > 0 ? -score / sqrt(denumSqr) : FLT_MAX;
            float tmp = score;
            if (scoreStdDev) {
                ui64 seed = globalSeed + bf[i + tid].FeatureId;
                AdvanceSeed(&seed, 4);

                tmp += NextNormal(&seed) * scoreStdDev;
            }
            if (tmp < bestScore) {
                bestScore = tmp;
                bestIndex = i + tid;
            }
        }

        __shared__ float scores[BLOCK_SIZE];
        scores[tid] = bestScore;
        __shared__ int indices[BLOCK_SIZE];
        indices[tid] = bestIndex;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if (scores[tid] > scores[tid + s] ||
                    (scores[tid] == scores[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            result->FeatureId = bf[indices[0]].FeatureId;
            result->BinId = bf[indices[0]].BinId;
            result->Score = scores[0];
        }
    }



    void FindOptimalSplit(const TCBinFeature* binaryFeatures,ui32 binaryFeatureCount,
                          const float* splits, const TPartitionStatistics* parts, ui32 pCount, ui32 foldCount,
                          TBestSplitProperties* result, ui32 resultSize,
                          EScoreFunction scoreFunction, double l2, bool normalize,
                          double scoreStdDev, ui64 seed,
                          TCudaStream stream)
    {

        if (binaryFeatureCount > 0) {
            const int blockSize = 128;

            if (foldCount == 1) {
                switch (scoreFunction)
                {
                    case  EScoreFunction::SolarL2:
                    {
                        FindOptimalSplitSolarSingleFoldImpl<blockSize> << < resultSize, blockSize, 0, stream >> >
                                                                                                      (binaryFeatures, binaryFeatureCount, splits, parts, pCount, result);
                        break;
                    }
                    case  EScoreFunction::Correlation:
                    {
                        FindOptimalSplitCorrelationSingleFoldImpl<blockSize> << < resultSize, blockSize, 0, stream >> >
                                                                                                            (binaryFeatures, binaryFeatureCount, splits, parts, pCount, l2, normalize, scoreStdDev, seed, result);
                        break;
                    }
                    default:
                    {
                        throw std::exception();
                    }
                }
            } else {
                switch (scoreFunction)
                {
                    case  EScoreFunction::SolarL2:
                    {
                        FindOptimalSplitSolarImpl<blockSize> << < resultSize, blockSize, 0, stream >> >
                                                                                            (binaryFeatures, binaryFeatureCount, splits, parts, pCount, foldCount, result);
                        break;
                    }
                    case  EScoreFunction::Correlation:
                    {
                        FindOptimalSplitCorrelationImpl<blockSize> << < resultSize, blockSize, 0, stream >> >
                                                                                                  (binaryFeatures, binaryFeatureCount, splits, parts, pCount, foldCount, l2, normalize, scoreStdDev, seed, result);
                        break;
                    }
                    default:
                    {
                        throw std::exception();
                    }
                }
            }
        }
    }


    template <int BLOCK_SIZE, int HIST_COUNT>
    __global__ void GatherHistogramsByLeavesImpl(const int binFeatureCount,
                                                 const float* histogram,
                                                 const int histCount,
                                                 const int leafCount,
                                                 const int foldCount,
                                                 float* result) {

        const int featuresPerBlock = BLOCK_SIZE / leafCount;
        const int featureId = blockIdx.x * featuresPerBlock + threadIdx.x / leafCount;
        const int leafId = threadIdx.x & (leafCount - 1);

        const int foldId = blockIdx.y;
        TPartOffsetsHelper helper(gridDim.y);

        if (featureId < binFeatureCount) {
            float leafVals[HIST_COUNT];
            #pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId) {
                leafVals[histId] = LdgWithFallback(histogram,
                                                   (featureId + (size_t)binFeatureCount * helper.GetHistogramOffset(leafId, foldId)) * HIST_COUNT + histId);
            }

            #pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId)
            {
                const  ui64 idx = ((size_t)featureId * leafCount * foldCount + leafId * foldCount + foldId) * HIST_COUNT + histId;
                result[idx] = leafVals[histId];
            }
        }
    }

    bool GatherHistogramByLeaves(const float* histogram,
                                 const ui32 binFeatureCount,
                                 const ui32 histCount,
                                 const ui32 leafCount,
                                 const ui32 foldCount,
                                 float* result,
                                 TCudaStream stream
    )
    {
        const int blockSize = 1024;
        dim3 numBlocks;
        numBlocks.x = (binFeatureCount + (blockSize / leafCount) - 1) / (blockSize / leafCount);
        numBlocks.y = foldCount;
        numBlocks.z = 1;

        switch (histCount) {
            case 1: {
                GatherHistogramsByLeavesImpl<blockSize, 1> <<<numBlocks, blockSize, 0, stream>>>(binFeatureCount, histogram, histCount, leafCount, foldCount, result);
                return true;
            }
            case 2: {
                GatherHistogramsByLeavesImpl<blockSize, 2> <<<numBlocks, blockSize, 0, stream>>>(binFeatureCount, histogram, histCount, leafCount, foldCount, result);
                return true;
            }
            case 4: {
                GatherHistogramsByLeavesImpl<blockSize, 4> <<<numBlocks, blockSize, 0, stream>>>(binFeatureCount, histogram, histCount, leafCount, foldCount, result);
                return true;
            }
            default: {
                return false;
            }
        }
    }

    template <int BLOCK_SIZE>
    __global__ void PartitionUpdateImpl(const float* target,
                                        const float* weights,
                                        const float* counts,
                                        const struct TDataPartition* parts,
                                        struct TPartitionStatistics* partStats)
    {
        const int tid = threadIdx.x;
        parts += blockIdx.x;
        partStats += blockIdx.x;
        const int size = parts->Size;

        __shared__ volatile double localBuffer[BLOCK_SIZE];

        double tmp = 0;

        if (weights != 0) {
            localBuffer[tid] = ComputeSum<BLOCK_SIZE>(weights + parts->Offset, size);
            __syncthreads();
            tmp =  Reduce<double, BLOCK_SIZE>(localBuffer);
        }

        if (tid == 0)
        {
            partStats->Weight = tmp;
        }
        tmp =  0;
        __syncthreads();

        if (target != 0) {
            localBuffer[tid] = ComputeSum<BLOCK_SIZE>(target + parts->Offset, size);
            __syncthreads();
            tmp = Reduce<double, BLOCK_SIZE>(localBuffer);
        }

        if (tid == 0)
        {
             partStats->Sum = tmp;
        }

        tmp  = 0;
        __syncthreads();

        if (counts != 0) {
            localBuffer[tid] = ComputeSum<BLOCK_SIZE>(counts + parts->Offset, size);
            __syncthreads();
            tmp =  Reduce<double, BLOCK_SIZE>(localBuffer);
        } else {
           tmp = size;
        }

        if (tid == 0)
        {
            partStats->Count = tmp;
        }

    }

    void UpdatePartitionProps(const float* target,
                              const float* weights,
                              const float* counts,
                              const struct TDataPartition* parts,
                              struct TPartitionStatistics* partStats,
                              int partsCount,
                              TCudaStream stream
    )
    {
        const int blockSize = 1024;
        if (partsCount)
        {
            PartitionUpdateImpl<blockSize> << < partsCount, blockSize, 0, stream >> > (target, weights, counts, parts, partStats);
        }
    }



}
