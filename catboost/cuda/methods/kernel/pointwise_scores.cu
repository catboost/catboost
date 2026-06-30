/**/#include "pointwise_scores.cuh"
#include "score_calcers.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

#include <cmath>
#include <exception>
#include <cfloat>

namespace {
    // load data in a streaming manner (LOAD_CS stands for cache streaming)
    // so that we do not pollute our cache
    template <int BLOCK_SIZE>
    __forceinline__ __device__  float ComputeSum(const float* buffer, ui32 count) {
        float sum = 0.f;
        const ui32 tid = threadIdx.x;
        // manually unroll inner loop instead of using #pragma unroll 16
        // because nvcc 11.4+ refuses to perform that kind of loop unrolling
        ui32 i = tid;
        const int ITERS = 16;
        for (; i + (ITERS - 1) * BLOCK_SIZE < count;) {
#pragma unroll
            for (int iter = 0; iter < ITERS; ++iter, i += BLOCK_SIZE) {
                sum += NKernel::StreamLoad(buffer + i);
            }
        }
        for (; i < count; i += BLOCK_SIZE) {
            sum += NKernel::StreamLoad(buffer + i);
        }

        return sum;
    };
}

namespace NKernel {


    template <int BLOCK_SIZE>
    __global__ void FindOptimalSplitSolarImpl(const TCBinFeature* bf,
                                              int binFeatureCount,
                                              const float* catFeaturesWeights,
                                              const float* binFeaturesWeights,
                                              int binFeaturesWeightsCount,
                                              const float* binSums,
                                              const TPartitionStatistics* parts,
                                              int pCount, int foldCount,
                                              double scoreBeforeSplit,
                                              TBestSplitProperties* result)
    {
        float bestScore = FLT_MAX;
        float bestGain = FLT_MAX;
        int bestIndex = 0;
        int tid = threadIdx.x;
        result += blockIdx.x;

        TPointwisePartOffsetsHelper helper(foldCount);

        for (int i = blockIdx.x * BLOCK_SIZE; i < binFeatureCount; i += BLOCK_SIZE * gridDim.x) {
            if (i + tid >= binFeatureCount) {
                break;
            }
            if (bf[i + tid].SkipInScoreCount) {
                continue;
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

            ui32 featureId = bf[i + tid].FeatureId;
            score *= __ldg(catFeaturesWeights + featureId);
            float gain = (score - scoreBeforeSplit) * __ldg(binFeaturesWeights + featureId);

            if (gain < bestGain) {
                bestScore = score;
                bestGain = gain;
                bestIndex = i + tid;
            }
        }

        __shared__ float scores[BLOCK_SIZE];
        scores[tid] = bestScore;
        __shared__ int indices[BLOCK_SIZE];
        indices[tid] = bestIndex;
        __shared__ float gains[BLOCK_SIZE];
        gains[tid] = bestGain;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
            if ( gains[tid] > gains[tid + s] ||
                (gains[tid] == gains[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                    gains[tid] = gains[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            const int index = indices[0];
            result->FeatureId =  index < binFeatureCount ? bf[index].FeatureId : 0;
            result->BinId = index < binFeatureCount ? bf[index].BinId : 0;
            result->Score = scores[0];
            result->Gain = gains[0];
        }
    }



    class TDirectHistLoader {
    public:
        __forceinline__ __device__ TDirectHistLoader(const float* binSums,
                                      TPointwisePartOffsetsHelper& helper,
                                     int binFeatureId,
                                     int /* leaf count*/,
                                     int binFeatureCount)
                : BinSums(binSums + 2 * binFeatureId)
                , Helper(helper)
                , BinFeatureCount(binFeatureCount) {

        }

        __forceinline__ __device__ float LoadWeight(int leaf) {
            return BinSums[(size_t)BinFeatureCount * Helper.GetHistogramOffset(leaf, 0) * 2];
        }

        __forceinline__ __device__ float LoadSum(int leaf) {
            return BinSums[(size_t)BinFeatureCount * Helper.GetHistogramOffset(leaf, 0) * 2 + 1];
        }
    private:
        const float* BinSums;
         TPointwisePartOffsetsHelper& Helper;
        int BinFeatureCount;
    };


    class TGatheredByLeavesHistLoader {
    public:
        __forceinline__ __device__ TGatheredByLeavesHistLoader(const float* binSums,
                                                                TPointwisePartOffsetsHelper&,
                                                               int binFeatureId,
                                                               int leafCount,
                                                               int /*binFeatureCount*/)
                : BinSums(binSums)
                , LeafCount(leafCount)
                , FeatureId(binFeatureId) {

        }

        __forceinline__ __device__ int GetOffset(int leaf) {
            return 2 * (FeatureId * LeafCount + leaf);
        }

        __forceinline__ __device__ float LoadWeight(int leaf) {
            return BinSums[GetOffset(leaf)];
        }

        __forceinline__ __device__ float LoadSum(int leaf) {
            return BinSums[GetOffset(leaf) + 1];
        }

    private:
        const float* BinSums;
        int LeafCount;
        int FeatureId;
    };

    template <int BLOCK_SIZE,
            class THistLoader,
            class TScoreCalcer>
    __global__ void FindOptimalSplitSingleFoldImpl(const TCBinFeature* bf,
                                                   int binFeatureCount,
                                                   const float* catFeaturesWeights,
                                                   const float* binFeaturesWeights,
                                                   int binFeaturesWeightsCount,
                                                   double scoreBeforeSplit,
                                                   const float* binSums,
                                                   const TPartitionStatistics* parts,
                                                   int pCount,
                                                   TScoreCalcer calcer,
                                                   TBestSplitProperties* result) {
        float bestScore = FLT_MAX;
        float bestGain = FLT_MAX;
        int bestIndex = 0;
        int tid = threadIdx.x;
        result += blockIdx.x;

        TPointwisePartOffsetsHelper helper(1);

        for (int i = blockIdx.x * BLOCK_SIZE; i < binFeatureCount; i += BLOCK_SIZE * gridDim.x) {
            if (i + tid >= binFeatureCount) {
                break;
            }
            if (bf[i + tid].SkipInScoreCount) {
                continue;
            }
            calcer.NextFeature(bf[i + tid]);

            THistLoader histLoader(binSums,
                                   helper,
                                   i + tid,
                                   pCount,
                                   binFeatureCount);

            for (int leaf = 0; leaf < pCount; leaf++) {
                TPartitionStatistics part = LdgWithFallback(parts, helper.GetDataPartitionOffset(leaf, 0));

                float weightLeft = histLoader.LoadWeight(leaf);
                float weightRight = max(part.Weight - weightLeft, 0.0f);

                float sumLeft = histLoader.LoadSum(leaf);
                float sumRight = static_cast<float>(part.Sum - sumLeft);

                calcer.AddLeaf(sumLeft, weightLeft);
                calcer.AddLeaf(sumRight, weightRight);
            }
            float score = calcer.GetScore();

            ui32 featureId = bf[i + tid].FeatureId;
            score *= __ldg(catFeaturesWeights + featureId);
            float gain = score - scoreBeforeSplit;
            gain *= __ldg(binFeaturesWeights + featureId);


            if (gain < bestGain) {
                bestScore = score;
                bestGain = gain;
                bestIndex = i + tid;
            }
        }

        __shared__ float scores[BLOCK_SIZE];
        scores[tid] = bestScore;
        __shared__ int indices[BLOCK_SIZE];
        indices[tid] = bestIndex;
        __shared__ float gains[BLOCK_SIZE];
        gains[tid] = bestGain;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if ( gains[tid] > gains[tid + s] ||
                     (gains[tid] == gains[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                    gains[tid] = gains[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            const int index = indices[0];
            result->FeatureId =  index < binFeatureCount ? bf[index].FeatureId : 0;
            result->BinId = index < binFeatureCount ? bf[index].BinId : 0;
            result->Score = scores[0];
            result->Gain = gains[0];
        }
    }





    template <int BLOCK_SIZE>
    __global__ void FindOptimalSplitCosineImpl(const TCBinFeature* bf, int binFeatureCount,
                                               const float* catFeaturesWeights,
                                               const float* binFeaturesWeights,
                                               int binFeaturesWeightsCount,
                                               const float* binSums,
                                               const TPartitionStatistics* parts, int pCount, int foldCount,
                                               double scoreBeforeSplit,
                                               double l2, bool normalize,
                                               double scoreStdDev, ui64 globalSeed,
                                               TBestSplitProperties* result)
    {
        float bestScore = FLT_MAX;
        float bestGain = FLT_MAX;
        int bestIndex = 0;
        int tid = threadIdx.x;
        result += blockIdx.x;
        TPointwisePartOffsetsHelper helper(foldCount);



        for (int i = blockIdx.x * BLOCK_SIZE; i < binFeatureCount; i += BLOCK_SIZE * gridDim.x) {
            if (i + tid >= binFeatureCount) {
                break;
            }
            if (bf[i + tid].SkipInScoreCount) {
                continue;
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

            score = denumSqr > 1e-15f ? -score / sqrt(denumSqr) : FLT_MAX;

            ui32 featureId = bf[i + tid].FeatureId;
            score *= __ldg(catFeaturesWeights + featureId);

            float noisyScore = score;
            if (scoreStdDev) {
                ui64 seed = globalSeed + featureId;
                AdvanceSeed(&seed, 4);

                noisyScore += NextNormal(&seed) * scoreStdDev;
            }
            const float gain = (noisyScore - scoreBeforeSplit) * __ldg(binFeaturesWeights + featureId);
            if (gain < bestGain) {
                bestScore = noisyScore;
                bestGain = gain;
                bestIndex = i + tid;
            }
        }

        __shared__ float scores[BLOCK_SIZE];
        scores[tid] = bestScore;
        __shared__ int indices[BLOCK_SIZE];
        indices[tid] = bestIndex;
        __shared__ float gains[BLOCK_SIZE];
        gains[tid] = bestGain;
        __syncthreads();

        for (ui32 s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if (gains[tid] > gains[tid + s] ||
                    (gains[tid] == gains[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                    gains[tid] = gains[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            const int index = indices[0];
            result->FeatureId =  index < binFeatureCount ? bf[index].FeatureId : 0;
            result->BinId = index < binFeatureCount ? bf[index].BinId : 0;
            result->Score = scores[0];
            result->Gain = gains[0];
        }
    }




    void FindOptimalSplitDynamic(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                                 const float* catFeaturesWeights,
                                 const float* binFeaturesWeights, ui32 binaryFeatureWeightsCount,
                                 const float* splits, const TPartitionStatistics* parts, ui32 pCount, ui32 foldCount,
                                 double scoreBeforeSplit,
                                 TBestSplitProperties* result, ui32 resultSize,
                                 EScoreFunction scoreFunction, double l2, bool normalize,
                                 double scoreStdDev, ui64 seed,
                                 TCudaStream stream) {
        const int blockSize = 128;
        switch (scoreFunction)
        {
            case  EScoreFunction::SolarL2: {
                FindOptimalSplitSolarImpl<blockSize> << < resultSize, blockSize, 0, stream >> > (binaryFeatures, binaryFeatureCount, catFeaturesWeights, binFeaturesWeights, binaryFeatureWeightsCount, splits, parts, pCount, foldCount, scoreBeforeSplit, result);
                break;
            }
            case  EScoreFunction::Cosine:
            case  EScoreFunction::NewtonCosine: {
                FindOptimalSplitCosineImpl<blockSize> << < resultSize, blockSize, 0, stream >> > (binaryFeatures, binaryFeatureCount, catFeaturesWeights, binFeaturesWeights, binaryFeatureWeightsCount, splits, parts, pCount, foldCount, scoreBeforeSplit, l2, normalize, scoreStdDev, seed, result);
                break;
            }
            default: {
                throw std::exception();
            }
        }
    }

    template <class TLoader>
    void FindOptimalSplitPlain(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                               const float* catFeaturesWeights,
                               const float* binFeaturesWeights, ui32 binaryFeatureWeightsCount,
                               const float* splits, const TPartitionStatistics* parts, ui32 pCount,
                               double scoreBeforeSplit,
                               TBestSplitProperties* result, ui32 resultSize,
                               EScoreFunction scoreFunction, double l2, double metaL2Exponent, double metaFrequency, bool normalize,
                               double scoreStdDev, ui64 seed,
                               TCudaStream stream) {
        const int blockSize = 128;
        #define RUN() \
        FindOptimalSplitSingleFoldImpl<blockSize, TLoader, TScoreCalcer> << < resultSize, blockSize, 0, stream >> > (binaryFeatures, binaryFeatureCount, catFeaturesWeights, binFeaturesWeights, binaryFeatureWeightsCount, scoreBeforeSplit, splits, parts, pCount, scoreCalcer, result);


        switch (scoreFunction)
        {
            case  EScoreFunction::SolarL2: {
                using TScoreCalcer = TSolarScoreCalcer;
                TScoreCalcer scoreCalcer(static_cast<float>(l2));
                RUN()
                break;
            }
            case  EScoreFunction::SatL2: {
                using TScoreCalcer = TSatL2ScoreCalcer;
                TScoreCalcer scoreCalcer(static_cast<float>(l2));
                RUN()
                break;
            }
            case  EScoreFunction::LOOL2: {
                using TScoreCalcer = TLOOL2ScoreCalcer;
                TScoreCalcer scoreCalcer(static_cast<float>(l2));
                RUN()
                break;
            }
            case EScoreFunction::L2:
            case EScoreFunction::NewtonL2: {
                using TScoreCalcer = TL2ScoreCalcer;
                const float metaExponent = (NextUniform(&seed) >= metaFrequency ? 1.0f : static_cast<float>(metaL2Exponent));
                TScoreCalcer scoreCalcer(static_cast<float>(l2), metaExponent);
                RUN()
                break;
            }
            case  EScoreFunction::Cosine:
            case  EScoreFunction::NewtonCosine: {
                using TScoreCalcer = TCosineScoreCalcer;
                TCosineScoreCalcer scoreCalcer(static_cast<float>(l2),
                                                    normalize,
                                                    static_cast<float>(scoreStdDev),
                                                    seed);
                RUN()
                break;
            }
            default: {
                throw std::exception();
            }
        }
        #undef RUN
    }


    void FindOptimalSplit(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                          const float* catFeaturesWeights,
                          const float* binFeaturesWeights, ui32 binaryFeatureWeightsCount,
                          const float* splits, const TPartitionStatistics* parts, ui32 pCount, ui32 foldCount,
                          double scoreBeforeSplit,
                          TBestSplitProperties* result, ui32 resultSize,
                          EScoreFunction scoreFunction, double l2, double metaL2Exponent, double metaL2Frequency, bool normalize,
                          double scoreStdDev, ui64 seed, bool gatheredByLeaves,
                          TCudaStream stream)
    {

        if (foldCount == 1) {
            if (gatheredByLeaves) {
                using THistLoader = TGatheredByLeavesHistLoader;
                FindOptimalSplitPlain<THistLoader>(binaryFeatures, binaryFeatureCount, catFeaturesWeights, binFeaturesWeights, binaryFeatureWeightsCount, splits, parts, pCount, scoreBeforeSplit, result, resultSize, scoreFunction, l2, metaL2Exponent, metaL2Frequency, normalize, scoreStdDev, seed, stream);
            } else {
                using THistLoader = TDirectHistLoader;
                FindOptimalSplitPlain<THistLoader>(binaryFeatures, binaryFeatureCount, catFeaturesWeights, binFeaturesWeights, binaryFeatureWeightsCount, splits, parts, pCount, scoreBeforeSplit, result, resultSize, scoreFunction, l2, metaL2Exponent, metaL2Frequency, normalize, scoreStdDev, seed, stream);
            }
        } else {
            FindOptimalSplitDynamic(binaryFeatures, binaryFeatureCount, catFeaturesWeights, binFeaturesWeights, binaryFeatureWeightsCount, splits, parts, pCount, foldCount, scoreBeforeSplit, result, resultSize, scoreFunction, l2, normalize, scoreStdDev, seed, stream);
        }
    }


    template <int BLOCK_SIZE, int HIST_COUNT>
    __global__ void GatherHistogramsByLeavesImpl(const int binFeatureCount,
                                                 const float* histogram,
                                                 const int histCount,
                                                 const int leafCount,
                                                 const int foldCount,
                                                 float* result) {

        const int featuresPerBlock = (BLOCK_SIZE + leafCount - 1) / leafCount;
        const int featureId = blockIdx.x * featuresPerBlock + threadIdx.x / leafCount;
        const int leafId = (threadIdx.x & (leafCount - 1)) + threadIdx.z * BLOCK_SIZE;

        const int foldId = blockIdx.y;
        TPointwisePartOffsetsHelper helper(gridDim.y);

        if (featureId < binFeatureCount) {
            float leafVals[HIST_COUNT];
            #pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId) {
                leafVals[histId] = LdgWithFallback(histogram,
                                                   (featureId + (size_t)binFeatureCount * helper.GetHistogramOffset(leafId, foldId)) * HIST_COUNT + histId);
            }

            #pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId) {
                const  ui64 idx = ((size_t)featureId * leafCount * foldCount + leafId * foldCount + foldId) * HIST_COUNT + histId;
                result[idx] = leafVals[histId];
            }
        }
    }

    void GatherHistogramByLeaves(const float* histogram,
                                 const ui32 binFeatureCount,
                                 const ui32 histCount,
                                 const ui32 leafCount,
                                 const ui32 foldCount,
                                 float* result,
                                 TCudaStream stream
    )
    {
        const int blockSize = 1024;
        const int leavesInBlock = Min<int>(leafCount, blockSize);
        dim3 numBlocks;
        numBlocks.x = (binFeatureCount + (blockSize / leavesInBlock) - 1) / (blockSize / leavesInBlock);
        numBlocks.y = foldCount;
        numBlocks.z = (leafCount + blockSize - 1) / blockSize;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        switch (histCount) {
            case 1: {
                GatherHistogramsByLeavesImpl<blockSize, 1> <<<numBlocks, blockSize, 0, stream>>>(binFeatureCount, histogram, histCount, leafCount, foldCount, result);
                return;
            }
            case 2: {
                GatherHistogramsByLeavesImpl<blockSize, 2> <<<numBlocks, blockSize, 0, stream>>>(binFeatureCount, histogram, histCount, leafCount, foldCount, result);
                return;
            }
            case 4: {
                GatherHistogramsByLeavesImpl<blockSize, 4> <<<numBlocks, blockSize, 0, stream>>>(binFeatureCount, histogram, histCount, leafCount, foldCount, result);
                return;
            }
            default: {
                CB_ENSURE_INTERNAL(false, "histCount should be 1, 2, or 4, not " << histCount);
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
        const ui32 tid = threadIdx.x;
        parts += blockIdx.x;
        partStats += blockIdx.x;
        const ui32 size = parts->Size;

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

        if (tid == 0) {
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
        if (partsCount) {
            PartitionUpdateImpl<blockSize> << < partsCount, blockSize, 0, stream >> > (target, weights, counts, parts, partStats);
        }
    }



}
