#include "compute_scores.cuh"

#include <catboost/cuda/methods/kernel/score_calcers.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <contrib/libs/cub/cub/block/block_reduce.cuh>

#include <cmath>
#include <exception>
#include <cfloat>


namespace NKernel {

//    histId * binFeatureCount * statCount + statId * binFeatureCount + features->FirstFoldIndex;

    template <int BlockSize,
              class TScoreCalcer>
    __global__ void ComputeOptimalSplits(const TCBinFeature* bf,
                                         ui32 binFeatureCount,
                                         const float* histograms,
                                         const double* partStats, int statCount, const ui32* partIds, int pCount,
                                         bool multiclassOptimization,
                                         TScoreCalcer calcer,
                                         TBestSplitProperties* result) {

        float bestScore = FLT_MAX;
        int bestIndex = -1;
        int tid = threadIdx.x;

        result += blockIdx.x + blockIdx.y * gridDim.x;
        partIds += blockIdx.y * pCount;

        for (int offset = blockIdx.x * BlockSize; offset < binFeatureCount; offset += BlockSize * gridDim.x) {
            const int binFeatureId = offset + tid;

            if (binFeatureId >= binFeatureCount) {
                break;
            }
            calcer.NextFeature(bf[binFeatureId]);

            for (int i = 0; i < pCount; i++) {
                const int leafId = __ldg(partIds + i);

                const float weightLeft = max(__ldg(histograms + leafId * statCount * binFeatureCount + binFeatureId), 0.0f);
                const float weightRight = max(__ldg(partStats + leafId * statCount) - weightLeft, 0.0f);

                double totalSumLeft = 0;
                double totalSumPart = 0;
                for (int statId = 1; statId < statCount; ++statId) {
                    float sumLeft = __ldg(histograms + leafId * statCount * binFeatureCount + statId * binFeatureCount + binFeatureId);
                    double partStat = __ldg(partStats + leafId * statCount + statId);
                    totalSumPart += partStat;
                    float sumRight = static_cast<float>(partStat - sumLeft);

                    calcer.AddLeaf(sumLeft, weightLeft);
                    calcer.AddLeaf(sumRight, weightRight);
                    totalSumLeft += sumLeft;


                }
                if (multiclassOptimization) {
                    double totalSumRight = totalSumPart - totalSumLeft;
                    calcer.AddLeaf(-totalSumLeft, weightLeft);
                    calcer.AddLeaf(-totalSumRight, weightRight);
                }
            }

            const float score = calcer.GetScore();


            if (score < bestScore) {
                bestScore = score;
                bestIndex = binFeatureId;
            }
        }


        __shared__ float scores[BlockSize];
        scores[tid] = bestScore;

        __shared__ int indices[BlockSize];
        indices[tid] = bestIndex;
        __syncthreads();

        for (ui32 s = BlockSize >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                if (scores[tid] > scores[tid + s] || (scores[tid] == scores[tid + s] && indices[tid] > indices[tid + s]) ) {
                    scores[tid] = scores[tid + s];
                    indices[tid] = indices[tid + s];
                }
            }
            __syncthreads();
        }

        if (!tid) {
            const int index = indices[0];

            if (index != -1 && index < binFeatureCount) {
                result->FeatureId = bf[index].FeatureId;
                result->BinId = bf[index].BinId;
                result->Score = scores[0];
            } else {
                result->FeatureId = -1;
                result->BinId = -1;
                result->Score = FLT_MAX;

            }
        }
    }




    void ComputeOptimalSplits(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                              const float* histograms,
                              const double* partStats, int statCount,
                              ui32* partIds, int partBlockSize, int partBlockCount,
                              TBestSplitProperties* result, ui32 argmaxBlockCount,
                              EScoreFunction scoreFunction,
                              bool multiclassOptimization,
                              double l2,
                              bool normalize,
                              double scoreStdDev,
                              ui64 seed,
                              TCudaStream stream) {
        const int blockSize = 128;

        dim3 numBlocks;
        numBlocks.x = argmaxBlockCount;
        numBlocks.y = partBlockCount;
        numBlocks.z = 1;

        #define RUN() \
        ComputeOptimalSplits<blockSize, TScoreCalcer> << < numBlocks, blockSize, 0, stream >> > (binaryFeatures, binaryFeatureCount, histograms, partStats,  statCount, partIds, partBlockSize, multiclassOptimization, scoreCalcer, result);


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
                TScoreCalcer scoreCalcer(static_cast<float>(l2));
                RUN()
                break;
            }
            case  EScoreFunction::Correlation:
            case  EScoreFunction::NewtonCorrelation: {
                using TScoreCalcer = TCorrelationScoreCalcer;
                TCorrelationScoreCalcer scoreCalcer(static_cast<float>(l2),
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


    template <int BlockSize>
    __global__ void ComputeTargetVarianceImpl(const float* stats,
                                              ui32 size,
                                              ui32 statCount,
                                              ui64 statLineSize,
                                              bool isMulticlass,
                                              double* aggregatedStats) {

        ui32 i = BlockSize * blockIdx.x + threadIdx.x;

        float weightedSum = 0;
        float weightedSum2 = 0;
        float totalWeight = 0;

        while (i < size) {
            const float w = stats[i];
            if (w > 1e-15f) {
                float statSum = 0;
                for (ui32 statId = 1; statId < statCount; ++statId) {
                    const float wt = stats[i + statLineSize * statId];
                    weightedSum += wt;
                    weightedSum2 += wt * wt / w; //cause we need sum w * t * t
                    statSum += wt;
                }
                if (isMulticlass) {
                    weightedSum += -statSum;
                    weightedSum2 += statSum * statSum / w;
                }
                totalWeight += w;
            }
            i += gridDim.x * BlockSize;
        }

        using BlockReduce = typename cub::BlockReduce<double, BlockSize>;
        __shared__ typename BlockReduce::TempStorage tempStorage;

        double blockWeightedSum = weightedSum;
        blockWeightedSum = BlockReduce(tempStorage).Sum(blockWeightedSum);

        double blockWeightedSum2 = weightedSum2;


        blockWeightedSum2 = BlockReduce(tempStorage).Sum(blockWeightedSum2);

        double blockTotalWeight = totalWeight;
        blockTotalWeight = BlockReduce(tempStorage).Sum(blockTotalWeight);



        if (threadIdx.x == 0) {
            TAtomicAdd<double>::Add(aggregatedStats, blockWeightedSum);
            TAtomicAdd<double>::Add(aggregatedStats + 1, blockWeightedSum2);
            TAtomicAdd<double>::Add(aggregatedStats + 2, blockTotalWeight);
        }
    }


    void ComputeTargetVariance(const float* stats,
                               ui32 size,
                               ui32 statCount,
                               ui64 statLineSize,
                               bool isMulticlass,
                               double* aggregatedStats,
                               TCudaStream stream) {

        const ui32 blockSize = 512;
        const ui32 numBlocks = min(4 * TArchProps::SMCount(), CeilDivide(size, blockSize));
        FillBuffer(aggregatedStats, 0.0, 3, stream);
        if (numBlocks) {
            ComputeTargetVarianceImpl<blockSize><<<numBlocks, blockSize, 0, stream>>>(stats, size, statCount, statLineSize, isMulticlass, aggregatedStats);
        }
    }


}
