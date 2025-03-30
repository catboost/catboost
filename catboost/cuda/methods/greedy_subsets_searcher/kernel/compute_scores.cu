#include "compute_scores.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>

#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/random_gen.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>
#include <catboost/cuda/methods/kernel/score_calcers.cuh>

#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <exception>
#include <cfloat>


namespace NKernel {

    #define ARGMAX() \
    __shared__ float scores[BlockSize]; \
    scores[tid] = bestScore; \
    __shared__ int indices[BlockSize]; \
    indices[tid] = bestIndex; \
    __shared__ float gains[BlockSize]; \
    gains[tid] = bestGain; \
    __syncthreads();\
    for (ui32 s = BlockSize >> 1; s > 0; s >>= 1) { \
        if (tid < s) { \
            if (gains[tid] > gains[tid + s] || (gains[tid] == gains[tid + s] && indices[tid] > indices[tid + s]) ) { \
                scores[tid] = scores[tid + s]; \
                indices[tid] = indices[tid + s]; \
                gains[tid] = gains[tid + s]; \
        }\
    }\
        __syncthreads();\
    } \
    if (!tid) { \
        const int index = indices[0];\
        if (index != -1 && index < binFeatureCount) { \
            result->FeatureId = bf[index].FeatureId;\
            result->BinId = bf[index].BinId;\
            result->Score = scores[0];\
            result->Gain = gains[0];\
        } else {\
            result->FeatureId = static_cast<ui32>(-1);\
            result->BinId = static_cast<ui32>(-1);\
            result->Score = FLT_MAX;\
            result->Gain = FLT_MAX;\
        }\
    }
//    histId * binFeatureCount * statCount + statId * binFeatureCount + features->FirstFoldIndex;

    template <int BlockSize,
              class TScoreCalcer>
    __global__ void ComputeOptimalSplits(const TCBinFeature* bf,
                                         ui32 binFeatureCount,
                                         const float* binFeaturesWeights, ui32 binFeaturesWeightsCount,
                                         const float* histograms,
                                         const double* partStats, int statCount,
                                         const ui32* partIds, int pCount,
                                         const ui32* restPartIds, int restPartCount,
                                         bool multiclassOptimization,
                                         TScoreCalcer calcer,
                                         TBestSplitProperties* result) {

        float bestScore = FLT_MAX;
        float bestGain = FLT_MAX;
        int bestIndex = -1;
        int tid = threadIdx.x;

        result += blockIdx.x + blockIdx.y * gridDim.x;
        partIds += blockIdx.y * pCount;

        for (int offset = blockIdx.x * BlockSize; offset < binFeatureCount; offset += BlockSize * gridDim.x) {
            const int binFeatureId = offset + tid;

            if (binFeatureId >= binFeatureCount) {
                break;
            }
            if (bf[binFeatureId].SkipInScoreCount) {
                continue;
            }
            calcer.NextFeature(bf[binFeatureId]);
            TScoreCalcer beforeSplitCalcer = calcer;

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

            //add fixed leaves
            for (int i = 0; i < restPartCount; i++) {
                const int leafId = __ldg(restPartIds + i);
                const float weight =  max(__ldg(partStats + leafId * statCount), 0.0f);
                double totalSum = 0;
                double totalSumPart = 0;

                for (int statId = 1; statId < statCount; ++statId) {
                    double sum = __ldg(partStats + leafId * statCount + statId);
                    totalSumPart += sum;

                    calcer.AddLeaf(sum, weight);
                    totalSum += sum;
                }
                if (multiclassOptimization) {
                    calcer.AddLeaf(-totalSum, weight);
                }
            }

            float score = calcer.GetScore();
            const float scoreBefore = beforeSplitCalcer.GetScore();

            float gain = score - scoreBefore;

            const ui32 featureId = bf[binFeatureId].FeatureId;
            gain *= __ldg(binFeaturesWeights + featureId);

            if (gain < bestGain) {
                bestScore = score;
                bestGain = gain;
                bestIndex = binFeatureId;
            }
        }


        ARGMAX()
    }




    void ComputeOptimalSplits(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                              const float* binFeaturesWeights, ui32 binFeaturesWeightsCount,
                              const float* histograms,
                              const double* partStats, int statCount,
                              const ui32* partIds, int partBlockSize, int partBlockCount,
                              const ui32* restPartIds, int restPartCount,
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
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        #define RUN() \
        ComputeOptimalSplits<blockSize, TScoreCalcer> << < numBlocks, blockSize, 0, stream >> > (binaryFeatures, binaryFeatureCount, binFeaturesWeights, binFeaturesWeightsCount, histograms, partStats,  statCount, partIds, partBlockSize, restPartIds, restPartCount, multiclassOptimization, scoreCalcer, result);


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



    template <int BlockSize,
            class TScoreCalcer>
    __global__ void ComputeOptimalSplitsRegion(const TCBinFeature* bf,
                                               ui32 binFeatureCount,
                                               const float* binFeaturesWeights,
                                               ui32 binFeaturesWeightsCount,
                                               const float* histograms,
                                               const double* partStats, int statCount,
                                               const ui32* partIds,
                                               bool multiclassOptimization,
                                               TScoreCalcer calcer,
                                               TBestSplitProperties* result) {

        float bestScore = FLT_MAX;
        int bestIndex = -1;
        float bestGain = FLT_MAX;
        int tid = threadIdx.x;

        result += blockIdx.x + blockIdx.y * gridDim.x;
        partIds += blockIdx.y;

        const int thisPartId = partIds[0];

        for (int offset = blockIdx.x * BlockSize; offset < binFeatureCount; offset += BlockSize * gridDim.x) {
            const int binFeatureId = offset + tid;

            if (binFeatureId >= binFeatureCount) {
                break;
            }
            if (bf[binFeatureId].SkipInScoreCount) {
                continue;
            }
            calcer.NextFeature(bf[binFeatureId]);
            TScoreCalcer beforeSplitCalcer = calcer;

            const double partWeight = __ldg(partStats + thisPartId  * statCount);
            const float weightLeft = max(__ldg(histograms + thisPartId  * statCount * binFeatureCount + binFeatureId), 0.0f);
            const float weightRight = max(partWeight - weightLeft, 0.0f);
            bool toZeroPartSplit = false;

            if (weightLeft < 1e-20f || weightRight < 1e-20f) {
                toZeroPartSplit = true;
            }

            double totalSumLeft = 0;
            double totalSumPart = 0;

            for (int statId = 1; statId < statCount; ++statId) {
                float sumLeft = __ldg(histograms + thisPartId * statCount * binFeatureCount + statId * binFeatureCount + binFeatureId);
                double partStat = __ldg(partStats + thisPartId * statCount + statId);
                totalSumPart += partStat;
                float sumRight = static_cast<float>(partStat - sumLeft);

                calcer.AddLeaf(sumLeft, weightLeft);
                calcer.AddLeaf(sumRight, weightRight);

                beforeSplitCalcer.AddLeaf(partStat, partWeight);
                totalSumLeft += sumLeft;
            }
            if (multiclassOptimization) {
                double totalSumRight = totalSumPart - totalSumLeft;
                calcer.AddLeaf(-totalSumLeft, weightLeft);
                calcer.AddLeaf(-totalSumRight, weightRight);

                beforeSplitCalcer.AddLeaf(-totalSumPart, partWeight);
            }


            const bool skip = toZeroPartSplit;
            const float scoreAfter = !skip ? calcer.GetScore() : FLT_MAX;
            const float scoreBefore = !skip ? beforeSplitCalcer.GetScore() : FLT_MAX;

            //-10 - 0 = -10
            //in GPU catboost all scores are inverse, lower is better
            float gain = !skip ? (scoreAfter - scoreBefore) : 0;

            const ui32 featureId = bf[binFeatureId].FeatureId;
            gain *= __ldg(binFeaturesWeights + featureId);

            if (gain < bestScore) {
                bestScore = gain;
                bestIndex = binFeatureId;
                bestGain = gain;
            }
        }

        ARGMAX()
    }


    template <int BlockSize,
        class TScoreCalcer>
    __global__ void ComputeOptimalSplit(const TCBinFeature* bf,
                                        ui32 binFeatureCount,
                                        const float* binFeaturesWeights,
                                        ui32 binFeaturesWeightsCount,
                                        const float* histograms,
                                        const double* partStats, int statCount,
                                        const int partId,
                                        const int maybeSecondPartId,
                                        bool multiclassOptimization,
                                        TScoreCalcer calcer,
                                        TBestSplitProperties* result) {

        float bestScore = FLT_MAX;
        int bestIndex = -1;
        float bestGain = FLT_MAX;
        int tid = threadIdx.x;
        result += blockIdx.x + blockIdx.y * gridDim.x;
        const int thisPartId = blockIdx.y == 0 ? partId : maybeSecondPartId;


        for (int offset = blockIdx.x * BlockSize; offset < binFeatureCount; offset += BlockSize * gridDim.x) {
            const int binFeatureId = offset + tid;

            if (binFeatureId >= binFeatureCount) {
                break;
            }
            if (bf[binFeatureId].SkipInScoreCount) {
                continue;
            }
            calcer.NextFeature(bf[binFeatureId]);
            TScoreCalcer beforeSplitCalcer = calcer;

            const double partWeight = __ldg(partStats + thisPartId  * statCount);
            const float weightLeft = max(__ldg(histograms + thisPartId  * statCount * binFeatureCount + binFeatureId), 0.0f);
            const float weightRight = max(partWeight - weightLeft, 0.0f);
            bool toZeroPartSplit = false;

            if (weightLeft < 1e-20f || weightRight < 1e-20f) {
                toZeroPartSplit = true;
            }

            double totalSumLeft = 0;
            double totalSumPart = 0;

            for (int statId = 1; statId < statCount; ++statId) {
                float sumLeft = __ldg(histograms + thisPartId * statCount * binFeatureCount + statId * binFeatureCount + binFeatureId);
                double partStat = __ldg(partStats + thisPartId * statCount + statId);
                totalSumPart += partStat;
                float sumRight = static_cast<float>(partStat - sumLeft);

                calcer.AddLeaf(sumLeft, weightLeft);
                calcer.AddLeaf(sumRight, weightRight);

                beforeSplitCalcer.AddLeaf(partStat, partWeight);
                totalSumLeft += sumLeft;
            }
            if (multiclassOptimization) {
                double totalSumRight = totalSumPart - totalSumLeft;
                calcer.AddLeaf(-totalSumLeft, weightLeft);
                calcer.AddLeaf(-totalSumRight, weightRight);

                beforeSplitCalcer.AddLeaf(-totalSumPart, partWeight);
            }


            const bool skip = toZeroPartSplit;
            const float scoreAfter = !skip ? calcer.GetScore() : FLT_MAX;
            const float scoreBefore = !skip ? beforeSplitCalcer.GetScore() : FLT_MAX;

            //-10 - 0 = -10
            //in GPU catboost all scores are inverse, lower is better
            float gain = !skip ? (scoreAfter - scoreBefore) : 0;

            const ui32 featureId = bf[binFeatureId].FeatureId;
            gain *= __ldg(binFeaturesWeights + featureId);

            if (gain < bestScore) {
                bestScore = gain;
                bestIndex = binFeatureId;
                bestGain = gain;
            }
        }

        ARGMAX()
    }


    void ComputeOptimalSplitsRegion(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                                    const float* binFeaturesWeights, ui32 binFeaturesWeightsCount,
                                    const float* histograms,
                                    const double* partStats, int statCount,
                                    const ui32* partIds, int partCount,
                                    TBestSplitProperties* result, ui32 argmaxBlockCount,
                                    EScoreFunction scoreFunction,
                                    bool multiclassOptimization,
                                    double l2,
                                    bool normalize,
                                    double scoreStdDev,
                                    ui64 seed,
                                    TCudaStream stream) {
        const int blockSize = 256;

        dim3 numBlocks;
        numBlocks.x = argmaxBlockCount;
        numBlocks.y = partCount;
        numBlocks.z = 1;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        #define RUN() \
        ComputeOptimalSplitsRegion<blockSize, TScoreCalcer> << < numBlocks, blockSize, 0, stream >> > (binaryFeatures, binaryFeatureCount, binFeaturesWeights, binFeaturesWeightsCount, histograms, partStats,  statCount, partIds, multiclassOptimization, scoreCalcer, result);


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


    void ComputeOptimalSplit(const TCBinFeature* binaryFeatures, ui32 binaryFeatureCount,
                            const float* binFeaturesWeights, ui32 binFeaturesWeightsCount,
                            const float* histograms,
                            const double* partStats, int statCount,
                            ui32 partId, ui32 maybeSecondPartId,
                            TBestSplitProperties* result, ui32 argmaxBlockCount,
                            EScoreFunction scoreFunction,
                            bool multiclassOptimization,
                            double l2,
                            bool normalize,
                            double scoreStdDev,
                            ui64 seed,
                            TCudaStream stream) {
        const int blockSize = 256;

        dim3 numBlocks;
        numBlocks.x = argmaxBlockCount;
        numBlocks.y = partId == maybeSecondPartId ? 1 : 2;
        numBlocks.z = 1;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        #define RUN() \
        ComputeOptimalSplit<blockSize, TScoreCalcer> << < numBlocks, blockSize, 0, stream >> > (binaryFeatures, binaryFeatureCount, binFeaturesWeights, binFeaturesWeightsCount, histograms, partStats,  statCount, partId, maybeSecondPartId, multiclassOptimization, scoreCalcer, result);


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



    //seems like this'll be faster on CPU
    template <class TScoreCalcer>
    void ComputeTreeScoreImpl(const double* partStats, int statCount,
                              const ui32* allPartIds, int allPartCount,
                              bool multiclassOptimization,
                              TScoreCalcer calcer,
                              double* result) {
        calcer.NextFeature(TCBinFeature({100500, 42}));

        for (int i = 0; i < allPartCount; ++i) {
            const int leafId = allPartIds[i];
            const double weight = max(partStats[leafId * statCount], 0.0);
            double totalSum = 0;

            for (int statId = 1; statId < statCount; ++statId) {
                double sum = partStats[leafId * statCount + statId];

                calcer.AddLeaf(sum, weight);
                totalSum += sum;
            }
            if (multiclassOptimization) {
                calcer.AddLeaf(-totalSum, weight);
            }
        }
        result[0] = calcer.GetScore();
    }



    void ComputeTreeScore(
        const double* partStats,
        int statCount,
        const ui32* allPartIds,
        int allPartCount,
        EScoreFunction scoreFunction,
        bool multiclassOptimization,
        double l2,
        bool normalize,
        double scoreStdDev,
        ui64 seed,
        double* result,
        TCudaStream) {

        #define RUN() \
        ComputeTreeScoreImpl(partStats, statCount, allPartIds, allPartCount, multiclassOptimization, scoreCalcer, result);


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

    #undef ARGMAX
}
