#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/inplace_scan.cuh>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cassert>

namespace NKernel {

    inline ui32 EstimateBlockPerFeatureMultiplier(dim3 numBlocks, ui32 dsSize, ui32 limit = 128) {
        int blocksPerSm = TArchProps::GetMajorVersion() < 5 ? 1 : 2;
        ui32 multiplier = 1;
        while ((numBlocks.x * numBlocks.y * min(numBlocks.z, 8) * multiplier < TArchProps::SMCount() * blocksPerSm * 1.25) &&
               ((dsSize / multiplier) > 10000) && (multiplier < limit)) {
            multiplier *= 2;
        }
        return multiplier;
    }

    __forceinline__ __device__ ui32 GetMaxBinCount(const TCFeature* features, int fCount, ui32* smem) {

        ui32 binCount = threadIdx.x < fCount ? features[threadIdx.x].Folds : 0;
        smem[threadIdx.x] = binCount;
        __syncthreads();

        if (threadIdx.x < 2) {
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + 2]);
        }

        __syncthreads();
        if (threadIdx.x < 1) {
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + 1]);
        }
        __syncthreads();
        ui32 result = smem[0];
        __syncthreads();

        return result;
    }


    __forceinline__ __device__ bool HasOneHotFeatures(const TCFeature* features, int fCount, int* smem) {

        int flag = threadIdx.x < fCount && features[threadIdx.x].OneHotFeature ? 1 : 0;
        smem[threadIdx.x] = flag;
        __syncthreads();

        if (threadIdx.x < 2) {
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + 2]);
        }

        __syncthreads();
        if (threadIdx.x < 1) {
            smem[threadIdx.x] = max(smem[threadIdx.x], smem[threadIdx.x + 1]);
        }
        __syncthreads();
        int result = smem[0];
        __syncthreads();

        return result;
    }


    struct TPointwisePartOffsetsHelper {
        ui32 FoldCount;

        __forceinline__ __device__ TPointwisePartOffsetsHelper(ui32 foldCount)
                : FoldCount(foldCount) {

        }
        __forceinline__ __device__ ui32 GetHistogramOffset(ui32 partId, ui32 foldId) {
            return partId * FoldCount + foldId;
        }

        __forceinline__ __device__ ui32 GetDataPartitionOffset(ui32 partId, ui32 foldId) {
            const int foldStripe = 1 << static_cast<ui32>(ceil(log2(static_cast<float>(FoldCount))));
            return partId * foldStripe + foldId;
        }


        __forceinline__ __device__ void ShiftPartAndBinSumsPtr(const TDataPartition* __restrict__& partition,
                                                               float* __restrict__& binSums,
                                                               ui32 totalFeatureCount,
                                                               bool fullPass,
                                                               int histCount = 2) {
            const int histLineSize = histCount * totalFeatureCount;
            if (fullPass) {
                partition += GetDataPartitionOffset(blockIdx.y, blockIdx.z);

                binSums +=  GetHistogramOffset(blockIdx.y, blockIdx.z) * histLineSize;
            } else {
                const ui64 leftPartOffset = GetDataPartitionOffset(blockIdx.y, blockIdx.z);
                const ui64 rightPartOffset = GetDataPartitionOffset(gridDim.y | blockIdx.y, blockIdx.z);
                const int leftPartSize = partition[leftPartOffset].Size;
                const int rightPartSize = partition[rightPartOffset].Size;

                partition += (leftPartSize < rightPartSize) ? leftPartOffset : rightPartOffset;
                binSums += histLineSize * GetHistogramOffset(gridDim.y | blockIdx.y, blockIdx.z);
            }
        }
    };


    //reduce histograms in leaf
    //we need this only for non-binary features
    template <int BLOCK_SIZE, int HIST_COUNT>
    __global__ void ScanHistogramsImpl(const TCFeature* feature,
                                       const int featureCount,
                                       const int histLineSize,
                                       float* histogram)
    {

        __shared__ float sums[BLOCK_SIZE * HIST_COUNT];
        const int featuresPerBlock = BLOCK_SIZE / 32;

        const int partId = TPointwisePartOffsetsHelper(gridDim.z).GetHistogramOffset(blockIdx.y, blockIdx.z);
        const int fold = threadIdx.x & 31;
        const int featureId = blockIdx.x * featuresPerBlock + threadIdx.x / 32;

        if (featureId < featureCount) {
            feature += featureId;
            if (!feature->OneHotFeature) {

                histogram += (partId * histLineSize + feature->FirstFoldIndex) * HIST_COUNT;
                const int folds = feature->Folds;
                volatile float* featureBinSums = sums + (threadIdx.x / 32) * 32;
                float sum[HIST_COUNT];

                for (int i = 0; i < HIST_COUNT; ++i) {
                    sum[i] = 0;
                };
                const int n = ((folds + 31) / 32) * 32;


                for (int binOffset = 0; binOffset < n; binOffset += 32) {

                    {
    #pragma unroll
                        for (int histId = 0; histId < HIST_COUNT; ++histId) {
                            featureBinSums[fold + histId * BLOCK_SIZE] = (binOffset + fold) < folds
                                                                         ? histogram[(binOffset + fold) * HIST_COUNT + histId]
                                                                         : 0.0f;
                        }
                    }

                    __syncthreads();
    #pragma unroll
                    for (int histId = 0; histId < HIST_COUNT; ++histId) {
                        InclusiveScanInWarp(featureBinSums + BLOCK_SIZE * histId, fold);
                    }

                    __syncthreads();
    #pragma unroll
                    for (int histId = 0; histId < HIST_COUNT; ++histId) {
                        sum[histId] += featureBinSums[fold + histId * BLOCK_SIZE];
                    }

                    __syncthreads();
                    if ((binOffset + fold) < folds) {
                        for (int histId = 0; histId < HIST_COUNT; ++histId) {
                            histogram[(binOffset + fold) * HIST_COUNT + histId] = sum[histId];
                        }
                    }
                    __syncthreads();

                    if ((binOffset + 32) < n) {
                        for (int histId = 0; histId < HIST_COUNT; ++histId) {
                            featureBinSums[fold + BLOCK_SIZE * histId] = sum[histId];
                            __syncwarp();
                            sum[histId] = featureBinSums[31 + BLOCK_SIZE * histId];
                        }
                    }
                    __syncthreads();
                }
            }
        }
    };

    // converts block indices in matrix to linear part index
    __forceinline__ __device__ int ConvertBlockToPart(int x, int y) {
        int partNo = 0;
        partNo |= (x & 1) | ((y & 1) << 1);
        partNo |= ((x & 2) << 1)   | ((y & 2) << 2);
        partNo |= ((x & 4) << 2)   | ((y & 4) << 3);
        partNo |= ((x & 8) << 3)   | ((y & 8) << 4);
        partNo |= ((x & 16) << 4)  | ((y & 16) << 5);
        partNo |= ((x & 32) << 5)  | ((y & 32) << 6);
        partNo |= ((x & 64) << 6)  | ((y & 64) << 7);
        partNo |= ((x & 128) << 7) | ((y & 128) << 8);
        return partNo;
    }

    __forceinline__ __device__ int GetPairwisePartIdToCalculate(const TDataPartition* partition) {

        const int depth = (int)log2((float)gridDim.y);
        //
        int partIds[4];
        int partSizes[4];

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int partId = (i << depth) | blockIdx.y;
            partIds[i] = partId;
            partSizes[i] = partition[partId].Size;
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = i + 1; j < 4; ++j) {
                if (partSizes[j] > partSizes[i]) {
                    const int tmpSize = partSizes[j];
                    const int tmpId = partIds[j];

                    partSizes[j] = partSizes[i];
                    partIds[j] = partIds[i];

                    partSizes[i] = tmpSize;
                    partIds[i] = tmpId;
                }
            }
        }
        return partIds[blockIdx.z + 1];
    }


    struct TCmpBinsWithoutOneHot {

        __forceinline__ TCmpBinsWithoutOneHot() = default;
        __forceinline__ __device__ TCmpBinsWithoutOneHot (const TCFeature*,
                                                          int) {

        }

        __forceinline__ __device__ bool Compare(int, int bin1, int bin2, bool flag) {
            return bin1 >= bin2 == flag;
        }
    };


    //N should be power of two
    template <int N>
    struct TCmpBinsWithOneHot {
        bool IsOneHot[N];

        __forceinline__ __device__ TCmpBinsWithOneHot(const TCFeature* features,
                                                      int fCount) {

            for (int i = 0; i < N; ++i) {
                const int f = ((threadIdx.x / 2) + i) & (N - 1);
                IsOneHot[i] = f < fCount ? features[f].OneHotFeature : false;
            }
        }

        __forceinline__ __device__ bool Compare(int i, int bin1, int bin2, bool flag) {
            return  IsOneHot[i] ? bin1 == bin2 : bin1 >= bin2 == flag;
        }
    };



    template <bool WithOneHot>
    struct TCmpBinsOneByteTrait;

    template <>
    struct TCmpBinsOneByteTrait<false> {
        using TCmpBins = TCmpBinsWithoutOneHot;
    };

    template <>
    struct TCmpBinsOneByteTrait<true> {
        using TCmpBins = TCmpBinsWithOneHot<4>;

    };

    enum class ELoadType {
        OneElement,
        TwoElements,
        FourElements
    };


    inline bool HasOneHotFeatures(const TCFeature* features, int fCount) {
        for (int i = 0; i < fCount; ++i) {
            if (features[i].OneHotFeature) {
                return true;
            }
        }
        return false;
    }

}
