#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/cuda_util/kernel/inplace_scan.cuh>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cassert>

namespace NKernel {

    __forceinline__ __device__ int GetMaxBinCount(const TCFeature* features, int fCount, int* smem) {

        int binCount = threadIdx.x < fCount ? features[threadIdx.x].Folds : 0;
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
        int result = smem[0];
        __syncthreads();

        return result;
    }


    struct TPartOffsetsHelper {
        ui32 FoldCount;

        __forceinline__ __device__ TPartOffsetsHelper(ui32 foldCount)
                : FoldCount(foldCount) {

        }
        __forceinline__ __device__ ui32 GetHistogramOffset(ui32 partId, ui32 foldId) {
            return partId * FoldCount + foldId;
        }

        __forceinline__ __device__ ui32 GetDataPartitionOffset(ui32 partId, ui32 foldId) {
            const int foldStripe = 1 << static_cast<ui32>(ceil(log2(static_cast<float>(FoldCount))));
            return partId * foldStripe + foldId;
        }


        __forceinline__ __device__ void ShiftPartAndBinSumsPtr(const TDataPartition* __restrict__& partition, float* __restrict__& binSums, ui32 totalFeatureCount, bool fullPass) {
            if (fullPass) {
                partition += GetDataPartitionOffset(blockIdx.y, blockIdx.z);
                binSums +=  GetHistogramOffset(blockIdx.y, blockIdx.z) * 2 * totalFeatureCount;
            } else
            {
                const ui64 leftPartOffset = GetDataPartitionOffset(blockIdx.y, blockIdx.z);
                const ui64 rightPartOffset = GetDataPartitionOffset(gridDim.y | blockIdx.y, blockIdx.z);
                const int leftPartSize = partition[leftPartOffset].Size;
                const int rightPartSize = partition[rightPartOffset].Size;

                partition += (leftPartSize < rightPartSize) ? leftPartOffset : rightPartOffset;
                binSums += 2 * totalFeatureCount * GetHistogramOffset(gridDim.y | blockIdx.y, blockIdx.z);
            }
        }


    };


    //reduce histograms in leaf
    //we need this only for non-binary features
    template<int BLOCK_SIZE, int HIST_COUNT>
    __global__ void ScanHistogramsImpl(const TCFeature* feature,
                                       const int featureCount,
                                       const int totalBinaryFeatureCount,
                                       float* histogram)
    {

        __shared__ float sums[BLOCK_SIZE * HIST_COUNT];
        const int featuresPerBlock = BLOCK_SIZE / 32;

        const int partId = TPartOffsetsHelper(gridDim.z).GetHistogramOffset(blockIdx.y, blockIdx.z);
        const int fold = threadIdx.x & 31;
        const int featureId = blockIdx.x * featuresPerBlock + threadIdx.x / 32;

        if (featureId < featureCount) {
            feature += featureId;
            if (!feature->OneHotFeature) {

                histogram += (partId * totalBinaryFeatureCount + feature->FirstFoldIndex) * HIST_COUNT;
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
                            sum[histId] = featureBinSums[31 + BLOCK_SIZE * histId];
                        }
                    }
                    __syncthreads();
                }
            }
        }
    };



    template <int HIST_COUNT>
    __global__ void UpdatePointwiseHistogramsImpl(float* histogram,
                                                  const int featuresCount,
                                                  const TDataPartition* parts) {

        TPartOffsetsHelper helper(gridDim.z);

        const int leftPartId = helper.GetDataPartitionOffset(blockIdx.y, blockIdx.z);
        const int rightPartId = helper.GetDataPartitionOffset(blockIdx.y | gridDim.y, blockIdx.z);
        const int binFeature = blockIdx.x * blockDim.x + threadIdx.x;

        if (binFeature < featuresCount) {
            const TDataPartition leftPart = parts[leftPartId];
            const TDataPartition rightPart = parts[rightPartId];

            const bool isLeftCalculated = leftPart.Size < rightPart.Size;


            const size_t leftOffset = HIST_COUNT * (helper.GetHistogramOffset(blockIdx.y, blockIdx.z) * featuresCount + binFeature);
            const size_t rightOffset = HIST_COUNT * (helper.GetHistogramOffset(blockIdx.y | gridDim.y, blockIdx.z) * featuresCount + binFeature);

            float calcVal[HIST_COUNT];
            float complementVal[HIST_COUNT];

#pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId)
            {
                calcVal[histId] = histogram[rightOffset + histId];
                complementVal[histId] = histogram[leftOffset + histId] - calcVal[histId];
            }

#pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId)
            {
                histogram[leftOffset + histId] = isLeftCalculated ? calcVal[histId] : complementVal[histId] ;
                histogram[rightOffset + histId] = isLeftCalculated ? complementVal[histId] : calcVal[histId];
            }
        }
    }

    inline bool UpdatePointwiseHistograms(float* histograms,
                                   int binFeatureCount,
                                   int partCount,
                                   int foldCount,
                                   int histCount,
                                   const TDataPartition* parts,
                                   TCudaStream stream
    ) {

        const int blockSize = 256;
        dim3 numBlocks;
        numBlocks.x = (binFeatureCount + blockSize - 1) / blockSize;
        numBlocks.y = partCount / 2;
        numBlocks.z = foldCount;

        if (histCount == 1) {
            UpdatePointwiseHistogramsImpl<1><<<numBlocks, blockSize, 0, stream>>>(histograms, binFeatureCount, parts);
        }
        else if (histCount == 2) {
            UpdatePointwiseHistogramsImpl<2><<<numBlocks, blockSize, 0, stream>>>(histograms, binFeatureCount, parts);
        } else {
            return false;
        }
        return true;
    }


    template <int BLOCK_SIZE>
    __forceinline__ __device__  float ComputeSum(const float* buffer, int count) {
        float sum = 0.f;
        const int tid = threadIdx.x;
#pragma unroll 16
        for (int i = tid; i < count; i += BLOCK_SIZE)
        {
            sum += buffer[i];
        }
        return sum;
    };


}
