#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"
#include "compute_point_hist2_loop.cuh"
#include "pointwise_hist2_half_byte_template.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{


    template <int BlockSize, bool IsFullPass, int M>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesHalfByteImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount) {


        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, IsFullPass);

        feature += (blockIdx.x / M) * 8;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 8, 8);

//
        __shared__ float smem[16 * BlockSize];


        using THist = TPointHistHalfByte<BlockSize>;


        #if __CUDA_ARCH__ > 350
            const bool use64BitLoad = IsFullPass;
        #else
            const bool use64BitLoad = false;
        #endif

        if (use64BitLoad)
        {
            #if __CUDA_ARCH__ <= 350
            const int OUTER_UNROLL = 2;
            #else
            const int OUTER_UNROLL = 1;
            #endif
            ComputeHistogram2 < BlockSize, OUTER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, smem);
        } else {
            #if __CUDA_ARCH__ <= 300
            const int INNER_UNROLL = 2;
            const int OUTER_UNROLL = 2;
            #elif __CUDA_ARCH__ <= 350
            const int INNER_UNROLL = 4;
            const int OUTER_UNROLL = 2;
            #else
            const int INNER_UNROLL = 1;
            const int OUTER_UNROLL = 1;
            #endif

            ComputeHistogram < BlockSize, OUTER_UNROLL, INNER_UNROLL, 1, M, THist > (
                    indices, partition->Offset, partition->Size,
                            target, weight,
                            cindex, smem);
        }

        __syncthreads();

        const int fid = (threadIdx.x / 32);
        const int fold = (threadIdx.x / 2) & 15;
        const int w = threadIdx.x & 1;

        if (fid < fCount && fold < feature[fid].Folds) {
            const float result = smem[fold * 16 + 2 * fid + w];
            if (abs(result) > 1e-20) {
                if (M > 1) {
                    atomicAdd(binSums + (feature[fid].FirstFoldIndex + fold) * 2 + w, result);
                } else {
                    binSums[(feature[fid].FirstFoldIndex + fold) * 2 + w] = result;
                }
            }
        }
    }


    template <int BlockSize,
             int BlocksPerFeatureCount>
    inline void RunComputeHist2HalfByteKernel(const TCFeature* nbFeatures, int nbCount,
                                              const ui32* cindex,
                                              const float* target, const float* weight, const ui32* indices,
                                              const TDataPartition* partition,
                                              float* binSums, const int binFeatureCount,
                                              bool fullPass,
                                              TCudaStream stream,
                                              dim3 numBlocks)
    {

        if (fullPass)
        {
            ComputeSplitPropertiesHalfByteImpl < BlockSize, true,
                    BlocksPerFeatureCount > << <numBlocks, BlockSize, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );

        } else
        {
            ComputeSplitPropertiesHalfByteImpl < BlockSize, false,
                    BlocksPerFeatureCount > << <numBlocks, BlockSize, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount);
        }

    }

    void ComputeHist2HalfByte(const TCFeature* halfByteFeatures, ui32 halfByteFeaturesCount,
                              const ui32* cindex,
                              const float* target, const float* weight, const ui32* indices,
                              ui32 size,
                              const TDataPartition* partition, ui32 partsCount, ui32 foldCount,
                              bool fullPass,
                              const ui32 histLineSize,
                              float* binSums,
                              TCudaStream stream) {
        dim3 numBlocks;
        numBlocks.x = static_cast<ui32>((halfByteFeaturesCount + 7) / 8);
        const int histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = static_cast<ui32>(histCount);
        numBlocks.z = foldCount;

        const int blockSize = 768;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
        numBlocks.x *= multiplier;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        if (halfByteFeaturesCount) {

            #define COMPUTE(k)\
            RunComputeHist2HalfByteKernel<blockSize, k>(halfByteFeatures, halfByteFeaturesCount, cindex,\
                                                        target,\
                                                        weight, indices, partition, binSums, histLineSize,\
                                                        fullPass,\
                                                        stream, numBlocks);

            if (multiplier == 1) {
                COMPUTE(1)
            } else if (multiplier == 2) {
                COMPUTE(2)
            } else if (multiplier == 4) {
                COMPUTE(4)
            } else if (multiplier == 8) {
                COMPUTE(8)
            } else if (multiplier == 16) {
                COMPUTE(16)
            } else if (multiplier == 32) {
                COMPUTE(32)
            } else if (multiplier == 64) {
                COMPUTE(64)
            } else {
                exit(1);
            }
            #undef COMPUTE
        }
    }

}
