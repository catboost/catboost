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
    __global__ void ComputeSplitPropertiesBImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__  partition, float* __restrict__ binSums, int totalFeatureCount) {

        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, IsFullPass);

        feature += (blockIdx.x / M) * 32;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 32, 32);

        __shared__ float counters[16 *BlockSize];

        if (partition->Size)
        {
            using THist = TPointHistHalfByte<BlockSize>;
            #if __CUDA_ARCH__ > 350
            const bool use64bitLoad = IsFullPass;
            #else
            const bool use64bitLoad = false;
            #endif

            if (use64bitLoad) {
                //full pass
                #if __CUDA_ARCH__ <= 350
                const int OUTER_UNROLL = 1;
                #else
                const int OUTER_UNROLL = 1;
                #endif
                ComputeHistogram2 <BlockSize, OUTER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, &counters[0]);
            } else {
                #if __CUDA_ARCH__ <= 300
                const int INNER_UNROLL = 2;
                const int OUTER_UNROLL = 1;
                #elif __CUDA_ARCH__ <= 350
                const int INNER_UNROLL = 4;
                const int OUTER_UNROLL = 1;
                #else
                const int INNER_UNROLL = 1;
                const int OUTER_UNROLL = 1;
                #endif

                ComputeHistogram <BlockSize, OUTER_UNROLL, INNER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, &counters[0]);
            }

            ui32 w = threadIdx.x & 1;
            ui32 fid = (threadIdx.x >> 1);

            if (fid < fCount)
            {
                const int groupId = fid / 4;
                uchar fMask = 1 << (3 - (fid & 3));

                float sum = 0.f;
                #pragma uroll
                for (int i = 0; i < 16; i++) {
                    if (!(i & fMask)) {
                        sum += counters[i * 16 + 2 * groupId + w];
                    }
                }

                if (abs(sum) > 1e-20f) {
                    if (M > 1)
                    {
                        atomicAdd(binSums + (feature[fid].FirstFoldIndex) * 2 + w, sum);
                    } else
                    {
                        binSums[(feature[fid].FirstFoldIndex) * 2 + w] = sum;
                    }
                }
            }
        }
    }



    template <int BlockSize, int BlocksPerFeatureCount>
    void RunComputeHist2BinaryKernel(const TCFeature* bFeatures, int bCount,
                                     const ui32* cindex,
                                     const float* target, const float* weight, const ui32* indices,
                                     const TDataPartition* partition,
                                     float* binSums, bool fullPass,
                                     int totalFeatureCount,
                                     TCudaStream stream,
                                     dim3 numBlocks)
    {
        if (fullPass)
        {
            ComputeSplitPropertiesBImpl <BlockSize, true,
                    BlocksPerFeatureCount > << <numBlocks,BlockSize, 0, stream>>>(
                    bFeatures, bCount, cindex, target, weight, indices, partition, binSums, totalFeatureCount
            );
        } else
        {
            ComputeSplitPropertiesBImpl <BlockSize, false,
                    BlocksPerFeatureCount > << <numBlocks,BlockSize, 0, stream>>>(
                    bFeatures, bCount, cindex, target, weight, indices, partition, binSums, totalFeatureCount
            );
        }
    };



    void ComputeHist2Binary(const TCFeature* bFeatures,  ui32 bCount,
                            const ui32* cindex,
                            const float* target, const float* weight,
                            const ui32* indices, ui32 size,
                            const TDataPartition* partition,
                            ui32 partsCount, ui32 foldCount,
                            bool fullPass,
                            ui32 totalFeatureCount,
                            float* binSums,
                            TCudaStream stream) {
        dim3 numBlocks;
        numBlocks.x = (bCount + 31) / 32;
        const int histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = histCount;
        numBlocks.z = foldCount;

        const int blockSize = 768;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
        numBlocks.x *= multiplier;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        if (bCount) {

            #define COMPUTE(k)  \
            RunComputeHist2BinaryKernel<blockSize, k>(bFeatures, bCount, cindex, target, weight, indices, \
                                                      partition, binSums, fullPass, totalFeatureCount, stream, numBlocks); \

            if (multiplier == 1) {
                COMPUTE(1)
            } else if (multiplier == 2) {
                COMPUTE(2)
            } else if (multiplier == 4) {
                COMPUTE(4)
            } else if (multiplier == 8) {
                COMPUTE(8);
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
