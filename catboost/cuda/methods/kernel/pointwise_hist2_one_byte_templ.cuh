#pragma once
#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"
#include "compute_point_hist2_loop.cuh"
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>


namespace NKernel {


    template <int HIGHER_BITS, bool IS_FIRST_LEVEL>
    struct TLoadEntriesTrait;


    template <int BITS, ELoadType LoadType>
    struct TUnrollsTrait;

    template <int BITS>
    struct TUnrollsTrait<BITS, ELoadType::OneElement> {

        constexpr static int Inner() {
            #if __CUDA_ARCH__ <= 350
            return BITS == 0 ? 8 : 4;
            #else
            return 1;
            #endif
        }

        constexpr static int Outer() {
            return 2;
        }
    };

    template <int BITS>
    struct TUnrollsTrait<BITS, ELoadType::TwoElements> {
        constexpr static int Outer() {
            #if __CUDA_ARCH__ < 700
            return BITS < 3 ? 4 : 2;
            #else
            return 1;
            #endif
        }
    };

    template <int BITS>
    struct TUnrollsTrait<BITS, ELoadType::FourElements> {
        constexpr static int Outer() {
            return 1;
        }
    };


    template <int BITS>
    struct TDeclarePassInnerOuterBitsTrait;


    template <int OUTER_HIST_BITS_COUNT,
            int INNER_HIST_BITS_COUNT,
            int BLOCK_SIZE>
    struct TPointHist;

    template <int BLOCK_SIZE, int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int BLOCKS_PER_FEATURE, bool IS_FULL_PASS>
    __forceinline__ __device__ void ComputeSplitPropertiesPass(const TCFeature* __restrict__ feature, const ui32* __restrict__ cindex,
                                                               const float* __restrict__ target, const float* __restrict__ weight,
                                                               const ui32* __restrict__ indices,
                                                               const TDataPartition* __restrict__ partition, int fCount,
                                                               float* binSumsForPart,
                                                               float* smem) {

        using THist = TPointHist<OUTER_HIST_BITS_COUNT, INNER_HIST_BITS_COUNT, BLOCK_SIZE>;
        const int stripeSize = BLOCK_SIZE;
        const int histBlockCount = 1;

        const int size = partition->Size;
        const int offset = partition->Offset;

        constexpr int Bits = OUTER_HIST_BITS_COUNT + INNER_HIST_BITS_COUNT;

        constexpr ELoadType type = TLoadEntriesTrait<Bits, IS_FULL_PASS>::LoadType();
        switch (type) {
            case ELoadType::OneElement: {
                constexpr int innerUnroll = TUnrollsTrait<Bits, ELoadType::OneElement>::Inner();
                constexpr int outerUnroll = TUnrollsTrait<Bits, ELoadType::OneElement>::Outer();
                ComputeHistogram<stripeSize, outerUnroll, innerUnroll, histBlockCount, BLOCKS_PER_FEATURE, THist>(indices,
                                                                                                                  partition->Offset,
                                                                                                                  partition->Size,
                                                                                                                  target,
                                                                                                                  weight,
                                                                                                                  cindex,
                                                                                                                  smem);
                break;
            }
            case ELoadType::TwoElements: {
                constexpr int outerUnroll = TUnrollsTrait<Bits, ELoadType::TwoElements>::Outer();

                ComputeHistogram2 < stripeSize, outerUnroll,  histBlockCount, BLOCKS_PER_FEATURE, THist > (indices,
                                                                                                            offset,
                                                                                                            size,
                                                                                                            target,
                                                                                                            weight,
                                                                                                            cindex,
                                                                                                            smem);
                break;
            }
            case ELoadType::FourElements: {
                constexpr int outerUnroll = TUnrollsTrait<Bits, ELoadType::FourElements>::Outer();
                ComputeHistogram4 < stripeSize, outerUnroll,  histBlockCount, BLOCKS_PER_FEATURE, THist > (indices,
                                                                                                           offset,
                                                                                                           size,
                                                                                                           target,
                                                                                                           weight,
                                                                                                           cindex,
                                                                                                           smem);
                break;

            }
        }

        __syncthreads();

        const int maxFoldCount = (1 << (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT));

        const int fid = (threadIdx.x / 64);
        const int w = threadIdx.x & 1;

        const int featureFolds =  fid < fCount ? feature[fid].Folds : 0;
        const int featureOffset = fid * maxFoldCount * 2 + w;

        for (int fold = (threadIdx.x / 2) & 31; fold < featureFolds; fold += 32) {

            if (fid < fCount) {
                const float val = smem[featureOffset + 2 * fold];

                if (abs(val) > 1e-20f) {
                    if (BLOCKS_PER_FEATURE > 1) {
                        atomicAdd(binSumsForPart + (feature[fid].FirstFoldIndex + fold) * 2 + w, val);
                    } else {
                        WriteThrough(binSumsForPart + (feature[fid].FirstFoldIndex + fold) * 2 + w, val);
                    }
                }
            }
        }
    }


#define DECLARE_PASS(O, I, M, FULL_PASS) \
    ComputeSplitPropertiesPass<BLOCK_SIZE, O, I, M, FULL_PASS>(feature, cindex, target, weight, indices, partition, fCount, binSums, &counters[0]);


    template <int BLOCK_SIZE, int BITS, bool FULL_PASS, int M>
    #if __CUDA_ARCH__ <= 350
    __launch_bounds__(BLOCK_SIZE, 1)
    #endif
    __global__ void ComputeSplitPropertiesNBImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount) {
        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 4;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 4, 4);

        __shared__ float counters[32 * BLOCK_SIZE];
        const ui32 maxBinCount = GetMaxBinCount(feature, fCount, (ui32*) &counters[0]);
        __syncthreads();

        static_assert(BITS >= 5, "Error: this specialization for 5-8 bit histograms");
        static_assert(BITS <= 8, "Error: this specialization for 5-8 bit histograms");

        constexpr ui32 upperBound = (1 << BITS);
        constexpr ui32 lowerBound = BITS > 5 ? upperBound / 2 : 15;

        if (maxBinCount <= lowerBound || maxBinCount > upperBound) {
            return;
        }

        if (partition->Size) {
            DECLARE_PASS(TDeclarePassInnerOuterBitsTrait<BITS - 5>::Outer(), TDeclarePassInnerOuterBitsTrait<BITS - 5>::Inner(), M, FULL_PASS);
        }
    }

    template <int BLOCK_SIZE,
             int Bits,
             int BLOCKS_PER_FEATURE_COUNT>
    inline void RunComputeHist2NonBinaryKernel(const TCFeature* nbFeatures, int nbCount,
                                               const ui32* cindex,
                                               const float* target, const float* weight, const ui32* indices,
                                               const TDataPartition* partition,
                                               float* binSums, const int binFeatureCount,
                                               bool fullPass,
                                               TCudaStream stream,
                                               dim3 numBlocks)
    {

        if (fullPass) {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, Bits, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );

        } else {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, Bits, false, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );
        }

    }




    template <int Bits>
    void ComputeHist2NonBinary(const TCFeature* nbFeatures, ui32 nbCount,
                               const ui32* cindex,
                               const float* target, const float* weight,
                               const ui32* indices, ui32 size,
                               const TDataPartition* partition, ui32 partCount, ui32 foldCount,
                               bool fullPass,
                               ui32 histLineSize,
                               float* binSums,
                               ui32 featureCountForBits,
                               TCudaStream stream) {
        if (featureCountForBits) {

            dim3 numBlocks;
            const int histPartCount = (fullPass ? partCount : partCount / 2);
            numBlocks.y = histPartCount;
            numBlocks.z = foldCount;
            const int blockSize = 384;

            numBlocks.x = (featureCountForBits + 3) / 4;
            const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
            numBlocks.x = ((nbCount + 3) / 4);
            numBlocks.x *= multiplier;
            if (IsGridEmpty(numBlocks)) {
                return;
            }

            #define COMPUTE(k)\
             RunComputeHist2NonBinaryKernel<blockSize, Bits, k>(nbFeatures, nbCount, cindex,  target, weight,  indices, \
                                                                partition, binSums, histLineSize, fullPass, stream, numBlocks);
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


    #define DEFINE_NON_BINARY_EXTERN(Bits) \
    extern template              \
    void ComputeHist2NonBinary<Bits>(const TCFeature* nbFeatures, ui32 nbCount, const ui32* cindex,\
                                     const float* target, const float* weight,\
                                     const ui32* indices, ui32 size,\
                                     const TDataPartition* partition, ui32 partCount, ui32 foldCount,\
                                     bool fullPass,\
                                     ui32 histLineSize,\
                                     float* binSums,    \
                                     ui32 bits,              \
                                     TCudaStream stream);

    #define DEFINE_NON_BINARY(Bits) \
    template              \
    void ComputeHist2NonBinary<Bits>(const TCFeature* nbFeatures, ui32 nbCount, const ui32* cindex,\
                                     const float* target, const float* weight,\
                                     const ui32* indices, ui32 size,\
                                     const TDataPartition* partition, ui32 partCount, ui32 foldCount,\
                                     bool fullPass,\
                                     ui32 histLineSize,\
                                     float* binSums,    \
                                     ui32,              \
                                     TCudaStream stream);

    DEFINE_NON_BINARY_EXTERN(5)
    DEFINE_NON_BINARY_EXTERN(6)
    DEFINE_NON_BINARY_EXTERN(7)
    DEFINE_NON_BINARY_EXTERN(8)

    #undef DEFINE_NON_BINARY_EXTERN
}
