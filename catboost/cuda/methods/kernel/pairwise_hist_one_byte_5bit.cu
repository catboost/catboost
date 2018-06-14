#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "compute_pair_hist_loop.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <cstdio>


using namespace cooperative_groups;

namespace NKernel {

    template<bool IsFullPass>
    struct TFiveBitPairwiseHistUnrollTrait {

        static constexpr int InnerUnroll() {
            #if __CUDA_ARCH__ <= 350
            return 4;
            #elif __CUDA_ARCH__ < 700
            return 2;
            #else
            return IsFullPass ? 4 : 8;
            #endif
        }

        static constexpr int OuterUnroll() {
            #if __CUDA_ARCH__ <= 350
            return 2;
            #elif __CUDA_ARCH__ < 700
            return 2;
            #else
            return 1;
            #endif
        }
    };



    template<int BlockSize, bool NeedLastBinMask /*is 32 histogram */, class TCmpBins = TCmpBinsWithoutOneHot>
    struct TFiveBitHistogram {
        TCmpBins CmpBinsFunc;
        float* Histogram;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            //2 blocks if INNER_HIST_BITS_COUNT = 0, else 1
            // x4 feature and x4 histograms, though histStart = blockIdx * 16
            return warpOffset +  (threadIdx.x & 16);
        }


        __forceinline__  __device__ TFiveBitHistogram(float* buff, TCmpBins cmpBins)
        : CmpBinsFunc(cmpBins) {
            Histogram = buff;
            CmpBinsFunc = cmpBins;

            for (int i = threadIdx.x; i < BlockSize * 32; i += BlockSize) {
                Histogram[i] = 0;
            }
            Histogram += SliceOffset();
            __syncthreads();
        }



        __forceinline__ __device__ void AddPair(const ui32 ci1,
                                                const ui32 ci2,
                                                const float w) {
            thread_block_tile<16> groupTile = tiled_partition<16>(this_thread_block());

            const bool flag = threadIdx.x & 1;

            const int shift = 4 * (threadIdx.x & 6);
            const ui32 bins1 = RotateRight(flag ? ci2 : ci1, shift);
            const ui32 bins2 = RotateRight(flag ? ci1 : ci2, shift);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = (threadIdx.x + 2 * i) & 6;
                int bin1 = (bins1 >> (24 - 8 * i)) & 255;
                int bin2 = (bins2 >> (24 - 8 * i)) & 255;


                const float w1 = (!NeedLastBinMask || bin1 < 32) ? w : 0;
                const float w2 = (!NeedLastBinMask || bin2 < 32) ? w : 0;

                const int tmp = (CmpBinsFunc.Compare(i, bin1, bin2, flag) ? 0 : 8) + f;

                int offset1 = tmp + ((bin1 & 31) << 5) + flag;
                int offset2 = tmp + ((bin2 & 31) << 5) + !flag;


                groupTile.sync();

                if (groupTile.thread_rank() < 8) {
                    Histogram[offset1] += w1;
                }

                groupTile.sync();

                if (groupTile.thread_rank() >= 8) {
                    Histogram[offset1] += w1;
                }

                groupTile.sync();

                if (groupTile.thread_rank() < 8) {
                    Histogram[offset2] += w2;
                }

                groupTile.sync();

                if (groupTile.thread_rank() >= 8) {
                    Histogram[offset2] += w2;
                }
            }
        }


        #if __CUDA_ARCH__ < 700
        template <int N>
        __forceinline__ __device__ void AddPairs(const ui32* ci1,
                                                 const ui32* ci2,
                                                 const float* w) {
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                AddPair(ci1[k], ci2[k], w[k]);
            }
        }
        #else
        template <int N>
        __forceinline__ __device__ void AddPairs(const ui32* ci1,
                                                 const ui32* ci2,
                                                 const float* w) {
            thread_block_tile<16> groupTile = tiled_partition<16>(this_thread_block());

            const bool flag = threadIdx.x & 1;
            const int shift = 4 * (threadIdx.x & 6);

            ui32 bins1[N];
            ui32 bins2[N];

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = (threadIdx.x + 2 * i) & 6;

                int bin1[N];
                int bin2[N];

                float w1[N];
                float w2[N];

                int offset1[N];
                int offset2[N];

                #pragma unroll
                for (int k = 0; k < N;++k) {
                    if (i == 0) {
                        bins1[k] = RotateRight(flag ? ci2[k] : ci1[k], shift);
                        bins2[k] = RotateRight(flag ? ci1[k] : ci2[k], shift);
                    }
                    bin1[k] = (bins1[k] >> (24 - 8 * i)) & 255;
                    bin2[k] = (bins2[k] >> (24 - 8 * i)) & 255;

                    w1[k] = (!NeedLastBinMask || bin1[k] < 32) ? w[k] : 0;
                    w2[k] = (!NeedLastBinMask || bin2[k] < 32) ? w[k] : 0;

                    const int tmp = (CmpBinsFunc.Compare(i, bin1[k], bin2[k], flag)  ? 0 : 8) + f;
                    offset1[k] = tmp + ((bin1[k] & 31) * 32) + flag;
                    offset2[k] = tmp + ((bin2[k] & 31) * 32) + !flag;
                }


                groupTile.sync();

                if (groupTile.thread_rank() < 8) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset1[k]] += w1[k];
                    }
                }

                groupTile.sync();

                if (groupTile.thread_rank() >= 8) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset1[k]] += w1[k];
                    }
                }

                groupTile.sync();

                if (groupTile.thread_rank() < 8) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset2[k]] += w2[k];
                    }
                }

                groupTile.sync();

                if (groupTile.thread_rank() >= 8) {
                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        Histogram[offset2[k]] += w2[k];
                    }
                }
            }
        }
        #endif

        __forceinline__ __device__  void Reduce() {
            Histogram -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BlockSize) {
                    float sum = 0;

                    #pragma unroll 12
                    for (int i = start; i < 32 * BlockSize; i += warpHistSize) {
                        sum += Histogram[i];
                    }

                    Histogram[warpHistSize + start] = sum;
                }
            }
            __syncthreads();

            const int maxFoldCount = 32;
            const int fold = (threadIdx.x >> 1) & 31;
            const int f = threadIdx.x / 64;


            if (threadIdx.x < 256) {
                float weightLeq = 0;
                float weightGe = 0;
                const bool isSecondBin = (threadIdx.x & 1);

                if (fold < maxFoldCount) {
                    const volatile float* __restrict__ src = Histogram
                                                             + 1024  //warpHistSize
                                                             + 32 * fold
                                                             + 2 * f
                                                             + isSecondBin;

                    weightLeq = src[0] + src[16];
                    weightGe = src[8] + src[24];

                    Histogram[4 * (maxFoldCount * f + fold) + isSecondBin] = weightLeq;
                    Histogram[4 * (maxFoldCount * f + fold) + 2 + isSecondBin] = weightGe;
                }
            }

            __syncthreads();
        }
    };


    template<int BlockSize, bool IsFullPass, int M>
    #if __CUDA_ARCH__ <= 350
    __launch_bounds__(BlockSize, 1)
    #elif __CUDA_ARCH__ < 700
    __launch_bounds__(BlockSize, 2)
    #endif
    __global__ void ComputeSplitPropertiesNonBinaryPairs(const TCFeature* feature, int fCount, const ui32* cindex,
                                                         const uint2* pairs, const float* weight,
                                                         const TDataPartition* partition,
                                                         int histLineSize,
                                                         float* histogram) {

        const int featureOffset = (blockIdx.x / M) * 4;
        feature += featureOffset;
        cindex += feature->Offset;
        fCount = min(fCount - featureOffset, 4);


        __shared__ float localHist[32 * BlockSize];

        const int maxBinCount = GetMaxBinCount(feature, fCount, (int*) &localHist[0]);

        if (maxBinCount > 32) {
            return;
        }
        __syncthreads();


        if (IsFullPass) {
            partition += blockIdx.y;
            histogram += blockIdx.y * histLineSize * 4ULL;
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partition += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * histLineSize * 4ULL;
        }

        if (partition->Size == 0) {
            return;
        }


        constexpr int histBlockCount = 1;
        constexpr int innerUnroll = TFiveBitPairwiseHistUnrollTrait<IsFullPass>::InnerUnroll();
        constexpr int outerUnroll = TFiveBitPairwiseHistUnrollTrait<IsFullPass>::OuterUnroll();

        const bool needOneHot = HasOneHotFeatures(feature, fCount, (int*)&localHist[0]);

        #define DECLARE_PASS(NEED_MASK, TBinCmp)   \
        {                                   \
            using THist = TFiveBitHistogram<BlockSize, NEED_MASK, TBinCmp>;\
            TBinCmp cmp(feature, fCount);\
            THist hist(&localHist[0], cmp);\
            ComputePairHistogram< BlockSize, histBlockCount, innerUnroll, outerUnroll, M, THist>(partition->Offset, cindex, partition->Size, pairs, weight, hist);\
        }
        if (maxBinCount < 32) {
            if (needOneHot) {
                DECLARE_PASS(false, TCmpBinsWithOneHot<4>);
            } else {
                DECLARE_PASS(false, TCmpBinsWithoutOneHot);
            }
        } else {
            if (needOneHot) {
                DECLARE_PASS(true, TCmpBinsWithOneHot<4>);
            } else {
                DECLARE_PASS(true, TCmpBinsWithoutOneHot);
            }
        }
        #undef DECLARE_PASS

        if (threadIdx.x < 256) {
            const int histId = threadIdx.x & 3;
            const int binId = (threadIdx.x >> 2) & 15;
            const int fid = (threadIdx.x >> 6) & 3;
            const int maxFoldCount = 1 << 5;

            if (fid < fCount) {
                const ui32 bfStart = feature[fid].FirstFoldIndex;
                histogram += 4 * bfStart;

                for (int fold = binId; fold < feature[fid].Folds; fold += 16) {
                    const int readOffset = 4 * (maxFoldCount * fid + fold) + histId;
                    if (M > 1) {
                        atomicAdd(histogram + 4 * fold + histId, localHist[readOffset]);
                    } else {
                        histogram[4 * fold + histId] += localHist[readOffset];
                    }
                }
            }
        }
    }



    void ComputePairwiseHistogramOneByte5Bits(const TCFeature* features,
                                              const ui32 featureCount,
                                              const ui32 fiveBitsFeatureCount,
                                              const ui32* compressedIndex,
                                              const uint2* pairs, ui32 pairCount,
                                              const float* weight,
                                              const TDataPartition* partition,
                                              ui32 partCount,
                                              ui32 histLineSize,
                                              bool fullPass,
                                              float* histogram,
                                              TCudaStream stream) {

        if (fiveBitsFeatureCount > 0) {
            const int blockSize = 384;
            dim3 numBlocks;
            numBlocks.x = (fiveBitsFeatureCount+ 3) / 4;
            numBlocks.y = fullPass ? partCount : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;
            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier(numBlocks, pairCount, 64);
            numBlocks.x = (featureCount + 3) / 4;
            numBlocks.x *= blockPerFeatureMultiplier;



            #define NB_HIST(IS_FULL, BLOCKS_PER_FEATURE)   \
            ComputeSplitPropertiesNonBinaryPairs < blockSize, IS_FULL, BLOCKS_PER_FEATURE > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition,  histLineSize, histogram);

            #define DISPATCH(BLOCKS_PER_FEATURE)  \
            if (fullPass) {                       \
                NB_HIST(true, BLOCKS_PER_FEATURE) \
            } else {                              \
                NB_HIST(false, BLOCKS_PER_FEATURE)\
            }


            if (blockPerFeatureMultiplier == 1) {
                DISPATCH(1);
            } else if (blockPerFeatureMultiplier == 2) {
                DISPATCH(2);
            } else if (blockPerFeatureMultiplier == 4) {
                DISPATCH(4);
            } else if (blockPerFeatureMultiplier == 8) {
                DISPATCH(8);
            } else if (blockPerFeatureMultiplier == 16) {
                DISPATCH(16);
            } else if (blockPerFeatureMultiplier == 32) {
                DISPATCH(32);
            } else if (blockPerFeatureMultiplier == 64) {
                DISPATCH(64);
            } else {
                exit(0);
            }
            #undef NB_HIST
            #undef DISPATCH
        }
    }
}
