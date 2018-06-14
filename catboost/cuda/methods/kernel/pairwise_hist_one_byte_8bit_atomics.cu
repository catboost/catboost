#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "compute_pair_hist_loop.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <cstdio>

namespace NKernel {

    template<bool IsFullPass>
    struct TEightBitPairwiseHistUnrollTrait {

        static constexpr int InnerUnroll() {
            return 1;
        }

        static constexpr int OuterUnroll() {
            return 2;
        }
    };

    template<int BlockSize, class TCmpBins = TCmpBinsWithoutOneHot>
    struct TEightBitHistogram {
        TCmpBins CmpBinsFunc;
        float* Histogram;

        uchar CachedBinsLeq[8];
        uchar CachedBinsGe[8];

        float CachedSumsLeq[8];
        float CachedSumsGe[8];


        __forceinline__ __device__ int SliceOffset() {
            //2 blocks if INNER_HIST_BITS_COUNT = 0, else 1
            // x4 feature and x4 histograms, though histStart = blockIdx * 16
            return (threadIdx.x & 16);
        }


        __forceinline__  __device__ TEightBitHistogram(float* buff, TCmpBins cmpBins)
                : CmpBinsFunc(cmpBins) {
            Histogram = buff;
            for (int i = threadIdx.x; i < 256 * 32; i += BlockSize) {
                Histogram[i] = 0;
            }
            Histogram += SliceOffset();

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                CachedBinsLeq[i] = 0;
                CachedBinsGe[i] = 0;
                CachedSumsLeq[i] = 0;
                CachedSumsGe[i] = 0;
            }
            __syncthreads();
        }


        __forceinline__ __device__ void AddPair(const ui32 ci1,
                                                const ui32 ci2,
                                                const float w) {
            const bool flag = threadIdx.x & 1;

            const int shift = 4 * (threadIdx.x & 6);
            const ui32 bins1 = RotateRight(flag ? ci2 : ci1, shift);
            const ui32 bins2 = RotateRight(flag ? ci1 : ci2, shift);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = (threadIdx.x + 2 * i) & 6;
                int bin1 = (bins1 >> (24 - 8 * i)) & 255;
                int bin2 = (bins2 >> (24 - 8 * i)) & 255;

                //for one hot is't note leq
                const bool isLeqHist = CmpBinsFunc.Compare(i, bin1, bin2, flag);

                const bool needDrop1 = isLeqHist ? CachedBinsLeq[i] != bin1 : CachedBinsGe[i] != bin1;
                const bool needDrop2 = isLeqHist ? CachedBinsLeq[i + 4] != bin2 : CachedBinsGe[i + 4] != bin2;


                if (needDrop1) {
                    int offset1 = f + (isLeqHist ? 0 : 8)  + ((isLeqHist ? CachedBinsLeq[i] : CachedBinsGe[i]) * 32) + flag;
                    float toAdd = isLeqHist ? CachedSumsLeq[i] : CachedSumsGe[i];
                    atomicAdd(Histogram + offset1, toAdd);

                    if (isLeqHist) {
                        CachedBinsLeq[i] = static_cast<uchar>(bin1);
                        CachedSumsLeq[i] = 0.0f;
                    } else {
                        CachedBinsGe[i] = static_cast<uchar>(bin1);
                        CachedSumsGe[i] = 0.0f;
                    }
                }

                float tmp =  isLeqHist ? CachedSumsLeq[i] : CachedSumsGe[i];
                tmp += w;

                if (isLeqHist) {
                    CachedSumsLeq[i] = tmp;
                } else {
                    CachedSumsGe[i] = tmp;
                }

                if (needDrop2) {
                    int offset2 = f + (isLeqHist ? 0 : 8) + ((isLeqHist ? CachedBinsLeq[i + 4] : CachedBinsGe[i + 4]) * 32) + !flag;
                    float toAdd = isLeqHist ? CachedSumsLeq[i + 4] : CachedSumsGe[i + 4];
                    atomicAdd(Histogram + offset2, toAdd);

                    if (isLeqHist) {
                        CachedBinsLeq[i + 4] = static_cast<uchar>(bin2);
                        CachedSumsLeq[i + 4] = 0.0f;
                    } else {
                        CachedBinsGe[i + 4] = static_cast<uchar>(bin2);
                        CachedSumsGe[i + 4] = 0.0f;
                    }
                }

                tmp = isLeqHist ? CachedSumsLeq[4 + i] : CachedSumsGe[4 + i];
                tmp += w;

                if (isLeqHist) {
                    CachedSumsLeq[4 + i] = tmp;
                } else {
                    CachedSumsGe[4 + i] = tmp;
                }
            }
        }

        template <int N>
        __forceinline__ __device__ void AddPairs(const ui32* ci1,
                                                 const ui32* ci2,
                                                 const float* w) {
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                AddPair(ci1[k], ci2[k], w[k]);
            }
        }

        __forceinline__ __device__  void Reduce() {

            const bool flag = threadIdx.x & 1;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int f = (threadIdx.x + 2 * i) & 6;
                {
                    int offset10 = f + ((CachedBinsLeq[i]) * 32) + flag;
                    int offset11 = f + 8 + (CachedBinsGe[i] * 32) + flag;
                    atomicAdd(Histogram + offset10, CachedSumsLeq[i]);
                    atomicAdd(Histogram + offset11, CachedSumsGe[i]);
                }
                {
                    int offset10 = f + ((CachedBinsLeq[i + 4]) * 32) + !flag;
                    int offset11 = f + 8 + (CachedBinsGe[i + 4] * 32) + !flag;
                    atomicAdd(Histogram + offset10, CachedSumsLeq[i + 4]);
                    atomicAdd(Histogram + offset11, CachedSumsGe[i + 4]);
                }
            }

            Histogram -= SliceOffset();
            __syncthreads();
        }
    };


    template<int BlockSize, bool IsFullPass, int M>
    #if __CUDA_ARCH__ <= 350
    __launch_bounds__(BlockSize, 1)
    #else
    __launch_bounds__(BlockSize)
    #endif
    __global__ void ComputeSplitPropertiesNonBinaryPairs8Bit(const TCFeature* feature, int fCount, const ui32* cindex,
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

        if (maxBinCount <= 128) {
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
        const bool needOneHot = HasOneHotFeatures(feature, fCount, (int*)&localHist[0]);


        constexpr int histBlockCount = 1;
        constexpr int innerUnroll = TEightBitPairwiseHistUnrollTrait<IsFullPass>::InnerUnroll();
        constexpr int outerUnroll = TEightBitPairwiseHistUnrollTrait<IsFullPass>::OuterUnroll();

        #define DECLARE_PASS(TBinCmp)   \
        {                                 \
            using THist = TEightBitHistogram<BlockSize, TBinCmp>; \
            TBinCmp cmp(feature, fCount);\
            THist hist(&localHist[0], cmp);\
            ComputePairHistogram< BlockSize, histBlockCount, innerUnroll, outerUnroll, M, THist>(partition->Offset, cindex, partition->Size, pairs, weight, hist);\
        }

        if (needOneHot) {
            DECLARE_PASS(TCmpBinsWithOneHot<4>)
        } else {
            DECLARE_PASS(TCmpBinsWithoutOneHot)
        }
        #undef DECLARE_PASS

        if (threadIdx.x < 256) {
            const int histId = threadIdx.x & 3;
            const int binId = (threadIdx.x >> 2) & 15;
            const int fid = (threadIdx.x >> 6) & 3;

            if (fid < fCount) {
                const ui32 bfStart = feature[fid].FirstFoldIndex;
                histogram += 4 * bfStart;
                const int leqOrGeqShift = histId < 2 ? 0 : 8;
                const int isSecond = (histId & 1);

                for (int fold = binId; fold < feature[fid].Folds; fold += 16) {
                    const int readOffset = 32 * fold
                                           + 2 * fid
                                           + isSecond
                                           + leqOrGeqShift;

                    const float result = localHist[readOffset] + localHist[readOffset + 16];
                    if (M > 1) {
                        atomicAdd(histogram + 4 * fold + histId, result);
                    } else {
                        histogram[4 * fold + histId] += result;
                    }
                }
            }
        }
    }

    inline ui32 EstimateBlockPerFeatureMultiplier8Bit(dim3 numBlocks, ui32 dsSize, int limit = 64) {
        int blocksPerSm = TArchProps::GetMajorVersion() < 5 ? 1 : 3;
        ui32 multiplier = 1;
        while ((numBlocks.x * numBlocks.y * min(numBlocks.z, 8) * multiplier < TArchProps::SMCount() * blocksPerSm * 1.25) &&
               ((dsSize / multiplier) > 10000) && (multiplier < limit)) {
            multiplier *= 2;
        }
        return multiplier;
    }

    void ComputePairwiseHistogramOneByte8BitAtomics(const TCFeature* features,
                                                    const ui32 featureCount,
                                                    const ui32 sixBitsFeatureCount,
                                                    const ui32* compressedIndex,
                                                    const uint2* pairs, ui32 pairCount,
                                                    const float* weight,
                                                    const TDataPartition* partition,
                                                    ui32 partCount,
                                                    ui32 histLineSize,
                                                    bool fullPass,
                                                    float* histogram,
                                                    TCudaStream stream) {

        if (sixBitsFeatureCount > 0) {
            const int blockSize = 256;
            dim3 numBlocks;
            numBlocks.x = (sixBitsFeatureCount+ 3) / 4;
            numBlocks.y = fullPass ? partCount : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;
            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier8Bit(numBlocks, pairCount, 64);
            numBlocks.x = (featureCount + 3) / 4;
            numBlocks.x *= blockPerFeatureMultiplier;



            #define NB_HIST(IS_FULL, BLOCKS_PER_FEATURE)   \
            ComputeSplitPropertiesNonBinaryPairs8Bit < blockSize, IS_FULL, BLOCKS_PER_FEATURE > << <numBlocks, blockSize, 0, stream>>>(\
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
