#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "compute_pair_hist_loop.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <cstdio>


using namespace cooperative_groups;

namespace NKernel {

    template <bool IsFullPass>
    struct TSevenBitPairwiseHistUnrollTrait {

        static constexpr int InnerUnroll(bool isOneHot) {
            #if __CUDA_ARCH__ <= 350
            return isOneHot ? 2 : 4;
            #elif __CUDA_ARCH__ < 700
            return isOneHot ? 1 : 2;
            #else
            return isOneHot ? 10 : 16;
//            return 1;//IsFullPass ? 8 : 16;
            #endif
        }

        static constexpr int OuterUnroll(bool isOneHot) {
            #if __CUDA_ARCH__ <= 350
            return isOneHot ? 2 : 4;
            #elif __CUDA_ARCH__ < 700
            return isOneHot? 2 : 4;//IsFullPass ? 4 : 8;
            #else
            return 1;
            #endif
        }
    };

    template <int BlockSize, bool NeedLastBinMask /*is 32 histogram */, class TCmpBins = TCmpBinsWithoutOneHot>
    struct TSevenBitHistogram {
        TCmpBins CmpBins;
        float* Histogram;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 4096 * (threadIdx.x / 32);
            //2 blocks if INNER_HIST_BITS_COUNT = 0, else 1
            // x4 feature and x4 histograms, though histStart = blockIdx * 16
            return warpOffset + (threadIdx.x & 16);
        }


        __forceinline__  __device__ TSevenBitHistogram(float* buff, TCmpBins cmpBins)
        : CmpBins(cmpBins) {
            Histogram = buff;
            for (int i = threadIdx.x; i < BlockSize * 128; i += BlockSize) {
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


                const float w1 = (!NeedLastBinMask || bin1 < 128) ? w : 0;
                const float w2 = (!NeedLastBinMask || bin2 < 128) ? w : 0;

                const int tmp = (CmpBins.Compare(i, bin1, bin2, flag) ? 0 : 8) + f;

                int offset1 = tmp + ((bin1 & 127) << 5) + flag;
                int offset2 = tmp + ((bin2 & 127) << 5) + !flag;


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
        __forceinline__ __device__ void AddPairsImpl(const ui32* ci1,
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

                    w1[k] = (!NeedLastBinMask || bin1[k] < 128) ? w[k] : 0;
                    w2[k] = (!NeedLastBinMask || bin2[k] < 128) ? w[k] : 0;

                    const int tmp = (CmpBins.Compare(i, bin1[k], bin2[k], flag) ? 0 : 8) + f;
                    offset1[k] = tmp + ((bin1[k] & 127) * 32) + flag;
                    offset2[k] = tmp + ((bin2[k] & 127) * 32) + !flag;
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

          template <int N>
        __forceinline__ __device__ void AddPairs(const ui32* ci1,
                                                 const ui32* ci2,
                                                 const float* w) {
            const int NN = N;

            #pragma unroll
            for (int k = 0; k < N;  k += NN) {
                AddPairsImpl<NN>(ci1 + k, ci2 + k, w + k);
            }
            static_assert(N % NN == 0, "Error");
        }
        #endif

        __forceinline__ __device__  void Reduce() {
            Histogram -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 4 * 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BlockSize) {
                    float sum = 0;

                    #pragma unroll
                    for (int i = start; i < 128 * BlockSize; i += warpHistSize) {
                        sum += Histogram[i];
                    }

                    Histogram[start] = sum;
                }
            }
            __syncthreads();
        }
    };


    template <int BlockSize, bool IsFullPass, bool OneHotPass>
    #if __CUDA_ARCH__ <= 350
    __launch_bounds__(BlockSize, 1)
    #else
    __launch_bounds__(BlockSize)
    #endif
    __global__ void ComputeSplitPropertiesNonBinaryPairs7Bit(const TCFeature* feature, int fCount, const ui32* cindex,
                                                             const uint2* pairs, const float* weight,
                                                             const TDataPartition* partition,
                                                             int histLineSize,
                                                             float* histogram) {

        const int maxBlocksPerPart = gridDim.x / ((fCount + 3) / 4);

        const int featureOffset = (blockIdx.x / maxBlocksPerPart) * 4;
        feature += featureOffset;
        cindex += feature->Offset;
        fCount = min(fCount - featureOffset, 4);


        __shared__ float localHist[128 * BlockSize];

        const ui32 maxBinCount = GetMaxBinCount(feature, fCount, (ui32*) &localHist[0]);

        if (maxBinCount > 128  || (maxBinCount <= 64)) {
            return;
        }

        const bool needOneHot = HasOneHotFeatures(feature, fCount, (int*)&localHist[0]);
        if (needOneHot != OneHotPass) {
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


        constexpr int innerUnroll = TSevenBitPairwiseHistUnrollTrait<IsFullPass>::InnerUnroll(OneHotPass);
        constexpr int outerUnroll = TSevenBitPairwiseHistUnrollTrait<IsFullPass>::OuterUnroll(OneHotPass);

        const int localBlockIdx = blockIdx.x % maxBlocksPerPart;
        const int minDocsPerBlock = BlockSize * innerUnroll * 16;
        const int activeBlockCount = min((partition->Size + minDocsPerBlock - 1) / minDocsPerBlock, maxBlocksPerPart);

        if (localBlockIdx >= activeBlockCount) {
            return;
        }

        #define DECLARE_PASS(NEED_MASK)   \
        {                                   \
            using TBinCmp = typename TCmpBinsOneByteTrait<OneHotPass>::TCmpBins;\
            using THist = TSevenBitHistogram<BlockSize, NEED_MASK, TBinCmp>;\
            TBinCmp cmp(feature, fCount);\
            THist hist(&localHist[0], cmp);\
            ComputePairHistogram< BlockSize, innerUnroll, outerUnroll, THist>(partition->Offset, partition->Size,  cindex, pairs, weight, localBlockIdx, activeBlockCount, hist);\
        }
        if (maxBinCount < 128) {
            DECLARE_PASS(false);
        } else {
            DECLARE_PASS(true);
        }
        #undef DECLARE_PASS

        if (threadIdx.x < 64) {
            const int histId = threadIdx.x & 3;
            const int binId = (threadIdx.x >> 2) & 15;

            for (int fid = 0; fid < fCount; ++fid) {
                const ui32 bfStart = feature[fid].FirstFoldIndex;
                const int leqOrGeqShift = histId < 2 ? 0 : 8;
                const int isSecond = (histId & 1);

                for (int fold = binId; fold < feature[fid].Folds; fold += 16) {
                    const int readOffset = 32 * fold
                                           + 2 * fid
                                           + isSecond
                                           + leqOrGeqShift;

                    const float result = localHist[readOffset] + localHist[readOffset + 16];
                    if (abs(result) > 1e-20f) {
                        if (activeBlockCount > 1) {
                            atomicAdd(histogram + 4 * bfStart + 4 * fold + histId, result);
                        } else {
                            histogram[4 * bfStart + 4 * fold + histId] += result;
                        }
                    }
                }
            }
        }
    }


    template <bool OneHotPass>
    void ComputePairwiseHistogramOneByte7BitsImpl(const TCFeature* features,
                                                  const TCFeature* featuresCpu,
                                                  const ui32 featureCount,
                                                  const ui32 sevenBitsFeatureCount,
                                                  const ui32* compressedIndex,
                                                  const uint2* pairs,
                                                  ui32 /*pairCount*/,
                                                  const float* weight,
                                                  const TDataPartition* partition,
                                                  ui32 partCount,
                                                  ui32 histLineSize,
                                                  bool fullPass,
                                                  float* histogram,
                                                  int parallelStreams,
                                                  TCudaStream stream) {

        const bool hasOneHot = HasOneHotFeatures(featuresCpu, featureCount);

        if (!hasOneHot && OneHotPass) {
            return;
        }

        if (sevenBitsFeatureCount > 0 && partCount / (fullPass ? 1 : 4)) {
            const int blockSize = 96;
            dim3 numBlocks;
            numBlocks.x = (sevenBitsFeatureCount + 3) / 4;
            numBlocks.y = fullPass ? partCount : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
            const int blockPerFeatureMultiplier = CeilDivide<int>(TArchProps::SMCount() * blocksPerSm * 4, (parallelStreams * numBlocks.x * numBlocks.y * numBlocks.z));

            numBlocks.x = (featureCount + 3) / 4;
            numBlocks.x *= blockPerFeatureMultiplier;


            #define NB_HIST(IS_FULL)   \
            ComputeSplitPropertiesNonBinaryPairs7Bit < blockSize, IS_FULL, OneHotPass > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition,  histLineSize, histogram);

            if (fullPass) {
                NB_HIST(true)
            } else {
                NB_HIST(false)
            }


            #undef NB_HIST
        }
    }



    #define DEFINE_EXTERN(flag) \
    extern template             \
    void  ComputePairwiseHistogramOneByte7BitsImpl<flag>(const TCFeature* features, const TCFeature* featuresCpu,\
                                                        const ui32 featureCount,\
                                                        const ui32 fiveBitsFeatureCount,\
                                                        const ui32* compressedIndex,\
                                                        const uint2* pairs, ui32 pairCount,\
                                                        const float* weight,\
                                                        const TDataPartition* partition,\
                                                        ui32 partCount,\
                                                        ui32 histLineSize,\
                                                        bool fullPass,\
                                                        float* histogram, int parallelStreams,\
                                                        TCudaStream stream);
    DEFINE_EXTERN(false)
    DEFINE_EXTERN(true)

    #undef DEFINE_EXTERN
}
