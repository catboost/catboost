#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "compute_pair_hist_loop.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <cstdio>

namespace NKernel {

    //shared-memory histograms
    //we store histogram via blocks.
    // every block is 32 bins x 4 features
    // Every binary features has no more, than 8 bits.
    // first 5 bits is index in block
    // next INNER_HIST_BIT is block in warp. For pairwise hists we have no more, than 2 blocks per warp
    // next OUTER_HIST_BITS sets number of warps, that'll be used, to store other part of hist
    // 1 << OUTER_HIST_BITS is number of warps, that will be "at the same time" compute 32 sequential points
    // this logic allows us to reuse l1-cache and make stripped-reads in one pass, instead of (binarization >> 5) passes;
    template<int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int BLOCK_SIZE>
    struct TPairHistOneByte
    {
        volatile float* Slice;
        uchar BlockId;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            //2 blocks if INNER_HIST_BITS_COUNT = 0, else 1
            const int blocks = 2  >> INNER_HIST_BITS_COUNT;
            //we store 4 histograms per block
            // x4 feature and x4 histograms, though histStart = blockIdx * 16
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 4)));
            return warpOffset + innerHistStart;
        }


        __forceinline__  __device__ TPairHistOneByte(float* buff)
        {
            Slice = buff;
            for (int i = threadIdx.x; i < BLOCK_SIZE * 32; i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            const int warpId = threadIdx.x / 32;
            BlockId = (uchar) (warpId & ((1 << OUTER_HIST_BITS_COUNT) - 1));
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const uint ci1,
                                                const uint ci2,
                                                const float w)
        {
            const int binMask = ((1 << (5 + INNER_HIST_BITS_COUNT)) - 1);
            const uchar shift = (threadIdx.x >> 2) & 3;

            #pragma unroll
            for (int i = 0; i < 4; i++)
            {
                const uchar f = 4 * ((shift + i) & 3);

                uint bin1 = bfe(ci1, 24 - 2 * f, 8);
                uint bin2 = bfe(ci2, 24 - 2 * f, 8);

                uchar mults = (((bin1 >> (5 + INNER_HIST_BITS_COUNT)) == BlockId)
                               | (((bin2 >> (5 + INNER_HIST_BITS_COUNT)) == BlockId) << 1));

                mults <<=  bin1 < bin2 ? 0 : 2;

                bin1 &= binMask;
                bin2 &= binMask;

                bin1 *= 1 << (5 - INNER_HIST_BITS_COUNT);
                bin2 *= 1 << (5 - INNER_HIST_BITS_COUNT);

                bin1 += f;
                bin2 += f + 1;

                #pragma  unroll
                for (int currentHist = 0; currentHist < 4; ++currentHist)
                {
                    const uchar histId = ((threadIdx.x + currentHist) & 3);
                    const int histOffset = histId < 2 ? 0  : 2;
                    const uint offset = ((histId & 1) ? bin2 : bin1) + histOffset;
                    const float toAdd = ((mults >> histId) & 1) ? w : 0;

                    //strange, but nvcc can't make this himself
                    if (INNER_HIST_BITS_COUNT != 0) {
                        #pragma unroll
                        for (int k = 0; k < (1 << INNER_HIST_BITS_COUNT); ++k)
                        {
                            if (((threadIdx.x >> 4) & ((1 << INNER_HIST_BITS_COUNT) - 1)) == k)
                            {
                                Slice[offset] += toAdd;
                            }
                        }
                    } else {
                        Slice[offset] += toAdd;
                    }
                }
            }
        }

        __forceinline__ __device__  void Reduce() {
            __syncthreads();
            Slice -= SliceOffset();

            const int outerHistCount =  1 << (OUTER_HIST_BITS_COUNT);
            const int totalLinesPerWarp = 32;

            const int warpIdx = (threadIdx.x / 32);
            const int warpCount = BLOCK_SIZE / 32; // 12
            const int x = (threadIdx.x & 31); // binIdx
            {
                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits)
                {
                    for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount)
                    {
                        int offset = 32 * line + x;
                        float sum = 0.0f;
                        #pragma unroll
                        for (int i = outerBits; i < warpCount; i += outerHistCount)
                        {
                            sum += Slice[offset + i * 1024];
                        }

                        Slice[offset + outerBits * 1024] = sum;
                    }
                }
                __syncthreads();


                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits)
                {
                    if ((INNER_HIST_BITS_COUNT == 0))
                    {
                        #pragma unroll
                        for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount)
                        {
                            //we use, that there can be only 1 or 2 hists in warp
                            if ((x < 16))
                            {
                                int offset = 32 * line + x + outerBits * 1024;
                                Slice[offset] += Slice[offset + 16];
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }
    };

//
    template<int BLOCK_SIZE>
    struct TPairHistOneByte <0, 0, BLOCK_SIZE> {
        volatile float* Slice;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            //2 blocks if INNER_HIST_BITS_COUNT = 0, else 1
            // x4 feature and x4 histograms, though histStart = blockIdx * 16
            const int innerHistStart = (threadIdx.x &  16);
            return warpOffset + innerHistStart;
        }


        __forceinline__  __device__ TPairHistOneByte(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < BLOCK_SIZE * 32; i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1,
                                                const ui32 ci2,
                                                const float w) {
            const int binMask = 31;
            const uchar shift = (threadIdx.x >> 2) & 3;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const uchar f = 4 * ((shift + i) & 3);

                ui32 bin1 = bfe(ci1, 24 - 2 * f, 8);
                ui32 bin2 = bfe(ci2, 24 - 2 * f, 8);

                uchar mults = ((bin2 < 32 ? 2 : 0)) + ((bin1 < 32) ? 1 : 0);
                mults <<=  bin1 < bin2 ? 0 : 2;

                bin1 &= binMask;
                bin2 &= binMask;

                bin1 *= 32;
                bin2 *= 32;

                bin1 += f;
                bin2 += f + 1;

                #pragma  unroll
                for (int currentHist = 0; currentHist < 4; ++currentHist) {
                    const uchar histId = ((threadIdx.x + currentHist) & 3);
                    const int histOffset = histId < 2 ? 0  : 2;
                    const ui32 offset = ((histId & 1) ? bin2 : bin1) + histOffset;
                    const float toAdd = ((mults >> histId) & 1) ? w : 0;
                    Slice[offset] += toAdd;
                }
            }
        }

        __forceinline__ __device__  void Reduce() {
            __syncthreads();
            Slice -= SliceOffset();

            const int outerHistCount =  1;
            const int totalLinesPerWarp = 32;

            const int warpIdx = (threadIdx.x / 32);
            const int warpCount = BLOCK_SIZE / 32; // 12
            const int x = (threadIdx.x & 31); // binIdx
            {
                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits) {
                    for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount) {
                        int offset = 32 * line + x;
                        float sum = 0.0f;
                        #pragma unroll
                        for (int i = outerBits; i < warpCount; i += outerHistCount) {
                            sum += Slice[offset + i * 1024];
                        }

                        Slice[offset + outerBits * 1024] = sum;
                    }
                }
                __syncthreads();


                #pragma unroll
                for (int outerBits = 0; outerBits < outerHistCount; ++outerBits) {
                    #pragma unroll
                    for (int line = warpIdx; line < totalLinesPerWarp; line += warpCount)
                    {
                        //we use, that there can be only 1 or 2 hists in warp
                        if ((x < 16)) {
                            int offset = 32 * line + x + outerBits * 1024;
                            Slice[offset] += Slice[offset + 16];
                        }
                    }
                }
            }

            __syncthreads();
        }
    };



    //TODO: trait class for unroll constants for different architectures
    template<int BLOCK_SIZE,
             int OUTER_BITS,
             int INNER_BITS,
             int N,
             int OUTER_UNROLL,
             int BLOCKS_PER_FEATURE>
    __forceinline__ __device__ void ComputeSplitPropertiesOneBytePass(const TCFeature* feature, int fCount,
                                                                       const ui32* __restrict cindex,
                                                                       const uint2* __restrict pairs,
                                                                       const float* __restrict  weight,
                                                                       const TDataPartition* partition,
                                                                       float* __restrict histogram,
                                                                       float*  smem) {

        using THist = TPairHistOneByte<OUTER_BITS, INNER_BITS, BLOCK_SIZE>;
        constexpr int stripeSize = BLOCK_SIZE >>  OUTER_BITS;
        constexpr int histBlockCount = 1 << OUTER_BITS;
        ComputePairHistogram< stripeSize, histBlockCount, N, OUTER_UNROLL, BLOCKS_PER_FEATURE, THist>(partition->Offset, cindex, partition->Size, pairs, weight, smem);

        if (threadIdx.x < 256) {
            const int histId = threadIdx.x & 3;
            const int binId = (threadIdx.x >> 2) & 15;
            const int fid = (threadIdx.x >> 6) & 3;
            const int binMask = ((1 << (5 + INNER_BITS)) - 1);

            if (fid < fCount) {
                const ui32 bfStart = feature[fid].FirstFoldIndex;
                histogram += 4 * bfStart;

                for (int fold = binId; fold < feature[fid].Folds; fold += 16) {
                    const int outerBits = fold >> (5 + INNER_BITS);
                    const int readBinIdx = (fold & binMask) << (5 - INNER_BITS);
                    const int readOffset = 1024 * outerBits + readBinIdx + (4 * fid +  histId);
                    if (BLOCKS_PER_FEATURE > 1) {
                        atomicAdd(histogram + 4 * fold + histId, smem[readOffset]);
                    } else {
                        histogram[4 * fold + histId] += smem[readOffset];
                    }
                }
            }
        }
    }



    #define DECLARE_PASS_ONE_BYTE(O, I, N, OUTER_UNROLL, M) \
        ComputeSplitPropertiesOneBytePass<BLOCK_SIZE, O, I, N, OUTER_UNROLL, M>(feature, fCount, cindex, pairs, weight, partition, histogram, &localHist[0]);



    template<int BLOCK_SIZE, bool FULL_PASS, int M>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
    #else
    __launch_bounds__(BLOCK_SIZE, 1)
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

        if (FULL_PASS) {
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

        __shared__ float localHist[32 * BLOCK_SIZE];

        const int maxBinCount = GetMaxBinCount(feature, fCount, (int*) &localHist[0]);
        __syncthreads();


        #if __CUDA__ARCH <= 350
        const int INNER_UNROLL = 4;
        const int OUTER_UNROLL = 2;
        #else
        //TODO(noxoomo): tune it on maxwell+
        const int INNER_UNROLL = 1;
        const int OUTER_UNROLL = 2;
        #endif

        if (maxBinCount <= 32) {
            DECLARE_PASS_ONE_BYTE(0, 0, INNER_UNROLL, OUTER_UNROLL, M)
        } else if (maxBinCount <= 64) {
            DECLARE_PASS_ONE_BYTE(1, 0, INNER_UNROLL, OUTER_UNROLL, M)
        } else if (maxBinCount <= 128) {
            DECLARE_PASS_ONE_BYTE(2, 0, INNER_UNROLL, OUTER_UNROLL, M)
        } else {
            DECLARE_PASS_ONE_BYTE(2, 1, 1, 1, M)
        }
    }


    void ComputePairwiseHistogramOneByte(const TCFeature* features,
                                         const ui32 featureCount,
                                         const ui32* compressedIndex,
                                         const uint2* pairs, ui32 pairCount,
                                         const float* weight,
                                         const TDataPartition* partition,
                                         ui32 partCount,
                                         ui32 histLineSize,
                                         bool fullPass,
                                         float* histogram,
                                         TCudaStream stream) {

        if (featureCount > 0) {
            const int blockSize = 384;
            dim3 numBlocks;
            numBlocks.x = (featureCount + 3) / 4;
            numBlocks.y = fullPass ? partCount : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier(numBlocks, pairCount, 64);
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
