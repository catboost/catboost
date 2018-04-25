#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "compute_pair_hist_loop.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {

    //768
    template<int BLOCK_SIZE>
    struct TPairHistHalfByte {
        volatile float* Slice;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            //we store 4 histograms per block
            // x8 feature and x4 histograms, though histStart = blockIdx * 16
            return warpOffset;
        }

        __forceinline__ __device__ int HistSize() {
            return 16 * BLOCK_SIZE;
        }

        __forceinline__  __device__ TPairHistHalfByte(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < BLOCK_SIZE * 16; i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1,
                                                const ui32 ci2,
                                                const float w)
        {
            const uchar shift = (threadIdx.x >> 2) & 7;

            for (int i = 0; i < 8; i++) {
                const uchar f = 4 * ((shift + i) & 7);

                ui32 bin1 = bfe(ci1, 28 - f, 4);
                ui32 bin2 = bfe(ci2, 28 - f, 4);
                const bool isLeq = bin1 < bin2;

                bin1 <<= 5;
                bin2 <<= 5;

                bin1 += f;
                bin2 += f + 1;

                for (int currentHist = 0; currentHist < 4; ++currentHist) {

                    const uchar histId = ((threadIdx.x + currentHist) & 3);
                    const bool addToLeqHist = histId < 2;
                    const ui32 offset = ((histId & 1) ? bin2 : bin1) + (addToLeqHist ? 0 : 2);
                    const float toAdd = (isLeq == addToLeqHist) ? w : 0;

                    //offset = 32 * bin + 4 * feature + histId
                    //feature from 0 to 7, histId from 0 to 3
                    //hist0 and hist2 use bin1
                    //host 1 and hist 3 use bin2
                    Slice[offset] += toAdd;
                }
            }
        }

        __forceinline__ __device__  void Reduce() {

            __syncthreads();
            Slice -= SliceOffset();

            float sum = 0.f;

            if (threadIdx.x < 512) {
                const int warpCount = BLOCK_SIZE / 32;
                int binId = threadIdx.x / 32;
                const int x = threadIdx.x & 31;
                Slice += 32 * binId + x;

                {
                    for (int warpId = 0; warpId < warpCount; ++warpId) {
                        sum += Slice[warpId * 512];
                    }
                }
            }
            __syncthreads();
            //bin0: f0: hist0 hist1 hist2 hist3 f1: hist0 hist1 hist2 hist3 …
            if (threadIdx.x < 512) {
                Slice[0] = sum;
            }
            __syncthreads();
        }
    };



    template<int BLOCK_SIZE, int N, int OUTER_UNROLL, int BLOCKS_PER_FEATURE>
    __forceinline__ __device__ void ComputeSplitPropertiesHalfBytePass(const TCFeature* feature, int fCount,
                                                                        const uint* __restrict cindex,
                                                                        const uint2* __restrict pairs, const float* __restrict  weight,
                                                                        const TDataPartition* partition,
                                                                        float* __restrict histogram,
                                                                        float* __restrict smem) {
        using THist = TPairHistHalfByte<BLOCK_SIZE>;
        ComputePairHistogram<BLOCK_SIZE, 1, N, OUTER_UNROLL, BLOCKS_PER_FEATURE,  THist >(partition->Offset, cindex, partition->Size, pairs, weight, smem);


        if (threadIdx.x < 512) {
            const int histId = threadIdx.x & 3;
            const int fold = (threadIdx.x >> 2) & 15;
            const int fid = (threadIdx.x >> 6) & 7;

            if (fid < fCount) {
                const ui32 bfStart = feature[fid].FirstFoldIndex;
                histogram += 4 * bfStart;

                if (fold < feature[fid].Folds) {
                    const int readOffset = 32 * fold + 4 * fid + histId;

                    if (BLOCKS_PER_FEATURE > 1) {
                        atomicAdd(histogram + 4 * fold + histId, smem[readOffset]);
                    } else {
                        histogram[4 * fold + histId] += smem[readOffset];
                    }
                }
            }
        }
    }



    #define DECLARE_PASS_HALF_BYTE(N, OUTER_UNROLL, M) \
        ComputeSplitPropertiesHalfBytePass<BLOCK_SIZE, N, OUTER_UNROLL, M>(feature, fCount, cindex, pairs, weight, partition, histogram, &localHist[0]);


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
    #else
    __launch_bounds__(BLOCK_SIZE, 1)
    #endif
    __global__ void ComputeSplitPropertiesHalfBytePairs(const TCFeature* feature, int fCount, const ui32* cindex,
                                                        const uint2* pairs, const float* weight,
                                                        const TDataPartition* partition,
                                                        int histLineSize,
                                                        float* histogram) {
        //histogram line size - size of one part hist.
        const int featureOffset = (blockIdx.x / M) * 8;
        feature += featureOffset;
        cindex += feature->Offset;
        fCount = min(fCount - featureOffset, 8);

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

        __shared__ float localHist[16 * BLOCK_SIZE];

        DECLARE_PASS_HALF_BYTE(1, 1, M)
    }





    void ComputePairwiseHistogramHalfByte(const TCFeature* features,
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
            const int blockSize = 768;
            dim3 numBlocks;
            numBlocks.x = (featureCount + 7) / 8;
            numBlocks.y = fullPass ? partCount : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier(numBlocks, pairCount, 32);
            numBlocks.x *= blockPerFeatureMultiplier;


            #define NB_HIST(IS_FULL, BLOCKS_PER_FEATURE)   \
            ComputeSplitPropertiesHalfBytePairs < blockSize, IS_FULL, BLOCKS_PER_FEATURE > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition, histLineSize, histogram);

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
            } else {
                exit(0);
            }
            #undef NB_HIST
            #undef DISPATCH
        }
    }


}
