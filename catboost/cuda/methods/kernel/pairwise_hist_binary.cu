#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "compute_pair_hist_loop.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
using namespace cooperative_groups;

namespace NKernel {

    template <int BLOCK_SIZE>
    struct TPairBinaryHist {
        float* Slice;

        __forceinline__ __device__ int HistSize() {
            return BLOCK_SIZE * 16;
        }

        __forceinline__ __device__ int SliceOffset() {
            return 512 * (threadIdx.x >> 5);

        }

        __forceinline__ __device__ TPairBinaryHist(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < HistSize(); i += BLOCK_SIZE) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1, const ui32 ci2, const float w) {
            thread_block_tile<4> currentHistTile = tiled_partition<4>(this_thread_block());

            #pragma unroll 4
            for (int i = 0; i < 8; i++) {
                int f = (((threadIdx.x >> 2) + i) & 7) << 2;

                const int bin1 = bfe(ci1, 28 - f, 4);
                const int bin2 = bfe(ci2, 28 - f, 4);

                const int invBin1 = (~bin1) & 15;
                const int invBin2 = (~bin2) & 15;

                //00 01 10 11
                const ui32 bins = (invBin1 & invBin2) | ((invBin1 & bin2) << 8) | ((bin1 & invBin2) << 16) | ((bin1 & bin2) << 24);

                #pragma unroll
                for (int currentHist = 0; currentHist < 4; ++currentHist) {
                    const uchar histOffset = (threadIdx.x + currentHist) & 3;
                    const short bin = (bins >> (histOffset << 3)) & 15;
                    // 32 * bin + 4 * featureId + histId
                    //512 floats per warp
                    Slice[f + (bin << 5) + histOffset] +=  w;
                    currentHistTile.sync();
                }
                __syncwarp();
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
                    #pragma unroll
                    for (int warpId = 0; warpId < warpCount; ++warpId) {
                        sum += Slice[warpId * 512];
                    }
                }
            }
            __syncthreads();
            if (threadIdx.x < 512) {
                Slice[0] = sum;
            }
            __syncthreads();
        }
    };




    template<int BLOCK_SIZE, int INNER_UNROLL, int OUTER_UNROLL, int BLOCKS_PER_FEATURE>
    __forceinline__ __device__ void ComputeSplitPropertiesBinaryPass(const TCFeature* feature, int fCount,
                                                                      const ui32* __restrict cindex,
                                                                      const uint2* __restrict pairs,
                                                                      const float* __restrict  weight,
                                                                      const TDataPartition* partition,
                                                                      float* __restrict histogram,
                                                                      float* __restrict smem) {
        using THist = TPairBinaryHist<BLOCK_SIZE>;
        ComputePairHistogram<BLOCK_SIZE, 1, INNER_UNROLL, OUTER_UNROLL, BLOCKS_PER_FEATURE, THist >(partition->Offset, cindex, partition->Size, pairs, weight, smem);

        const int histId = threadIdx.x & 3;
        const int fid = (threadIdx.x >> 2);

        __syncthreads();

        if (fid < fCount) {
            float sum = 0;
            const int groupId = fid / 4;
            const int fixedBitId = 3 - fid % 4;
            const int activeMask = (1 << fixedBitId);

            //fix i'th bit and iterate through others
            for (int i = 0; i < 16; ++i) {
                if (i & activeMask) {
                    sum += smem[32 * i + 4 * groupId + histId];
                }
            }

            if (BLOCKS_PER_FEATURE > 1) {
                atomicAdd(histogram + feature[fid].FirstFoldIndex * 4 + histId, sum);
            } else {
                histogram[feature[fid].FirstFoldIndex * 4 + histId] += sum;
            }
        }
        __syncthreads();
    }


    #define DECLARE_PASS_BINARY(N, OUTER_UNROLL, M) \
        ComputeSplitPropertiesBinaryPass<BLOCK_SIZE, N, OUTER_UNROLL, M>(feature, fCount, cindex, pairs, weight, partition, histogram, &localHist[0]);



    template<int BLOCK_SIZE, bool FULL_PASS, int M>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
    #else
    __launch_bounds__(BLOCK_SIZE, 1)
    #endif
    __global__ void ComputeSplitPropertiesBinaryPairs(const TCFeature* feature, int fCount, const ui32* cindex,
                                                      const uint2* pairs, const float* weight,
                                                      const TDataPartition* partition,
                                                      int histLineSize,
                                                      float* histogram) {
        {
            const int featureOffset =  (blockIdx.x / M) * 32;
            feature += featureOffset;
            cindex += feature->Offset;
            fCount = min(fCount - featureOffset, 32);
        }
        if (FULL_PASS) {
            partition += blockIdx.y;
            histogram += blockIdx.y * ((ui64)histLineSize * 4ULL);
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partition += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * ((ui64)histLineSize) * 4ULL;
        }

        __shared__ float localHist[16 * BLOCK_SIZE];

        if (partition->Size == 0) {
            return;
        }

        DECLARE_PASS_BINARY(1, 1, M);
    }


    void ComputePairwiseHistogramBinary(const TCFeature* features,
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
            numBlocks.x = (featureCount + 31) / 32;
            numBlocks.y = fullPass ? partCount : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const ui32 blockPerFeatureMultiplier = EstimateBlockPerFeatureMultiplier(numBlocks, pairCount, 64);
            numBlocks.x *= blockPerFeatureMultiplier;


            #define NB_HIST(IS_FULL, BLOCKS_PER_FEATURE)   \
            ComputeSplitPropertiesBinaryPairs < blockSize, IS_FULL, BLOCKS_PER_FEATURE > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition, histLineSize,  histogram);

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
            }  else if (blockPerFeatureMultiplier == 64) {
                DISPATCH(64);
            }  else {
                exit(0);
            }
            #undef NB_HIST
            #undef DISPATCH
        }
    }





}
