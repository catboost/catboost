#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "compute_pair_hist_loop.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
using namespace cooperative_groups;

namespace NKernel {

    template <int BlockSize>
    struct TPairBinaryHist {
        float* Slice;

        __forceinline__ __device__ int HistSize() {
            return BlockSize * 16;
        }

        __forceinline__ __device__ int SliceOffset() {
            return 512 * (threadIdx.x >> 5);

        }

        __forceinline__ __device__ TPairBinaryHist(float* buff) {
            Slice = buff;
            for (int i = threadIdx.x; i < HistSize(); i += BlockSize) {
                Slice[i] = 0;
            }
            Slice += SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void AddPair(const ui32 ci1, const ui32 ci2, const float w) {
            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

            #pragma unroll 1
            for (int i = 0; i < 8; i++) {
                uchar f = (((threadIdx.x >> 2) + i) & 7) << 2;

                const ui32 bin1 = bfe(ci1, 28 - f, 4);
                const ui32 bin2 = bfe(ci2, 28 - f, 4);

                const ui32 invBin1 = (~bin1) & 15;
                const ui32 invBin2 = (~bin2) & 15;

                //00 01 10 11
                const ui32 bins = (invBin1 & invBin2) | ((invBin1 & bin2) << 8) | ((bin1 & invBin2) << 16) | ((bin1 & bin2) << 24);

                #pragma unroll 2
                for (int currentHist = 0; currentHist < 4; ++currentHist) {
                    const int histOffset = (threadIdx.x + currentHist) & 3;
                    const int bin = (bins >> (histOffset << 3)) & 15;
                    // 32 * bin + 4 * featureId + histId
                    //512 floats per warp
                    syncTile.sync();
                    Slice[f + (bin << 5) + histOffset] +=  w;
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

            __syncthreads();
            Slice -= SliceOffset();

            float sum = 0.f;

            if (threadIdx.x < 512) {
                const int warpCount = BlockSize / 32;
                int binId = threadIdx.x / 32;
                const int x = threadIdx.x & 31;
                Slice += 32 * binId + x;

                for (int warpId = 0; warpId < warpCount; ++warpId) {
                    sum += Slice[warpId * 512];
                }
            }
            __syncthreads();
            if (threadIdx.x < 512) {
                Slice[0] = sum;
            }
            __syncthreads();
        }
    };






    template <int BlockSize, bool IsFullPass>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
    #else
    __launch_bounds__(BlockSize, 1)
    #endif
    __global__ void ComputeSplitPropertiesBinaryPairs(const TCFeature* feature, int fCount, const ui32* cindex,
                                                      const uint2* pairs, const float* weight,
                                                      const TDataPartition* partition,
                                                      int histLineSize,
                                                      float* histogram) {

        const int maxBlocksPerPart = gridDim.x / ((fCount + 31) / 32);

        {
            const int featureOffset =  (blockIdx.x / maxBlocksPerPart) * 32;
            feature += featureOffset;
            cindex += feature->Offset;
            fCount = min(fCount - featureOffset, 32);
        }

        if (IsFullPass) {
            partition += blockIdx.y;
            histogram += blockIdx.y * ((ui64)histLineSize * 4ULL);
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partition += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * ((ui64)histLineSize) * 4ULL;
        }

        __shared__ float localHist[16 * BlockSize];

        if (partition->Size == 0) {
            return;
        }

        const int innerUnroll = 1;
        const int outerUnroll = 1;

        const int minDocsPerBlock = BlockSize * innerUnroll * 8;
        const int localBlockIdx = blockIdx.x % maxBlocksPerPart;
        const int activeBlockCount = min((partition->Size + minDocsPerBlock - 1) / minDocsPerBlock, maxBlocksPerPart);

        if (localBlockIdx >= activeBlockCount) {
            return;
        }

        {
            using THist = TPairBinaryHist<BlockSize>;
            THist hist(localHist);
            ComputePairHistogram<BlockSize, innerUnroll, outerUnroll, THist >(partition->Offset, partition->Size,
                                                                              cindex, pairs, weight,
                                                                              localBlockIdx, activeBlockCount,
                                                                              hist);
        }

        const int histId = threadIdx.x & 3;
        const int fid = (threadIdx.x >> 2);

        __syncthreads();

        if (fid < fCount) {
            float sum = 0;
            const int groupId = fid / 4;
            const int fixedBitId = 3 - fid % 4;
            const int activeMask = (1 << fixedBitId);

            //fix i'th bit and iterate through others
            #pragma unroll 1
            for (int i = 0; i < 16; ++i) {
                if (i & activeMask) {
                    sum += localHist[32 * i + 4 * groupId + histId];
                }
            }

            if (abs(sum) > 1e-20f) {
                atomicAdd(histogram + feature[fid].FirstFoldIndex * 4 + histId, sum);
            }
        }
    }



    void ComputePairwiseHistogramBinary(const TCFeature* features,const TCFeature*,
                                        const ui32 featureCount,
                                        const ui32 binFeatureCount,
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

        Y_ASSERT(featureCount == binFeatureCount);

        if (featureCount > 0 && partCount / (fullPass ? 1 : 4) > 0) {
            const int blockSize = 768;
            dim3 numBlocks;
            numBlocks.x = (featureCount + 31) / 32;
            numBlocks.y = fullPass ? partCount : partCount / 4;
            numBlocks.z = fullPass ? 1 : 3;

            const int blocksPerSm = TArchProps::GetMajorVersion() > 3 ? 2 : 1;
            const int blockPerFeatureMultiplier = CeilDivide<int>(TArchProps::SMCount() * blocksPerSm * 2, (parallelStreams * numBlocks.x * numBlocks.y * numBlocks.z));
            numBlocks.x *= blockPerFeatureMultiplier;


            #define NB_HIST(IS_FULL)   \
            ComputeSplitPropertiesBinaryPairs < blockSize, IS_FULL > << <numBlocks, blockSize, 0, stream>>>(\
                                                  features, featureCount, compressedIndex,  pairs,\
                                                  weight, partition, histLineSize,  histogram);


            if (fullPass) {
                NB_HIST(true)
            } else {
                NB_HIST(false)
            }

            #undef NB_HIST
        }
    }





}
