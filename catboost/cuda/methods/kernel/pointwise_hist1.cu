#include "pointwise_hist1.cuh"
#include "split_properties_helpers.cuh"
#include <cooperative_groups.h>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template<int INNER_HIST_BITS_COUNT,
             int BLOCK_SIZE>
    struct TPointHistOneByte {
        volatile float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            const int blocks = 8 >> INNER_HIST_BITS_COUNT;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 2)));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistOneByte(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();

            __syncthreads();
        }

        __device__ void AddPoint(ui32 ci, const float t) {
            constexpr int outerLoopTileSize = 32 / (8 >> INNER_HIST_BITS_COUNT);
            constexpr int innerLoopTileSize = 1 << (2 + INNER_HIST_BITS_COUNT);
            thread_block_tile<outerLoopTileSize> outerLoopTile = tiled_partition<outerLoopTileSize>(this_thread_block());
            thread_block_tile<innerLoopTileSize> innerLoopTile = tiled_partition<innerLoopTileSize>(this_thread_block());

#pragma unroll
            for (int i = 0; i < 4; i++) {
                short f = (threadIdx.x + i) & 3;
                int bin = bfe(ci, 24 - 8 * f, 8);
                const float statToAdd =  (bin >> (5 + INNER_HIST_BITS_COUNT)) == 0 ? t : 0;

                const int mask = (1 << INNER_HIST_BITS_COUNT) - 1;
                const int higherBin = (bin >> 5) & mask;

                int offset = 4 * higherBin + f + ((bin & 31) << 5);

                if (INNER_HIST_BITS_COUNT > 0) {
#pragma unroll
                    for (int k = 0; k < (1 << INNER_HIST_BITS_COUNT); ++k) {
                        const int pass = ((threadIdx.x >> 2) + k) & mask;
                        if (pass == higherBin) {
                            Buffer[offset] += statToAdd;
                        }
                        innerLoopTile.sync();
                    }
                } else {
                    Buffer[offset] += statToAdd;
                }
                outerLoopTile.sync();
            }
        }

        __forceinline__ __device__ void Reduce() {

            Buffer -= SliceOffset();
            __syncthreads();
            {
                const int warpHistSize = 1024;
                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
                    //12 iterations
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }
            __syncthreads();

            //now we have only 1024 entries hist
            const int warpHistBlockCount = 8 >> INNER_HIST_BITS_COUNT;
            const int fold = threadIdx.x;
            const int histSize = 1 << (5 + INNER_HIST_BITS_COUNT);

            float sum[4];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum[i] = 0.0f;
            }

            if (fold < histSize) {
                const int warpHistSize = 1024;
                const int lowerBitsOffset = (fold & 31) << 5;
                const int higherBin = (fold >> 5) & ((1 << INNER_HIST_BITS_COUNT) - 1);
                const int blockSize = 4 * (1 << INNER_HIST_BITS_COUNT);

                const volatile float* src = Buffer + warpHistSize + lowerBitsOffset + 4 * higherBin;
                #pragma unroll
                for (int block = 0; block < warpHistBlockCount; ++block) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sum[i] += src[i + block * blockSize];
                    }
                }
            }

            __syncthreads();

            if (fold < histSize) {
                for (int i = 0; i < 4; ++i) {
                    Buffer[histSize * i + fold] = sum[i];
                }
            }
            __syncthreads();
        }
    };

    template<int BLOCK_SIZE>
    struct TPointHistHalfByte {
        volatile float* Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart = threadIdx.x & 24;
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistHalfByte(float* buff) {
            const int HIST_SIZE = 16 * BLOCK_SIZE;
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }
            __syncthreads();

            Buffer = buff + SliceOffset();
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t) {
            thread_block_tile<8> addToHistTile = tiled_partition<8>(this_thread_block());
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int f = (threadIdx.x + i) & 7;
                short bin = bfe(ci, 28 - 4 * f, 4);
                bin <<= 5;
                bin += f;
                Buffer[bin] += t;
                addToHistTile.sync();
            }
        }

        __device__ void Reduce() {
            Buffer -= SliceOffset();
            const int warpCount = BLOCK_SIZE >> 5;

            __syncthreads();
            {
                const int HIST_SIZE = 16 * BLOCK_SIZE;
                float sum = 0;

                if (threadIdx.x < 512) {
                    for (int i = threadIdx.x; i < HIST_SIZE; i += 512) {
                        sum += Buffer[i];
                    }
                }
                __syncthreads();

                if (threadIdx.x < 512) {
                    Buffer[threadIdx.x] = sum;
                }
                __syncthreads();
            }

            const int fold = (threadIdx.x  >> 3) & 15;
            float sum = 0.0f;

            if (threadIdx.x < 128) {
                const int featureId = threadIdx.x & 7;
                #pragma unroll
                for (int group = 0; group < 4; ++group) {
                    sum += Buffer[32 * fold + featureId + 8 * group];
                }
            }

            __syncthreads();

            if (threadIdx.x < 128) {
                Buffer[threadIdx.x] = sum;
            }

            __syncthreads();
        }
    };


    template<int STRIPE_SIZE, int OUTER_UNROLL, int N, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram(const ui32* __restrict__ indices,
                                                     int offset, int dsSize,
                                                     const float* __restrict__ target,
                                                     const ui32* __restrict__ cindex,
                                                     float* result) {

        target += offset;
        indices += offset;

        THist hist(result);
        int i = (threadIdx.x & 31) + (threadIdx.x / 32) * 32;

        //all operations should be warp-aligned
        //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
        {
            int lastId = min(dsSize, 32 - (offset & 31));

            if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0) {
                const int index = i < lastId ? __ldg(indices + i) : 0;
                const ui32 ci = i < lastId ? __ldg(cindex + index) : 0;
                const float wt = i < lastId ? __ldg(target + i) : 0;
                hist.AddPoint(ci, wt);
            }
            dsSize = max(dsSize - lastId, 0);

            indices += lastId;
            target += lastId;
        }

        //now lets align end
        const int unalignedTail = (dsSize & 31);

        if (unalignedTail != 0) {
            if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
            {
                const int tailOffset = dsSize - unalignedTail;
                const int index = i < unalignedTail ? __ldg(indices + tailOffset + i) : 0;
                const ui32 ci = i < unalignedTail ? __ldg(cindex + index) : 0;
                const float wt = i < unalignedTail ? __ldg(target + tailOffset + i) : 0;
                hist.AddPoint(ci, wt);
            }
        }
        dsSize -= unalignedTail;

        if (blockIdx.x % BLOCKS_PER_FEATURE == 0 && dsSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }


        indices += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        target += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        dsSize = max(dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE, 0);
        const int stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE;

        if (dsSize) {
            int iteration_count = (dsSize - i + (stripe - 1)) / stripe;
            int blocked_iteration_count = ((dsSize - (i | 31) + (stripe - 1)) / stripe) / N;

            target += i;
            indices += i;

#pragma unroll OUTER_UNROLL
            for (int j = 0; j < blocked_iteration_count; ++j) {
                ui32 local_index[N];
#pragma unroll
                for (int k = 0; k < N; k++) {
                    local_index[k] = __ldg(indices + stripe * k);
                }

                ui32 local_ci[N];
                float local_wt[N];

#pragma unroll
                for (int k = 0; k < N; ++k) {
                    local_ci[k] = __ldg(cindex + local_index[k]);
                    local_wt[k] = __ldg(target + stripe * k);
                }

#pragma unroll
                for (int k = 0; k < N; ++k) {
                    hist.AddPoint(local_ci[k], local_wt[k]);
                }

                indices += stripe * N;
                target += stripe * N;
            }

            for (int k = blocked_iteration_count * N; k < iteration_count; ++k) {
                const int index = __ldg(indices);
                ui32 ci = __ldg(cindex + index);
                float wt = __ldg(target);
                hist.AddPoint(ci, wt);
                indices += stripe;
                target += stripe;
            }
            __syncthreads();

            hist.Reduce();
        }
    }




    template<int STRIPE_SIZE, int OUTER_UNROLL, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram64BitLoads(const ui32* __restrict__ indices,
                                                               int offset, int dsSize,
                                                               const float* __restrict__ target,
                                                               const ui32* __restrict__ cindex,
                                                               float* result) {

        target += offset;
        indices += offset;

        THist hist(result);

        if (dsSize) {
            //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
            {
                int lastId = min(dsSize, 128 - (offset & 127));
                int colId = (threadIdx.x & 31) + (threadIdx.x / 32 ) * 32;

                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    for (; (colId < 128); colId += blockDim.x)
                    {
                        const int index = colId < lastId ? __ldg(indices + colId) : 0;
                        const ui32 ci = colId < lastId ? __ldg(cindex + index) : 0;
                        const float wt = colId < lastId ?  __ldg(target + colId) : 0;
                        hist.AddPoint(ci, wt);
                    }
                }

                dsSize = max(dsSize - lastId, 0);

                indices += lastId;
                target += lastId;
            }

            //now lets align end
            const int unalignedTail = (dsSize & 31);

            if (unalignedTail != 0) {
                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    int colId = (threadIdx.x & 31) + (threadIdx.x / 32 ) * 32;
                    const int tailOffset = dsSize - unalignedTail;

                    const int index = colId < unalignedTail ? __ldg(indices + tailOffset + colId) : 0;
                    const ui32 ci = colId < unalignedTail ? __ldg(cindex + index) : 0;
                    const float wt = colId < unalignedTail ? __ldg(target + tailOffset + colId) : 0;
                    hist.AddPoint(ci, wt);
                }
            }

            dsSize -= unalignedTail;

            if (dsSize <= 0) {
                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0) {
                    __syncthreads();
                    hist.Reduce();
                }
                return;
            }


            indices += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2;
            target += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2;

            const int stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE * 2;
            dsSize = max(dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2, 0);

            if (dsSize) {
                int iterCount;
                {
                    const int i = 2 * ((threadIdx.x & 31) + (threadIdx.x / 32) * 32);
                    target += i;
                    indices += i;
                    iterCount = (dsSize - i + (stripe - 1)) / stripe;
                }

                #pragma unroll OUTER_UNROLL
                for (int j = 0; j < iterCount; ++j) {
                    const uint2 localIndices = __ldg((uint2*) indices);
                    const ui32 firstBin = __ldg(cindex + localIndices.x);
                    const ui32 secondBin = __ldg(cindex + localIndices.y);
                    const float2 localTarget = __ldg((float2* )(target));

                    hist.AddPoint(firstBin, localTarget.x);
                    hist.AddPoint(secondBin, localTarget.y);

                    indices += stripe;
                    target += stripe;
                }
                __syncthreads();
                hist.Reduce();
            }
        }
    }


    template<int BLOCK_SIZE,
             int INNER_HIST_BITS_COUNT,
             int BLOCKS_PER_FEATURE,
             bool USE_64_BIT_LOAD>
    __forceinline__ __device__ void ComputeSplitPropertiesPass(const TCFeature* __restrict__ feature,
                                                               const ui32* __restrict__ cindex,
                                                               const float* __restrict__ target,
                                                               const ui32* __restrict__ indices,
                                                               const TDataPartition* __restrict__ partition, int fCount,
                                                               float* binSumsForPart,
                                                               float* __restrict__ smem) {

        using THist = TPointHistOneByte<INNER_HIST_BITS_COUNT, BLOCK_SIZE>;

        if (USE_64_BIT_LOAD) {
            #if __CUDA_ARCH__ < 300
            const int INNER_UNROLL = INNER_HIST_BITS_COUNT == 0 ? 4 : 2;
            const int OUTER_UNROLL = 2;
            #elif __CUDA_ARCH__ <= 350
            const int INNER_UNROLL = INNER_HIST_BITS_COUNT == 0 ? 8 : 4;
            const int OUTER_UNROLL = 2;
           #else
            const int INNER_UNROLL = 2;
            const int OUTER_UNROLL =  INNER_HIST_BITS_COUNT == 0 ? 4 : 2;
           #endif

            const int size = partition->Size;
            const int offset = partition->Offset;

            ComputeHistogram64BitLoads < BLOCK_SIZE, OUTER_UNROLL, BLOCKS_PER_FEATURE, THist > (indices, offset, size,
                                                                                                target,
                                                                                                cindex,
                                                                                                smem);
        } else {
            #if __CUDA_ARCH__ < 300
            const int INNER_UNROLL = INNER_HIST_BITS_COUNT == 0 ? 4 : 2;
            const int OUTER_UNROLL = 2;
            #elif __CUDA_ARCH__ <= 350
            const int INNER_UNROLL = INNER_HIST_BITS_COUNT == 0 ? 8 : 4;
            const int OUTER_UNROLL = 2;
            #else
            const int INNER_UNROLL = 4;
            const int OUTER_UNROLL = 2;
            #endif

            ComputeHistogram<BLOCK_SIZE, OUTER_UNROLL, INNER_UNROLL,  BLOCKS_PER_FEATURE, THist>(indices,
                                                                                                 partition->Offset,
                                                                                                 partition->Size,
                                                                                                 target,
                                                                                                 cindex,
                                                                                                 smem);
        }
        __syncthreads();

        const int fold = threadIdx.x;
        const int histSize = 1 << (5 + INNER_HIST_BITS_COUNT);

        #pragma unroll 4
        for (int fid = 0; fid < fCount; ++fid) {
            if (fold < feature[fid].Folds) {
                const float val = smem[fid * histSize + fold];
                if (abs(val) > 1e-20f) {
                    if (BLOCKS_PER_FEATURE > 1) {
                        atomicAdd(binSumsForPart + (feature[fid].FirstFoldIndex + fold), val);
                    } else {
                        WriteThrough(binSumsForPart + (feature[fid].FirstFoldIndex + fold), val);
                    }
                }
            }
        }
    }


#define DECLARE_PASS(I, M, USE_64_BIT_LOAD) \
    ComputeSplitPropertiesPass<BLOCK_SIZE, I, M, USE_64_BIT_LOAD>(feature, cindex, target, indices, partition, fCount, binSums, &counters[0]);


template<int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ == 600
    __launch_bounds__(BLOCK_SIZE, 1)
#elif __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesNBImpl(const TCFeature* __restrict__ feature, int fCount,
                                                 const ui32* __restrict__ cindex,
                                                 const float* __restrict__ target,
                                                 const ui32* __restrict__ indices,
                                                 const TDataPartition* __restrict__ partition,
                                                 float* __restrict__ binSums,
                                                 const int totalFeatureCount) {

        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS, 1);


        feature += (blockIdx.x / M) * 4;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 4, 4);

        __shared__ float counters[32 * BLOCK_SIZE];
        const int maxBinCount = GetMaxBinCount(feature, fCount, (int*) &counters[0]);
        __syncthreads();



        //CatBoost always use direct loads on first pass of histograms calculation and for this step 64-bits loads are almost x2 faster
        #if __CUDA_ARCH__ > 350
        const bool use64BitLoad =  FULL_PASS;// float2 for target/indices/weights
        #else
        const bool use64BitLoad =  false;
        #endif

        if (partition->Size) {
            if (maxBinCount <= 32) {
                DECLARE_PASS(0, M, use64BitLoad);
            } else if (maxBinCount <= 64) {
                DECLARE_PASS(1, M, false);
            } else if (maxBinCount <= 128) {
                DECLARE_PASS(2, M, false);
            } else {
                DECLARE_PASS(3, M, false);
            }
        }
    }





    template<int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesBImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__  partition,
            float* __restrict__ binSums,
            int totalFeatureCount) {

        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS, 1);

        feature += (blockIdx.x / M) * 32;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 32, 32);

        __shared__ float counters[16 * BLOCK_SIZE];

        if (partition->Size) {
            using THist = TPointHistHalfByte<BLOCK_SIZE>;
            #if __CUDA_ARCH__ > 350
            const bool use64bitLoad = FULL_PASS;
            #else
            const bool use64bitLoad = false;
            #endif
            if (use64bitLoad) {
                //full pass
                #if __CUDA_ARCH__ <= 350
                const int INNER_UNROLL = 4;
                const int OUTER_UNROLL = 1;
                #else
                const int INNER_UNROLL = 1;
                const int OUTER_UNROLL = 1;
                #endif
                ComputeHistogram64BitLoads < BLOCK_SIZE, OUTER_UNROLL,  M, THist > (indices, partition->Offset, partition->Size, target,  cindex, &counters[0]);
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

                ComputeHistogram < BLOCK_SIZE, OUTER_UNROLL, INNER_UNROLL, M, THist > (indices,
                                                                                       partition->Offset,
                                                                                       partition->Size,
                                                                                       target,
                                                                                       cindex,
                                                                                       &counters[0]);
            }

            ui32 fid = threadIdx.x;

            if (fid < fCount) {
                const int groupId = fid / 4;
                const int fMask = 1 << (3 - (fid & 3));

                float sum = 0.f;
                #pragma uroll
                for (int i = 0; i < 16; i++) {
                    if (!(i & fMask)) {
                        sum += counters[8 * i + groupId];
                    }
                }

                if (abs(sum) > 1e-20f) {
                    if (M > 1) {
                        atomicAdd(binSums + feature[fid].FirstFoldIndex, sum);
                    } else {
                        binSums[feature[fid].FirstFoldIndex] = sum;
                    }
                }
            }
        }
    }

    template<int BLOCK_SIZE,
            int BLOCKS_PER_FEATURE_COUNT>
    inline void RunComputeHist1NonBinaryKernel(const TCFeature* nbFeatures, int nbCount,
                                               const ui32* cindex,
                                               const float* target, const ui32* indices,
                                               const TDataPartition* partition,
                                               float* binSums,
                                               const int binFeatureCount,
                                               bool fullPass,
                                               TCudaStream stream,
                                               dim3 numBlocks)
    {

        if (fullPass) {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target,
                            indices, partition, binSums, binFeatureCount
            );

        } else {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, false, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target,
                            indices, partition, binSums, binFeatureCount
            );
        }

    }



    template<int BLOCK_SIZE, int BLOCKS_PER_FEATURE_COUNT>
    void RunComputeHist1BinaryKernel(const TCFeature* bFeatures, int bCount,
                                     const ui32* cindex,
                                     const float* target, const ui32* indices,
                                     const TDataPartition* partition,
                                     float* binSums,
                                     int histLineSize,
                                     bool fullPass,
                                     TCudaStream stream,
                                     dim3 numBlocks) {
        if (fullPass) {
            ComputeSplitPropertiesBImpl < BLOCK_SIZE, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(bFeatures, bCount, cindex, target,  indices, partition, binSums, histLineSize);
        } else {
            ComputeSplitPropertiesBImpl < BLOCK_SIZE, false, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(bFeatures, bCount, cindex, target,  indices, partition, binSums, histLineSize);
        }
    };




    template<int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesHalfByteImpl(
            const TCFeature* __restrict__ feature, int fCount,
            const ui32* __restrict__ cindex,
            const float* __restrict__ target,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount) {

        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS, 1);

        feature += (blockIdx.x / M) * 8;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 8, 8);

        __shared__ float smem[16 * BLOCK_SIZE];

        if (partition->Size) {

            using THist = TPointHistHalfByte<BLOCK_SIZE>;


            #if __CUDA_ARCH__ > 350
            const bool use64BitLoad = FULL_PASS;
            #else
            const bool use64BitLoad = false;
            #endif

            if (use64BitLoad) {
                #if __CUDA_ARCH__ <= 350
                const int INNER_UNROLL = 4;
                const int OUTER_UNROLL = 2;
                #else
                const int INNER_UNROLL = 1;
                const int OUTER_UNROLL = 1;
                #endif
                ComputeHistogram64BitLoads < BLOCK_SIZE, OUTER_UNROLL, M, THist >(indices, partition->Offset, partition->Size, target, cindex, &smem[0]);
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

                ComputeHistogram < BLOCK_SIZE, OUTER_UNROLL, INNER_UNROLL, M, THist > (indices, partition->Offset, partition->Size, target, cindex, &smem[0]);
            }

            __syncthreads();

            const int fid = threadIdx.x >> 4;
            const int fold = threadIdx.x & 15;


            if (fid < fCount && fold < feature[fid].Folds) {
                const float result = smem[fold * 8 + fid];
                if (abs(result) > 1e-20) {
                    if (M > 1) {
                        atomicAdd(binSums + feature[fid].FirstFoldIndex + fold, result);
                    } else {
                        binSums[feature[fid].FirstFoldIndex + fold] = result;
                    }
                }
            }
        }
    }


    template<int BLOCK_SIZE,
             int BLOCKS_PER_FEATURE_COUNT>
    inline void RunComputeHist1HalfByteKernel(const TCFeature* nbFeatures, int nbCount,
                                             const ui32* cindex,
                                             const float* target,
                                             const ui32* indices,
                                             const TDataPartition* partition,
                                             float* binSums,
                                              const int binFeatureCount,
                                             bool fullPass,
                                             TCudaStream stream,
                                             dim3 numBlocks)
    {

        if (fullPass) {
            ComputeSplitPropertiesHalfByteImpl < BLOCK_SIZE, true,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target,
                            indices, partition, binSums, binFeatureCount
            );

        } else {
            ComputeSplitPropertiesHalfByteImpl < BLOCK_SIZE, false,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target,
                            indices, partition, binSums, binFeatureCount);
        }
    }


    void ComputeHist1Binary(const TCFeature* bFeatures, ui32 bCount,
                            const ui32* cindex,
                            const float* target,
                            const ui32* indices,
                            ui32 size,
                            const TDataPartition* partition,
                            ui32 partsCount,
                            ui32 foldCount,
                            bool fullPass,
                            ui32 histLineSize,
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

        if (bCount) {

            #define COMPUTE(k)  \
            RunComputeHist1BinaryKernel<blockSize, k>(bFeatures, bCount, cindex, target,  indices, \
                                                      partition, binSums,  histLineSize, fullPass, stream, numBlocks); \

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

    void ComputeHist1HalfByte(const TCFeature* halfByteFeatures, ui32 halfByteFeaturesCount,
                              const ui32* cindex,
                              const float* target,
                              const ui32* indices,
                              ui32 size,
                              const TDataPartition* partition,
                              ui32 partsCount,
                              ui32 foldCount,
                              bool fullPass,
                              ui32 histLineSize,
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

        if (halfByteFeaturesCount) {

            #define COMPUTE(k)\
            RunComputeHist1HalfByteKernel<blockSize, k>(halfByteFeatures, halfByteFeaturesCount, cindex,\
                                                        target,\
                                                        indices, partition, binSums, histLineSize,\
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

    void ComputeHist1NonBinary(const TCFeature* nbFeatures, ui32 nbCount,
                               const ui32* cindex,
                               const float* target,
                               const ui32* indices,
                               ui32 size,
                               const TDataPartition* partition,
                               ui32 partCount,
                               ui32 foldCount,
                               bool fullPass,
                               ui32 histLineSize,
                               float* binSums,
                               TCudaStream stream) {
        if (nbCount) {

            dim3 numBlocks;
            numBlocks.x = (nbCount + 3) / 4;
            const int histCount = (fullPass ? partCount : partCount / 2);
            numBlocks.y = histCount;
            numBlocks.z = foldCount;
            const int blockSize = 384;
            const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
            numBlocks.x *= multiplier;

            #define COMPUTE(k)                                                                                              \
             RunComputeHist1NonBinaryKernel<blockSize, k>(nbFeatures, nbCount, cindex,  target, indices,                    \
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


}
