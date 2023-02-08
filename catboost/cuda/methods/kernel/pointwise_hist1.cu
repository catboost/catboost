#include "pointwise_hist1.cuh"
#include "split_properties_helpers.cuh"
#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

using namespace cooperative_groups;

namespace NKernel
{

    template <int InnerHistBitsCount,
             int BlockSize>
    struct TPointHistOneByte {
        volatile float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            const int blocks = 8 >> InnerHistBitsCount;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (InnerHistBitsCount + 2)));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistOneByte(float* buff) {

            const int HIST_SIZE = 32 * BlockSize;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BlockSize) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();

            __syncthreads();
        }

        __device__ void AddPoint(ui32 ci, const float t) {
            thread_block_tile<32> syncTile = tiled_partition<32>(this_thread_block());

#pragma unroll
            for (int i = 0; i < 4; i++) {
                short f = (threadIdx.x + i) & 3;
                int bin = bfe(ci, 24 - 8 * f, 8);
                const float statToAdd =  (bin >> (5 + InnerHistBitsCount)) == 0 ? t : 0;

                const int mask = (1 << InnerHistBitsCount) - 1;
                const int higherBin = (bin >> 5) & mask;

                int offset = 4 * higherBin + f + ((bin & 31) << 5);

                if (InnerHistBitsCount > 0) {
#pragma unroll
                    for (int k = 0; k < (1 << InnerHistBitsCount); ++k) {
                        const int pass = ((threadIdx.x >> 2) + k) & mask;
                        syncTile.sync();
                        if (pass == higherBin) {
                            Buffer[offset] += statToAdd;
                        }
                    }
                } else {
                    syncTile.sync();
                    Buffer[offset] += statToAdd;
                }
            }
        }

        __forceinline__ __device__ void Reduce() {

            Buffer -= SliceOffset();
            __syncthreads();
            {
                const int warpHistSize = 1024;
                for (int start = threadIdx.x; start < warpHistSize; start += BlockSize) {
                    float sum = 0;
                    //12 iterations
                    #pragma unroll 12
                    for (int i = start; i < 32 * BlockSize; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }
            __syncthreads();

            //now we have only 1024 entries hist
            const int warpHistBlockCount = 8 >> InnerHistBitsCount;
            const int fold = threadIdx.x;
            const int histSize = 1 << (5 + InnerHistBitsCount);

            float sum[4];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                sum[i] = 0.0f;
            }

            if (fold < histSize) {
                const int warpHistSize = 1024;
                const int lowerBitsOffset = (fold & 31) << 5;
                const int higherBin = (fold >> 5) & ((1 << InnerHistBitsCount) - 1);
                const int blockSize = 4 * (1 << InnerHistBitsCount);

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

    template <int BlockSize>
    struct TPointHistHalfByte {
        volatile float* Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart = threadIdx.x & 24;
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistHalfByte(float* buff) {
            const int histSize = 16 * BlockSize;
            for (int i = threadIdx.x; i < histSize; i += BlockSize) {
                buff[i] = 0;
            }
            __syncthreads();

            Buffer = buff + SliceOffset();
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t) {
            thread_block_tile<8> addToHistTile = tiled_partition<8>(this_thread_block());
#pragma unroll 4
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

            __syncthreads();
            {
                const int HIST_SIZE = 16 * BlockSize;
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


    template <int StripeSize, int OuterUnroll, int N,  typename THist>
    __forceinline__ __device__ void ComputeHistogram(int BlocksPerFeature, const ui32* __restrict__ indices,
                                                     ui32 offset, ui32 dsSize,
                                                     const float* __restrict__ target,
                                                     const ui32* __restrict__ cindex,
                                                     float* result) {

        target += offset;
        indices += offset;

        THist hist(result);
        ui32 i = (threadIdx.x & 31) + (threadIdx.x / 32) * 32;

        //all operations should be warp-aligned
        //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
        {
            ui32 lastId = min(dsSize, 32 - (offset & 31));

            if ((blockIdx.x % BlocksPerFeature) == 0) {
                const int index = i < lastId ? __ldg(indices + i) : 0;
                const ui32 ci = i < lastId ? __ldg(cindex + index) : 0;
                const float wt = i < lastId ? __ldg(target + i) : 0;
                hist.AddPoint(ci, wt);
            }
            dsSize = dsSize > lastId > 0 ? dsSize - lastId : 0;

            indices += lastId;
            target += lastId;
        }

        //now lets align end
        const ui32 unalignedTail = (dsSize & 31);

        if (unalignedTail != 0) {
            if ((blockIdx.x % BlocksPerFeature) == 0)
            {
                const ui32 tailOffset = dsSize - unalignedTail;
                const int index = i < unalignedTail ? __ldg(indices + tailOffset + i) : 0;
                const ui32 ci = i < unalignedTail ? __ldg(cindex + index) : 0;
                const float wt = i < unalignedTail ? __ldg(target + tailOffset + i) : 0;
                hist.AddPoint(ci, wt);
            }
        }
        dsSize -= unalignedTail;

        if (blockIdx.x % BlocksPerFeature == 0 && dsSize <= 0) {
            __syncthreads();
            hist.Reduce();
            return;
        }


        indices += (blockIdx.x % BlocksPerFeature) * StripeSize;
        target += (blockIdx.x % BlocksPerFeature) * StripeSize;
        dsSize = dsSize > (blockIdx.x % BlocksPerFeature) * StripeSize ? dsSize - (blockIdx.x % BlocksPerFeature) * StripeSize : 0;
        const ui32 stripe = StripeSize * BlocksPerFeature;

        if (dsSize) {
            ui32 iteration_count = dsSize > i ? (dsSize - i + (stripe - 1)) / stripe : 0;
            ui32 blocked_iteration_count = dsSize > (i | 31) ? ((dsSize - (i | 31) + (stripe - 1)) / stripe) / N : 0;

            target += i;
            indices += i;

#pragma unroll OuterUnroll
            for (ui32 j = 0; j < blocked_iteration_count; ++j) {
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

            for (ui32 k = blocked_iteration_count * N; k < iteration_count; ++k) {
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




    template <int StripeSize, int OuterUnroll,  typename THist>
    __forceinline__ __device__ void ComputeHistogram64BitLoads(int BlocksPerFeature, const ui32* __restrict__ indices,
                                                               ui32 offset, ui32 dsSize,
                                                               const float* __restrict__ target,
                                                               const ui32* __restrict__ cindex,
                                                               float* result) {

        target += offset;
        indices += offset;

        THist hist(result);

        if (dsSize) {
            //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
            {
                ui32 lastId = min(dsSize, 128 - (offset & 127));
                ui32 colId = (threadIdx.x & 31) + (threadIdx.x / 32 ) * 32;

                if ((blockIdx.x % BlocksPerFeature) == 0)
                {
                    for (; (colId < 128); colId += blockDim.x)
                    {
                        const int index = colId < lastId ? __ldg(indices + colId) : 0;
                        const ui32 ci = colId < lastId ? __ldg(cindex + index) : 0;
                        const float wt = colId < lastId ?  __ldg(target + colId) : 0;
                        hist.AddPoint(ci, wt);
                    }
                }

                dsSize = dsSize > lastId ? dsSize - lastId : 0;

                indices += lastId;
                target += lastId;
            }

            //now lets align end
            const ui32 unalignedTail = (dsSize & 63);

            if (unalignedTail != 0) {
                if ((blockIdx.x % BlocksPerFeature) == 0)
                {
                    ui32 colId = (threadIdx.x & 31) + (threadIdx.x / 32 ) * 32;
                    const ui32 tailOffset = dsSize - unalignedTail;
                    for (; (colId < 64); colId += blockDim.x) {
                        const int index = colId < unalignedTail ? __ldg(indices + tailOffset + colId) : 0;
                        const ui32 ci = colId < unalignedTail ? __ldg(cindex + index) : 0;
                        const float wt = colId < unalignedTail ? __ldg(target + tailOffset + colId) : 0;
                        hist.AddPoint(ci, wt);
                    }
                }
            }

            dsSize -= unalignedTail;

            if (dsSize <= 0) {
                if ((blockIdx.x % BlocksPerFeature) == 0) {
                    __syncthreads();
                    hist.Reduce();
                }
                return;
            }


            indices += (blockIdx.x % BlocksPerFeature) * StripeSize * 2;
            target += (blockIdx.x % BlocksPerFeature) * StripeSize * 2;

            const ui32 stripe = StripeSize * BlocksPerFeature * 2;
            dsSize = dsSize > (blockIdx.x % BlocksPerFeature) * StripeSize * 2 ? dsSize - (blockIdx.x % BlocksPerFeature) * StripeSize * 2 : 0;

            if (dsSize) {
                ui32 iterCount;
                {
                    const ui32 i = 2 * ((threadIdx.x & 31) + (threadIdx.x / 32) * 32);
                    target += i;
                    indices += i;
                    iterCount = dsSize > i ? (dsSize - i + (stripe - 1)) / stripe : 0;
                }

                #pragma unroll OuterUnroll
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


    template <int BlockSize,
             int InnerHistBitsCount,
             bool Use64BitLoads>
    __forceinline__ __device__ void ComputeSplitPropertiesPass(int BlocksPerFeature, const TCFeature* __restrict__ feature,
                                                               const ui32* __restrict__ cindex,
                                                               const float* __restrict__ target,
                                                               const ui32* __restrict__ indices,
                                                               const TDataPartition* __restrict__ partition, int fCount,
                                                               float* binSumsForPart,
                                                               float* __restrict__ smem) {

        using THist = TPointHistOneByte<InnerHistBitsCount, BlockSize>;

        if (Use64BitLoads) {
            #if __CUDA_ARCH__ < 300
            const int outerUnroll = 2;
            #elif __CUDA_ARCH__ <= 350
            const int outerUnroll = 2;
           #else
            const int outerUnroll =  InnerHistBitsCount == 0 ? 4 : 2;
           #endif

            const ui32 size = partition->Size;
            const ui32 offset = partition->Offset;

            ComputeHistogram64BitLoads < BlockSize, outerUnroll, THist > (BlocksPerFeature,
                                                                          indices,
                                                                          offset,
                                                                          size,
                                                                          target,
                                                                          cindex,
                                                                          smem);
        } else {
            #if __CUDA_ARCH__ < 300
            const int innerUnroll = InnerHistBitsCount == 0 ? 4 : 2;
            const int outerUnroll = 2;
            #elif __CUDA_ARCH__ <= 350
            const int innerUnroll = InnerHistBitsCount == 0 ? 8 : 4;
            const int outerUnroll = 2;
            #else
            const int innerUnroll = 4;
            const int outerUnroll = 2;
            #endif

            ComputeHistogram<BlockSize, outerUnroll, innerUnroll,  THist>(BlocksPerFeature,
                                                                          indices,
                                                                          partition->Offset,
                                                                          partition->Size,
                                                                          target,
                                                                          cindex,
                                                                          smem);
        }
        __syncthreads();

        const ui32 fold = threadIdx.x;
        const ui32 histSize = 1 << (5 + InnerHistBitsCount);

        #pragma unroll 4
        for (int fid = 0; fid < fCount; ++fid) {
            if (fold < feature[fid].Folds) {
                const float val = smem[fid * histSize + fold];
                if (abs(val) > 1e-20f) {
                    if (BlocksPerFeature > 1) {
                        atomicAdd(binSumsForPart + (feature[fid].FirstFoldIndex + fold), val);
                    } else {
                        WriteThrough(binSumsForPart + (feature[fid].FirstFoldIndex + fold), val);
                    }
                }
            }
        }
    }


#define DECLARE_PASS(I, M, USE_64_BIT_LOAD) \
    ComputeSplitPropertiesPass<BlockSize, I,  USE_64_BIT_LOAD>(M, feature, cindex, target, indices, partition, fCount, binSums, &counters[0]);


template <int BlockSize, bool IsFullPass>
#if __CUDA_ARCH__ == 600
    __launch_bounds__(BlockSize, 1)
#elif __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesNBImpl(int M, const TCFeature* __restrict__ feature, int fCount,
                                                 const ui32* __restrict__ cindex,
                                                 const float* __restrict__ target,
                                                 const ui32* __restrict__ indices,
                                                 const TDataPartition* __restrict__ partition,
                                                 float* __restrict__ binSums,
                                                 const int totalFeatureCount) {

        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, IsFullPass, 1);


        feature += (blockIdx.x / M) * 4;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 4, 4);

        __shared__ float counters[32 * BlockSize];
        const ui32 maxBinCount = GetMaxBinCount(feature, fCount, (ui32*) &counters[0]);
        __syncthreads();



        //CatBoost always use direct loads on first pass of histograms calculation and for this step 64-bits loads are almost x2 faster
        #if __CUDA_ARCH__ > 350
        const bool use64BitLoad =  IsFullPass;// float2 for target/indices/weights
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





    template <int BlockSize, bool IsFullPass>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesBImpl(int M,
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__  partition,
            float* __restrict__ binSums,
            int totalFeatureCount) {

        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, IsFullPass, 1);

        feature += (blockIdx.x / M) * 32;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 32, 32);

        __shared__ float counters[16 * BlockSize];

        if (partition->Size) {
            using THist = TPointHistHalfByte<BlockSize>;
            #if __CUDA_ARCH__ > 350
            const bool use64bitLoad = IsFullPass;
            #else
            const bool use64bitLoad = false;
            #endif
            if (use64bitLoad) {
                //full pass
                #if __CUDA_ARCH__ <= 350
                const int outerUnroll = 1;
                #else
                const int outerUnroll = 1;
                #endif
                ComputeHistogram64BitLoads < BlockSize, outerUnroll,   THist > (M, indices, partition->Offset, partition->Size, target,  cindex, &counters[0]);
            } else {
                #if __CUDA_ARCH__ <= 300
                const int innerUnroll = 2;
                const int outerUnroll = 1;
                #elif __CUDA_ARCH__ <= 350
                const int innerUnroll = 4;
                const int outerUnroll = 1;
                #else
                const int innerUnroll = 1;
                const int outerUnroll = 1;
                #endif

                ComputeHistogram < BlockSize, outerUnroll, innerUnroll,  THist > (M, indices,
                                                                                       partition->Offset,
                                                                                       partition->Size,
                                                                                       target,
                                                                                       cindex,
                                                                                       &counters[0]);
            }

            ui32 fid = threadIdx.x;

            if (fid < fCount) {
                const ui32 groupId = fid / 4;
                const ui32 fMask = 1 << (3 - (fid & 3));

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

    template <int BlockSize,
            int BlocksPerFeatureCount>
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
            ComputeSplitPropertiesNBImpl < BlockSize, true > << <numBlocks, BlockSize, 0, stream>>>(BlocksPerFeatureCount,
                    nbFeatures, nbCount, cindex, target,
                            indices, partition, binSums, binFeatureCount
            );

        } else {
            ComputeSplitPropertiesNBImpl < BlockSize, false > << <numBlocks, BlockSize, 0, stream>>>( BlocksPerFeatureCount,
                    nbFeatures, nbCount, cindex, target,
                            indices, partition, binSums, binFeatureCount
            );
        }

    }



    template <int BlockSize, int BlocksPerFeatureCount>
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
            ComputeSplitPropertiesBImpl < BlockSize, true > << <numBlocks, BlockSize, 0, stream>>>(BlocksPerFeatureCount, bFeatures, bCount, cindex, target,  indices, partition, binSums, histLineSize);
        } else {
            ComputeSplitPropertiesBImpl < BlockSize, false > << <numBlocks, BlockSize, 0, stream>>>(BlocksPerFeatureCount, bFeatures, bCount, cindex, target,  indices, partition, binSums, histLineSize);
        }
    };




    template <int BlockSize, bool IsFullPass>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BlockSize, 2)
#else
    __launch_bounds__(BlockSize, 1)
#endif
    __global__ void ComputeSplitPropertiesHalfByteImpl(
            int M,
            const TCFeature* __restrict__ feature, int fCount,
            const ui32* __restrict__ cindex,
            const float* __restrict__ target,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount) {

        TPointwisePartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, IsFullPass, 1);

        feature += (blockIdx.x / M) * 8;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 8, 8);

        __shared__ float smem[16 * BlockSize];

        if (partition->Size) {

            using THist = TPointHistHalfByte<BlockSize>;


            #if __CUDA_ARCH__ > 350
            const bool use64BitLoad = IsFullPass;
            #else
            const bool use64BitLoad = false;
            #endif

            if (use64BitLoad) {
                #if __CUDA_ARCH__ <= 350
                const int outerUnroll = 2;
                #else
                const int outerUnroll = 1;
                #endif
                ComputeHistogram64BitLoads < BlockSize, outerUnroll, THist >(M, indices, partition->Offset, partition->Size, target, cindex, &smem[0]);
            } else {
                #if __CUDA_ARCH__ <= 300
                const int innerUnroll = 2;
                const int outerUnroll = 2;
                #elif __CUDA_ARCH__ <= 350
                const int innerUnroll = 4;
                const int outerUnroll = 2;
                #else
                const int innerUnroll = 1;
                const int outerUnroll = 1;
                #endif

                ComputeHistogram < BlockSize, outerUnroll, innerUnroll, THist > (M, indices, partition->Offset, partition->Size, target, cindex, &smem[0]);
            }

            __syncthreads();

            const ui32 fid = threadIdx.x >> 4;
            const ui32 fold = threadIdx.x & 15;


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


    template <int BlockSize,
             int BlocksPerFeatureCount>
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
            ComputeSplitPropertiesHalfByteImpl < BlockSize, true > << <numBlocks, BlockSize, 0, stream>>>(
                    BlocksPerFeatureCount, nbFeatures, nbCount, cindex, target, indices, partition, binSums, binFeatureCount
            );

        } else {
            ComputeSplitPropertiesHalfByteImpl < BlockSize, false > << <numBlocks, BlockSize, 0, stream>>>(
                    BlocksPerFeatureCount, nbFeatures, nbCount, cindex, target, indices, partition, binSums, binFeatureCount);
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
        const ui32 histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = histCount;
        numBlocks.z = foldCount;

        constexpr ui32 BlockSize = 768;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64u);
        numBlocks.x *= multiplier;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        if (bCount) {

            #define COMPUTE(k)  \
            RunComputeHist1BinaryKernel<BlockSize, k>(bFeatures, bCount, cindex, target,  indices, \
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
                CB_ENSURE_INTERNAL(false, "Expected multiplier = 1, 2, 4, 8, 16, 32, or 64, not " << multiplier);
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
        const ui32 histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = static_cast<ui32>(histCount);
        numBlocks.z = foldCount;

        constexpr ui32 BlockSize = 768;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
        numBlocks.x *= multiplier;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        if (halfByteFeaturesCount) {

            #define COMPUTE(k)\
            RunComputeHist1HalfByteKernel<BlockSize, k>(halfByteFeatures, halfByteFeaturesCount, cindex,\
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
                CB_ENSURE_INTERNAL(false, "Expected multiplier = 1, 2, 4, 8, 16, 32, or 64, not " << multiplier);
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
            const ui32 histCount = (fullPass ? partCount : partCount / 2);
            numBlocks.y = histCount;
            numBlocks.z = foldCount;
            constexpr ui32 BlockSize = 384;
            const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
            numBlocks.x *= multiplier;
            if (IsGridEmpty(numBlocks)) {
                return;
            }

            #define COMPUTE(k)                                                                                              \
             RunComputeHist1NonBinaryKernel<BlockSize, k>(nbFeatures, nbCount, cindex,  target, indices,                    \
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
                CB_ENSURE_INTERNAL(false, "Expected multiplier = 1, 2, 4, 8, 16, 32, or 64, not " << multiplier);
            }
            #undef COMPUTE
        }
    }


}
