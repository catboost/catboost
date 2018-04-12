#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>


namespace NKernel
{

    template<int OUTER_HIST_BITS_COUNT,
             int INNER_HIST_BITS_COUNT,
             int BLOCK_SIZE>
    struct TPointHist {
        float* __restrict__ Buffer;

        float mostRecentStat1[4];
        float mostRecentStat2[4];
        uchar mostRecentBin[4];

        __forceinline__ __device__ int SliceOffset() {

            const int maxBlocks = BLOCK_SIZE * 32 / (1024 << OUTER_HIST_BITS_COUNT);
            static_assert(OUTER_HIST_BITS_COUNT <= 2, "Error: assume 12 warps, so limited by 128-bin histogram per warp");

            const int warpId = (threadIdx.x / 32) % maxBlocks;
            const int warpOffset = (1024 << OUTER_HIST_BITS_COUNT) * warpId;
            const int blocks = 4 >> INNER_HIST_BITS_COUNT;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 3)));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();

            __syncthreads();
            #pragma unroll
            for (int f = 0; f < 4; ++f) {
                mostRecentBin[f] = 0;
                mostRecentStat1[f] = 0;
                mostRecentStat2[f] = 0;
            }
        }

        __forceinline__ __device__ void Add(float val, float* dst) {
            if (OUTER_HIST_BITS_COUNT  > 0 || INNER_HIST_BITS_COUNT > 0) {
                atomicAdd(dst, val);
            } else {
                dst[0] += val;
            }
        }

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t, const float w) {
            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const short f = ((i + threadIdx.x / 2) & 3);
                const uchar bin = bfe(ci, 24 - (f << 3), 8);

                if (bin != mostRecentBin[i]) {
                    const bool pass = (mostRecentBin[i] >> (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT)) == 0;
                    if (pass) {
                        int offset = 2 * f;
                        const uchar mask = (1 << INNER_HIST_BITS_COUNT) - 1;
                        offset += 8 * (mostRecentBin[i] & mask);
                        offset += 32 * ((mostRecentBin[i] >> INNER_HIST_BITS_COUNT));

                        offset += flag;
                        Add(mostRecentStat1[i], Buffer + offset);
                        offset = flag ? offset - 1 : offset + 1;
                        Add(mostRecentStat2[i], Buffer + offset);
                    }

                    mostRecentBin[i] = bin;
                    mostRecentStat1[i] = 0;
                    mostRecentStat2[i] = 0;
                }

                {
                    mostRecentStat1[i] += stat1;
                    mostRecentStat2[i] += stat2;
                }
            }
        }


        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {
            {
                const bool flag = threadIdx.x & 1;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const short f = ((i + threadIdx.x / 2) & 3);
                    const bool pass = (mostRecentBin[i] >> (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT)) == 0;
                    if (pass) {
                        int offset = 2 * f;
                        const uchar mask = (1 << INNER_HIST_BITS_COUNT) - 1;
                        offset += 8 * (mostRecentBin[i] & mask);
                        offset += 32 * ((mostRecentBin[i] >> INNER_HIST_BITS_COUNT));

                        Add(mostRecentStat1[i], Buffer + offset + flag);
                        Add(mostRecentStat2[i], Buffer + offset + !flag);
                    }
                }
            }


            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024 << OUTER_HIST_BITS_COUNT;
                const int maxBlocks = BLOCK_SIZE * 32 / (1024 << OUTER_HIST_BITS_COUNT);

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll maxBlocks
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;

                float sum[4];

                const int maxFoldCount = (1 << (5 + INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT));
                for (int fold = (threadIdx.x >> 1); fold < maxFoldCount; fold += 128) {

                    #pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        sum[f] = 0;
                    }

                    const int innerHistCount = 4 >> INNER_HIST_BITS_COUNT;
                    const int lowBitMask = (1 << INNER_HIST_BITS_COUNT) - 1;
                    const float* __restrict__ src = Buffer
                                                    + (1024 << OUTER_HIST_BITS_COUNT)  //warpHistSize
                                                    + 8 * (fold & lowBitMask)
                                                    + 32 * (fold >> INNER_HIST_BITS_COUNT)
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        #pragma unroll
                        for (int f = 0; f < 4; ++f) {
                            sum[f] += src[2 * f + (inWarpHist << (3 + INNER_HIST_BITS_COUNT))];
                        }
                    }

                    #pragma unroll
                    for (int f = 0; f < 4; ++f) {
                        Buffer[2 * (maxFoldCount * f + fold) + w] = sum[f];
                    }
                }
            }
            __syncthreads();
        }
    };


    template<int BLOCK_SIZE>
    struct TPointHist<0, 0, BLOCK_SIZE> {
        float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 4;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 3));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            __syncthreads();
        }

        __forceinline__ __device__ void Add(float val, float* dst) {
            dst[0] += val;
        }

        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {
            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const short f = ((i + threadIdx.x / 2) & 3);
                const uchar bin = bfe(ci, 24 - (f << 3), 8);
                const bool pass = bin != 32;
                int offset = 2 * f;
                offset += 32 * (bin & 31);
                Buffer[offset + flag] += pass * stat1;
                Buffer[offset + !flag] += pass * stat2;
            }
        }


        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {
            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                float sum = 0.0f;
                const int fold = (threadIdx.x >> 1) & 31;
                const int maxFoldCount = 32;

                if (fold < maxFoldCount) {
                    const int innerHistCount = 4;
                    const float* __restrict__ src = Buffer
                                                    + 1024  //warpHistSize
                                                    + 32 * fold
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum += src[2 * f + (inWarpHist << 3)];
                    }

                    Buffer[2 * (maxFoldCount * f + fold) + w] = sum;
                }
            }
            __syncthreads();
        }
    };


    template<int BLOCK_SIZE>
    struct TPointHist<0, 1, BLOCK_SIZE> {
        volatile float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            const int blocks = 2;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << 4));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff) {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            __syncthreads();
        }


        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {
            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const short f = ((i + threadIdx.x / 2) & 3);
                const uchar bin = bfe(ci, 24 - (f << 3), 8);
                const bool pass = bin != 64;
                int offset = 2 * f;
                offset += 16 * (bin & 62) + 8 * (bin & 1);

                const bool writeFirstFlag = threadIdx.x & 8;

                const float val1 = pass * stat1;

                offset += flag;

                if (writeFirstFlag) {
                    Buffer[offset] += val1;
                }

                if (!writeFirstFlag) {
                    Buffer[offset] += val1;
                }

                const float val2 = pass * stat2;

//                offset -= flag;
//                offset += !flag;
                offset = flag ? offset - 1 : offset + 1;

                if (writeFirstFlag) {
                    Buffer[offset] += val2;
                }

                if (!writeFirstFlag) {
                    Buffer[offset] += val2;
                }
            }
        }


        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {

            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int fold0 = (threadIdx.x >> 1) & 31;

                const int maxFoldCount = 64;

                {
                    const int innerHistCount = 2;
                    const volatile float* __restrict__ src = Buffer
                                                    + 1024  //warpHistSize
                                                    + 8 * (fold0 & 1)
                                                    + 32 * (fold0 >> 1)
                                                    + w;

                    #pragma unroll
                    for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist) {
                        sum0 += src[2 * f + (inWarpHist << 4)];
                        sum1 += src[2 * f + (inWarpHist << 4) + 512];
                    }

                    Buffer[2 * (maxFoldCount * f + fold0) + w] = sum0;
                    Buffer[2 * (maxFoldCount * f + fold0 + 32) + w] = sum1;
                }
            }
            __syncthreads();
        }
    };


    template<int BLOCK_SIZE>
    struct TPointHist<0, 2, BLOCK_SIZE> {
        volatile float* __restrict__ Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpId = (threadIdx.x / 32);
            const int warpOffset = 1024 * warpId;
            return warpOffset;
        }

        __forceinline__ __device__ TPointHist(float* buff) {
            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE) {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            __syncthreads();
        }


        __forceinline__ __device__ void AddPoint(ui32 ci,
                                                 const float t,
                                                 const float w) {
            const bool flag = threadIdx.x & 1;

            const float stat1 = flag ? t : w;
            const float stat2 = flag ? w : t;

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const short f = ((i + threadIdx.x / 2) & 3);
                const int bin = bfe(ci, 24 - (f << 3), 8);
                const bool pass = bin != 128;
                int offset = 2 * f;
                offset += 8 * (bin & 127);
//
                const int writeTime = (threadIdx.x >> 3) & 3;

                const float val1 = pass * stat1;
                offset += flag;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        Buffer[offset] += val1;
                    }
                }

                const float val2 = pass * stat2;
                offset = flag ? offset - 1 : offset + 1;

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if (k == writeTime) {
                        Buffer[offset] += val2;
                    }
                }
            }
        }


        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce() {

            Buffer -= SliceOffset();
            __syncthreads();

            {
                const int warpHistSize = 1024;

                for (int start = threadIdx.x; start < warpHistSize; start += BLOCK_SIZE) {
                    float sum = 0;
//                    12 iterations at 32-bin
                    #pragma unroll 12
                    for (int i = start; i < 32 * BLOCK_SIZE; i += warpHistSize) {
                        sum += Buffer[i];
                    }
                    Buffer[warpHistSize + start] = sum;
                }
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                const int w = threadIdx.x & 1;
                const int f = threadIdx.x / 64;
                const int fold0 = (threadIdx.x >> 1) & 31;

                const int maxFoldCount = 128;

                {
                    const volatile float* __restrict__ src = Buffer
                                                             + 1024  //warpHistSize
                                                             + 2 * f
                                                             + w;

                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        int fold = fold0 + 32 * k;
                        Buffer[2 * (maxFoldCount * f + fold) + w] = src[8 * fold];
                    }
                }
            }
            __syncthreads();
        }
    };


    template<int STRIPE_SIZE, int OUTER_UNROLL, int N, int HIST_BLOCK_COUNT, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram(const ui32* __restrict__ indices, int offset, int dsSize,
                                                     const float* __restrict__ target, const float* __restrict__ weight,
                                                     const ui32* __restrict__ cindex, float* __restrict__ result) {

        weight += offset;
        target += offset;
        indices += offset;

        THist hist(result);

        indices += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        target += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        weight += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        dsSize = max(dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE, 0);
        const int stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE;

        if (dsSize)
        {
            int i = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
            int iteration_count = (dsSize - i + (stripe - 1)) / stripe;
            int blocked_iteration_count = ((dsSize - (i | 31) + (stripe - 1)) / stripe) / N;

            weight += i;
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
                float local_w[N];
                float local_wt[N];

#pragma unroll
                for (int k = 0; k < N; ++k) {
                    local_ci[k] = __ldg(cindex + local_index[k]);
                    local_w[k] = __ldg(weight + stripe * k);
                    local_wt[k] = __ldg(target + stripe * k);
                }

#pragma unroll
                for (int k = 0; k < N; ++k) {
                    hist.AddPoint(local_ci[k], local_wt[k], local_w[k]);
                }

                i += stripe * N;
                indices += stripe * N;
                target += stripe * N;
                weight += stripe * N;
            }

            for (int k = blocked_iteration_count * N; k < iteration_count; ++k) {
                const int index = __ldg(indices);
                ui32 ci = __ldg(cindex + index);
                float w = __ldg(weight);
                float wt = __ldg(target);
                hist.AddPoint(ci, wt, w);
                i += stripe;
                indices += stripe;
                target += stripe;
                weight += stripe;
            }
            __syncthreads();

            hist.Reduce();
        }
    }




    template<int STRIPE_SIZE, int OUTER_UNROLL, int HIST_BLOCK_COUNT, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram2(
            const ui32* __restrict__ indices,
            int offset, int dsSize,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ cindex, float* __restrict__ result) {

        weight += offset;
        target += offset;
        indices += offset;

        THist hist(result);

        if (dsSize) {
            //first: first warp make memory access aligned. it load first 32 - offset % 32 elements.
            {
                int lastId = min(dsSize, 128 - (offset & 127));
                int colId = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;

                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    for (; (colId < lastId); colId += blockDim.x / HIST_BLOCK_COUNT)
                    {
                        const int index = __ldg(indices + colId);
                        const ui32 ci = __ldg(cindex + index);
                        const float w = __ldg(weight + colId);
                        const float wt = __ldg(target + colId);
                        hist.AddPoint(ci, wt, w);
                    }
                }

                dsSize = max(dsSize - lastId, 0);

                indices += lastId;
                target += lastId;
                weight += lastId;
            }

            //now lets align end
            const int unalignedTail = (dsSize & 31);

            if (unalignedTail != 0) {
                if ((blockIdx.x % BLOCKS_PER_FEATURE) == 0)
                {
                    int colId = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
                    const int tailOffset = dsSize - unalignedTail;

                    for (; colId < unalignedTail; colId += blockDim.x / HIST_BLOCK_COUNT)
                    {
                        const int index = __ldg(indices + tailOffset + colId);
                        const ui32 ci = __ldg(cindex + index);
                        const float w = __ldg(weight + tailOffset + colId);
                        const float wt = __ldg(target + tailOffset + colId);
                        hist.AddPoint(ci, wt, w);
                    }
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
            weight += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2;

            const int stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE * 2;
            dsSize = max(dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE * 2, 0);

            if (dsSize) {
                int iterCount;
                {
                    const int i = 2 * ((threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32);
                    weight += i;
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
                    const float2 localWeight = __ldg((float2* )(weight));

                    hist.AddPoint(firstBin, localTarget.x, localWeight.x);
                    hist.AddPoint(secondBin, localTarget.y, localWeight.y);

                    indices += stripe;
                    target += stripe;
                    weight += stripe;
                }
                __syncthreads();
                hist.Reduce();
            }
        }
    }


    template<int BLOCK_SIZE, int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int BLOCKS_PER_FEATURE, bool USE_64_BIT_LOAD>
    __forceinline__ __device__ void ComputeSplitPropertiesPass(const TCFeature* __restrict__ feature, const ui32* __restrict__ cindex,
                                                               const float* __restrict__ target, const float* __restrict__ weight,
                                                               const ui32* __restrict__ indices,
                                                               const TDataPartition* __restrict__ partition, int fCount,
                                                               float* binSumsForPart,
                                                               float* smem)
    {

        using THist = TPointHist<OUTER_HIST_BITS_COUNT, INNER_HIST_BITS_COUNT, BLOCK_SIZE>;
        const int stripeSize = BLOCK_SIZE;
        const int histBlockCount = 1;



       if (USE_64_BIT_LOAD) {
           #if __CUDA_ARCH__ < 300
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 4 : 2;
           const int OUTER_UNROLL = 2;
           #elif __CUDA_ARCH__ <= 350
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 8 : 4;
           const int OUTER_UNROLL = 2;
           #else
           const int INNER_UNROLL = 1;
           const int OUTER_UNROLL =  (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) <= 2 ? 4 : 2;
           #endif
           const int size = partition->Size;
           const int offset = partition->Offset;
           ComputeHistogram2 < stripeSize, OUTER_UNROLL,  histBlockCount, BLOCKS_PER_FEATURE, THist > (indices, offset, size,
                                                                                                       target,
                                                                                                       weight,
                                                                                                       cindex,
                                                                                                       smem);
       } else {
           #if __CUDA_ARCH__ < 300
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 4 : 2;
           const int OUTER_UNROLL = 2;
           #elif __CUDA_ARCH__ <= 350
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 8 : 4;
           const int OUTER_UNROLL = 2;
           #else
           const int INNER_UNROLL = 1;
           const int OUTER_UNROLL = 2;
           #endif
           ComputeHistogram<stripeSize, OUTER_UNROLL, INNER_UNROLL, histBlockCount, BLOCKS_PER_FEATURE, THist>(indices,
                                                                                                               partition->Offset,
                                                                                                               partition->Size,
                                                                                                               target,
                                                                                                               weight,
                                                                                                               cindex,
                                                                                                               smem);
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


#define DECLARE_PASS(O, I, M, USE_64_BIT_LOAD) \
    ComputeSplitPropertiesPass<BLOCK_SIZE, O, I, M, USE_64_BIT_LOAD>(feature, cindex, target, weight, indices, partition, fCount, binSums, &counters[0]);


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ == 600
    __launch_bounds__(BLOCK_SIZE, 1)
#elif __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesNBImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount)
    {
        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);


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
                DECLARE_PASS(0, 0, M, use64BitLoad);
            } else if (maxBinCount <= 64) {
                DECLARE_PASS(0, 1, M, use64BitLoad);
            } else if (maxBinCount <= 128) {
                DECLARE_PASS(0, 2, M, use64BitLoad);
            } else {
                DECLARE_PASS(2, 1, M, use64BitLoad);
            }
        }
    }


    template<int BLOCK_SIZE>
    struct TPointHistHalfByte {
        volatile float* Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart = threadIdx.x & 16;
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

        __forceinline__ __device__ void AddPoint(ui32 ci, const float t, const float w)
        {

            const bool flag = threadIdx.x & 1;

#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                const short f = (threadIdx.x + (i << 1)) & 14;
                short bin = bfe(ci, 28 - (f << 1), 4);
                bin <<= 5;
                bin += f;
                const int offset0 = bin + flag;
                const int offset1 = bin + !flag;
                Buffer[offset0] += (flag ? t : w);
                Buffer[offset1] += (flag ? w : t);
            }
        }

        __device__ void Reduce() {
            Buffer -= SliceOffset();
            const int warpCount = BLOCK_SIZE >> 5;

            {
                const int fold = (threadIdx.x >> 5) & 15;
                const int sumOffset = threadIdx.x & 31;


                float sum = 0.0;
                if (threadIdx.x < 512)
                {
                    float* __restrict__ buffer = const_cast<float*>(Buffer);

                    #pragma unroll
                    for (int warpId = 0; warpId < warpCount; ++warpId)
                    {
                        const int warpOffset = 512 * warpId;
                        sum += buffer[warpOffset + sumOffset + 32 * fold];
                    }
                }
                __syncthreads();

                if (threadIdx.x < 512)
                {
                    Buffer[threadIdx.x] = sum;
                }
            }

            __syncthreads();
            const int fold = (threadIdx.x >> 4) & 15;
            float sum = 0.0f;

            if (threadIdx.x < 256)
            {
                const int histEntryId = (threadIdx.x & 15);
                sum = Buffer[32 * fold + histEntryId] + Buffer[32 * fold + histEntryId + 16];
            }

            __syncthreads();

            if (threadIdx.x < 256) {
                Buffer[threadIdx.x] = sum;
            }

            __syncthreads();
        }
    };


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesBImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__  partition, float* __restrict__ binSums, int totalFeatureCount)
    {

        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 32;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 32, 32);

        __shared__ float counters[16 * BLOCK_SIZE];

        if (partition->Size)
        {
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
                ComputeHistogram2 < BLOCK_SIZE, OUTER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, &counters[0]);
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

                ComputeHistogram < BLOCK_SIZE, OUTER_UNROLL, INNER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, &counters[0]);
            }

            ui32 w = threadIdx.x & 1;
            ui32 fid = (threadIdx.x >> 1);

            if (fid < fCount)
            {
                const int groupId = fid / 4;
                uchar fMask = 1 << (3 - (fid & 3));

                float sum = 0.f;
                #pragma uroll
                for (int i = 0; i < 16; i++)
                {
                    if (!(i & fMask))
                    {
                        sum += counters[i * 16 + 2 * groupId + w];
                    }
                }

                if (abs(sum) > 1e-20f) {
                    if (M > 1)
                    {
                        atomicAdd(binSums + (feature[fid].FirstFoldIndex) * 2 + w, sum);
                    } else
                    {
                        binSums[(feature[fid].FirstFoldIndex) * 2 + w] = sum;
                    }
                }
            }
        }
    }

    template<int BLOCK_SIZE,
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

        if (fullPass)
        {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );

        } else
        {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, false, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );
        }

    }

    inline ui32 EstimateBlockPerFeatureMultiplier(dim3 numBlocks, ui32 dsSize) {
        ui32 multiplier = 1;
        while ((numBlocks.x * numBlocks.y * min(numBlocks.z, 4) * multiplier < TArchProps::SMCount()) &&
               ((dsSize / multiplier) > 10000) && (multiplier < 64)) {
            multiplier *= 2;
        }
        return multiplier;
    }

    void ComputeHist2NonBinary(const TCFeature* nbFeatures, int nbCount,
                               const ui32* cindex,
                               const float* target, const float* weight,
                               const ui32* indices, ui32 size,
                               const TDataPartition* partition, ui32 partCount, ui32 foldCount,
                               float* binSums, const int binFeatureCount,
                               bool fullPass,
                               TCudaStream stream)
    {
        if (nbCount)
        {

            dim3 numBlocks;
            numBlocks.x = (nbCount + 3) / 4;
            const int histPartCount = (fullPass ? partCount : partCount / 2);
            numBlocks.y = histPartCount;
            numBlocks.z = foldCount;
            const int blockSize = 384;
            const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
            numBlocks.x *= multiplier;

            #define COMPUTE(k)\
             RunComputeHist2NonBinaryKernel<blockSize, k>(nbFeatures, nbCount, cindex,  target, weight,  indices, \
                                                          partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
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

            const int scanBlockSize = 256;
            dim3 scanBlocks;
            scanBlocks.x = (nbCount * 32 + scanBlockSize - 1) / scanBlockSize;
            scanBlocks.y = histPartCount;
            scanBlocks.z = foldCount;
            const int scanOffset = fullPass ? 0 : ((partCount / 2) * binFeatureCount * 2) * foldCount;
            ScanHistogramsImpl<scanBlockSize, 2> << < scanBlocks, scanBlockSize, 0, stream >> > (nbFeatures, nbCount, binFeatureCount, binSums +
                                                                                                 scanOffset);
            if (!fullPass)
            {
                UpdatePointwiseHistograms(binSums, binFeatureCount, partCount, foldCount, 2, partition, stream);
            }
        }
    }

    template<int BLOCK_SIZE, int BLOCKS_PER_FEATURE_COUNT>
    void RunComputeHist2BinaryKernel(const TCFeature* bFeatures, int bCount,
                                     const ui32* cindex,
                                     const float* target, const float* weight, const ui32* indices,
                                     const TDataPartition* partition,
                                     float* binSums, bool fullPass,
                                     TCudaStream stream,
                                     dim3 numBlocks)
    {
        if (fullPass)
        {
            ComputeSplitPropertiesBImpl < BLOCK_SIZE, true,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    bFeatures, bCount, cindex, target, weight, indices, partition, binSums, bCount
            );
        } else
        {
            ComputeSplitPropertiesBImpl < BLOCK_SIZE, false,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    bFeatures, bCount, cindex, target, weight, indices, partition, binSums, bCount
            );
        }
    };

    void ComputeHist2Binary(const TCFeature* bFeatures, int bCount,
                            const ui32* cindex,
                            const float* target, const float* weight,
                            const ui32* indices, ui32 size,
                            const TDataPartition* partition, ui32 partsCount, ui32 foldCount,
                            float* binSums, bool fullPass,
                            TCudaStream stream) {
        dim3 numBlocks;
        numBlocks.x = (bCount + 31) / 32;
        const int histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = histCount;

        numBlocks.z = foldCount;

        const int blockSize = 768;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
        numBlocks.x *= multiplier;

        if (bCount)
        {

            #define COMPUTE(k)  \
            RunComputeHist2BinaryKernel<blockSize, k>(bFeatures, bCount, cindex, target, weight, indices, \
                                                      partition, binSums, fullPass, stream, numBlocks); \

            if (multiplier == 1)
            {
                COMPUTE(1)
            } else if (multiplier == 2)
            {
                COMPUTE(2)
            } else if (multiplier == 4)
            {
                COMPUTE(4)
            } else if (multiplier == 8)
            {
                COMPUTE(8);
            } else if (multiplier == 16)
            {
                COMPUTE(16)
            } else if (multiplier == 32)
            {
                COMPUTE(32)
            } else if (multiplier == 64) {
                COMPUTE(64)
            } else {
                exit(1);
            }

            #undef COMPUTE

            if (!fullPass)
            {
                UpdatePointwiseHistograms(binSums, bCount, partsCount, foldCount, 2, partition, stream);
            }
        }
    }


    template<int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesHalfByteImpl(
            const TCFeature* __restrict__ feature, int fCount, const ui32* __restrict__ cindex,
            const float* __restrict__ target, const float* __restrict__ weight,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount)
    {


        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 8;
        cindex += feature->Offset;
        fCount = min(fCount - (blockIdx.x / M) * 8, 8);

//
        __shared__ float smem[16 * BLOCK_SIZE];


        using THist = TPointHistHalfByte<BLOCK_SIZE>;


        #if __CUDA_ARCH__ > 350
            const bool use64BitLoad = FULL_PASS;
        #else
            const bool use64BitLoad = false;
        #endif

        if (use64BitLoad)
        {
            #if __CUDA_ARCH__ <= 350
            const int INNER_UNROLL = 4;
            const int OUTER_UNROLL = 2;
            #else
            const int INNER_UNROLL = 1;
            const int OUTER_UNROLL = 1;
            #endif
            ComputeHistogram2 < BLOCK_SIZE, OUTER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, smem);
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

            ComputeHistogram < BLOCK_SIZE, OUTER_UNROLL, INNER_UNROLL, 1, M, THist > (
                    indices, partition->Offset, partition->Size,
                            target, weight,
                            cindex, smem);
        }

        __syncthreads();

        const int fid = (threadIdx.x / 32);
        const int fold = (threadIdx.x / 2) & 15;
        const int w = threadIdx.x & 1;


        if (fid < fCount && fold < feature[fid].Folds)
        {
            const float result = smem[fold * 16 + 2 * fid + w];
            if (abs(result) > 1e-20)
            {
                if (M > 1)
                {
                    atomicAdd(binSums + (feature[fid].FirstFoldIndex + fold) * 2 + w, result);
                } else
                {
                    binSums[(feature[fid].FirstFoldIndex + fold) * 2 + w] = result;
                }
            }
        }
    }


    template<int BLOCK_SIZE,
            int BLOCKS_PER_FEATURE_COUNT>
    inline void RunComputeHist2HalfByteKernel(const TCFeature* nbFeatures, int nbCount,
                                              const ui32* cindex,
                                              const float* target, const float* weight, const ui32* indices,
                                              const TDataPartition* partition,
                                              float* binSums, const int binFeatureCount,
                                              bool fullPass,
                                              TCudaStream stream,
                                              dim3 numBlocks)
    {

        if (fullPass)
        {
            ComputeSplitPropertiesHalfByteImpl < BLOCK_SIZE, true,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount
            );

        } else
        {
            ComputeSplitPropertiesHalfByteImpl < BLOCK_SIZE, false,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight,
                            indices, partition, binSums, binFeatureCount);
        }

    }

    void ComputeHist2HalfByte(const TCFeature* halfByteFeatures, int halfByteFeaturesCount,
                              const ui32* cindex,
                              const float* target, const float* weight, const ui32* indices,
                              ui32 size,
                              const TDataPartition* partition, ui32 partsCount, ui32 foldCount,
                              float* binSums, const int binFeatureCount,
                              bool fullPass,
                              TCudaStream stream)
    {
        dim3 numBlocks;
        numBlocks.x = static_cast<ui32>((halfByteFeaturesCount + 7) / 8);
        const int histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = static_cast<ui32>(histCount);
        numBlocks.z = foldCount;

        const int blockSize = 768;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, size), 64);
        numBlocks.x *= multiplier;

        if (halfByteFeaturesCount)
        {

            #define COMPUTE(k)\
            RunComputeHist2HalfByteKernel<blockSize, k>(halfByteFeatures, halfByteFeaturesCount, cindex,\
                                                        target,\
                                                        weight, indices, partition, binSums, binFeatureCount,\
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

            const int scanBlockSize = 256;
            dim3 scanBlocks;
            scanBlocks.x = static_cast<ui32>((halfByteFeaturesCount * 32 + scanBlockSize - 1) / scanBlockSize);
            scanBlocks.y = static_cast<ui32>(histCount);
            scanBlocks.z = foldCount;
            const int scanOffset = fullPass ? 0 : ((partsCount / 2) * binFeatureCount * 2) * foldCount;
            ScanHistogramsImpl<scanBlockSize, 2> << < scanBlocks, scanBlockSize, 0, stream >> >
                                                          (halfByteFeatures, halfByteFeaturesCount, binFeatureCount,
                                                                  binSums + scanOffset);

            if (!fullPass) {
                UpdatePointwiseHistograms(binSums, binFeatureCount, partsCount, foldCount, 2, partition, stream);
            }
        }
    }


    __global__ void UpdateBinsImpl(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                                   ui32 loadBit, ui32 foldBits)
    {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size)
        {
            const ui32 idx = LdgWithFallback(docIndices, i);
            const ui32 bit = (LdgWithFallback(bins, idx) >> loadBit) & 1;
            dstBins[i] = dstBins[i] | (bit << (loadBit + foldBits));
        }
    }

    void UpdateFoldBins(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                        ui32 loadBit, ui32 foldBits, TCudaStream stream)
    {


        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        UpdateBinsImpl << < numBlocks, blockSize, 0, stream >> > (dstBins, bins, docIndices, size, loadBit, foldBits);
    }
}
