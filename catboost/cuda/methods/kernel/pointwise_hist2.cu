#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>


namespace NKernel
{

    template<int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int BLOCK_SIZE>
    struct TPointHist
    {
        volatile float* __restrict__ Buffer;
        int BlockId;

        __forceinline__ __device__ int SliceOffset()
        {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            const int blocks = 4 >> INNER_HIST_BITS_COUNT;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 3)));
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHist(float* buff)
        {

            const int HIST_SIZE = 32 * BLOCK_SIZE;

            #pragma unroll 8
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE)
            {
                buff[i] = 0;
            }

            Buffer = buff + SliceOffset();
            BlockId = (threadIdx.x / 32) & ((1 << OUTER_HIST_BITS_COUNT) - 1);


            __syncthreads();
        }

        __device__ void AddPoint(ui32 ci, const float t, const float w)
        {
            const bool flag = threadIdx.x & 1;


#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                short f = (threadIdx.x + (i << 1)) & 6;
                short bin = bfe(ci, 24 - (f << 2), 8);
                bool pass = (bin >> (5 + INNER_HIST_BITS_COUNT)) == BlockId;
                int offset0 = f + flag;
                int offset1 = f + !flag;

                const int mask = (1 << INNER_HIST_BITS_COUNT) - 1;

                const int tmp = (((bin >> INNER_HIST_BITS_COUNT) & 31) << 5) + 8 * (bin & mask);
                offset0 += tmp;
                offset1 += tmp;

                if (INNER_HIST_BITS_COUNT > 0)
                {
#pragma unroll
                    for (int k = 0; k < (1 << INNER_HIST_BITS_COUNT); ++k)
                    {
                        if (((threadIdx.x >> 3) & ((1 << INNER_HIST_BITS_COUNT) - 1)) == k)
                        {
                            Buffer[offset0] += (flag ? t : w) * pass;
                            Buffer[offset1] += (flag ? w : t) * pass;
                        }
                    }
                } else {
                    Buffer[offset0] += (flag ? t : w) * pass;
                    Buffer[offset1] += (flag ? w : t) * pass;
                }
            }
        }

        //After reduce we store histograms by blocks: 256 floats (4 x 2 x 32)
        // for first 32 bins; than 256 floats for second 32 bins, etc
        __forceinline__ __device__ void Reduce()
        {

            Buffer -= SliceOffset();
            const int warpCount = BLOCK_SIZE >> 5;
            const int innerHistCount = 4 >> INNER_HIST_BITS_COUNT;
            const int warpHistCount = warpCount >> OUTER_HIST_BITS_COUNT;
            const int fold = (threadIdx.x >> 3) & 31;

            const int mask = (1 << INNER_HIST_BITS_COUNT) - 1;
            const int binOffset = ((fold >> INNER_HIST_BITS_COUNT) << 5) + 8 * (fold & mask);
            const int offset = (threadIdx.x & 7) + binOffset;


            const float* __restrict__ buffer = const_cast<float*>(Buffer);

#pragma unroll
            for (int outerBits = 0; outerBits < 1 << (OUTER_HIST_BITS_COUNT); ++outerBits)
            {
#pragma unroll
                for (int innerBits = 0; innerBits < (1 << (INNER_HIST_BITS_COUNT)); ++innerBits)
                {
                    float sum = 0.0f;
                    const int innerOffset = innerBits << (10 - INNER_HIST_BITS_COUNT);
                    const int tmp = innerOffset + offset;

                    {

#pragma unroll
                        for (int hist = 0; hist < warpHistCount; ++hist)
                        {
                            const int warpOffset = ((hist << OUTER_HIST_BITS_COUNT) + outerBits) * 1024;
                            const int tmp2 = tmp + warpOffset;
#pragma unroll
                            for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist)
                            {
                                sum += buffer[tmp2 + (inWarpHist << (3 + INNER_HIST_BITS_COUNT))];
                            }
                        }
                    }
                    __syncthreads();

                    if (threadIdx.x < 256)
                    {
                        Buffer[threadIdx.x + 256 * (innerBits | (outerBits << INNER_HIST_BITS_COUNT))] = sum;
                    }
                }
            }
            __syncthreads();
        }
    };


    template<int STRIPE_SIZE, int OUTER_UNROLL, int N, int HIST_BLOCK_COUNT, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram(
            const ui32* __restrict__ indices,
            int offset, int dsSize,
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
            for (int j = 0; j < blocked_iteration_count; ++j)
            {
                ui32 local_index[N];
#pragma unroll
                for (int k = 0; k < N; k++)
                {
                    local_index[k] = __ldg(indices + stripe * k);
                }

                ui32 local_ci[N];
                float local_w[N];
                float local_wt[N];

#pragma unroll
                for (int k = 0; k < N; ++k)
                {
                    local_ci[k] = __ldg(cindex + local_index[k]);
                    local_w[k] = __ldg(weight + stripe * k);
                    local_wt[k] = __ldg(target + stripe * k);
                }

#pragma unroll
                for (int k = 0; k < N; ++k)
                {
                    hist.AddPoint(local_ci[k], local_wt[k], local_w[k]);
                }

                i += stripe * N;
                indices += stripe * N;
                target += stripe * N;
                weight += stripe * N;
            }

            for (int k = blocked_iteration_count * N; k < iteration_count; ++k)
            {
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

        if (dsSize)
        {
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
        const int stripeSize = (BLOCK_SIZE >> OUTER_HIST_BITS_COUNT);
        const int histBlockCount = 1 << OUTER_HIST_BITS_COUNT;



       if (USE_64_BIT_LOAD)
       {
           #if __CUDA_ARCH__ <= 350
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 8 : 4;
           const int OUTER_UNROLL = 2;
           #else
           const int INNER_UNROLL = 1;
           const int OUTER_UNROLL =  (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 4 : 1;
           #endif
           const int size = partition->Size;
           const int offset = partition->Offset;
           ComputeHistogram2 < stripeSize, OUTER_UNROLL,  histBlockCount, BLOCKS_PER_FEATURE, THist > (indices, offset, size,
                                                                                                       target,
                                                                                                       weight,
                                                                                                       cindex,
                                                                                                       smem);
       }
       else {
           #if __CUDA_ARCH__ <= 350
           const int INNER_UNROLL = (INNER_HIST_BITS_COUNT + OUTER_HIST_BITS_COUNT) == 0 ? 8 : 4;
           const int OUTER_UNROLL = 2;
           #else
           const int INNER_UNROLL = 1;
           const int OUTER_UNROLL = 1;
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

        int fid = (threadIdx.x / 64);
        int fold = (threadIdx.x / 2) & 31;

        #pragma unroll
        for (int upperBits = 0; upperBits < (1 << (OUTER_HIST_BITS_COUNT + INNER_HIST_BITS_COUNT)); ++upperBits)
        {
            const int binOffset = upperBits << 5;

            if (fid < fCount && fold < min((int) feature[fid].Folds - binOffset, 32))
            {
                int w = threadIdx.x & 1;
                const float val = smem[fold * 8 + 2 * fid + w + 256 * upperBits];
                if (abs(val) > 1e-20f)
                {
                    if (BLOCKS_PER_FEATURE > 1)
                    {
                        atomicAdd(binSumsForPart + (feature[fid].FirstFoldIndex + fold + binOffset) * 2 + w, val);
                    } else
                    {
                        WriteThrough(binSumsForPart + (feature[fid].FirstFoldIndex + fold + binOffset) * 2 + w, val);
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
            const float* __restrict__ target, const float* __restrict__ weight, int dsSize,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount)
    {
        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);


        feature += (blockIdx.x / M) * 4;
        cindex += feature->Offset * ((size_t) dsSize);
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
                DECLARE_PASS(0, 1, M, false);
            } else if (maxBinCount <= 128) {
                DECLARE_PASS(0, 2, M, false);
            } else {
                DECLARE_PASS(1, 2, M, false);
            }
        }
    }


    template<int BLOCK_SIZE>
    struct TPointHistHalfByte
    {
        volatile float* Buffer;

        __forceinline__ __device__ int SliceOffset()
        {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart = threadIdx.x & 16;
            return warpOffset + innerHistStart;
        }

        __forceinline__ __device__ TPointHistHalfByte(float* buff)
        {
            const int HIST_SIZE = 16 * BLOCK_SIZE;
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE)
            {
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

        __device__ void Reduce()
        {
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

            if (threadIdx.x < 256)
            {
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
            const float* __restrict__ target, const float* __restrict__ weight, int dsSize,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__  partition, float* __restrict__ binSums, int totalFeatureCount)
    {

        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 32;
        cindex += feature->Offset * ((size_t) dsSize);
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
                const int OUTER_UNROLL = 2;
                #endif
                ComputeHistogram2 < BLOCK_SIZE, OUTER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, &counters[0]);
            } else {
                #if __CUDA_ARCH__ <= 350
                const int INNER_UNROLL = 4;
                const int OUTER_UNROLL = 1;
                #else
                const int INNER_UNROLL = 1;
                const int OUTER_UNROLL = 1;
                #endif

                ComputeHistogram < BLOCK_SIZE, OUTER_UNROLL, INNER_UNROLL, 1, M, THist > (
                        indices,
                                partition->Offset, partition->Size,
                                target, weight,
                                cindex,
                                &counters[0]);
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

                if (abs(sum) > 1e-20f)
                {
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
                                               const ui32* cindex, int dsSize,
                                               const float* target, const float* weight, const ui32* indices,
                                               const TDataPartition* partition,
                                               float* binSums, const int binFeatureCount,
                                               bool fullPass,
                                               TCudaStream stream,
                                               dim3 numBlocks)
    {

        if (fullPass)
        {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, true,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, binFeatureCount
            );

        } else
        {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, false,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, binFeatureCount
            );
        }

    }

    inline ui32 EstimateBlockPerFeatureMultiplier(dim3 numBlocks, ui32 dsSize)
    {
        ui32 multiplier = 1;
        while ((numBlocks.x * numBlocks.y * min(numBlocks.z, 4) * multiplier < TArchProps::SMCount()) &&
               ((dsSize / multiplier) > 10000) && (multiplier < 64))
        {
            multiplier *= 2;
        }
        return multiplier;
    }

    void ComputeHist2NonBinary(const TCFeature* nbFeatures, int nbCount,
                               const ui32* cindex, int dsSize,
                               const float* target, const float* weight, const ui32* indices,
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
            const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, dsSize), 64);
            numBlocks.x *= multiplier;

            if (multiplier == 1)
            {
                RunComputeHist2NonBinaryKernel<blockSize, 1>(nbFeatures, nbCount, cindex, dsSize, target, weight,
                                                             indices,
                                                             partition, binSums, binFeatureCount, fullPass, stream,
                                                             numBlocks);
            } else if (multiplier == 2) {
                RunComputeHist2NonBinaryKernel<blockSize, 2>(nbFeatures, nbCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 4) {
                RunComputeHist2NonBinaryKernel<blockSize, 4>(nbFeatures, nbCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 8) {
                RunComputeHist2NonBinaryKernel<blockSize, 8>(nbFeatures, nbCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 16) {
                RunComputeHist2NonBinaryKernel<blockSize, 16>(nbFeatures, nbCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 32) {
                RunComputeHist2NonBinaryKernel<blockSize, 32>(nbFeatures, nbCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 64) {
                RunComputeHist2NonBinaryKernel<blockSize, 64>(nbFeatures, nbCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else
            {
                exit(1);
            }

            const int scanBlockSize = 256;
            dim3 scanBlocks;
            scanBlocks.x = (nbCount * 32 + scanBlockSize - 1) / scanBlockSize;
            scanBlocks.y = histPartCount;
            scanBlocks.z = foldCount;
            const int scanOffset = fullPass ? 0 : ((partCount / 2) * binFeatureCount * 2) * foldCount;
            ScanHistogramsImpl<scanBlockSize,
                    2> << < scanBlocks, scanBlockSize, 0, stream >> > (nbFeatures, nbCount, binFeatureCount, binSums +
                                                                                                             scanOffset);
            if (!fullPass)
            {
                UpdatePointwiseHistograms(binSums, binFeatureCount, partCount, foldCount, 2, partition, stream);
            }
        }
    }

    template<int BLOCK_SIZE, int BLOCKS_PER_FEATURE_COUNT>
    void RunComputeHist2BinaryKernel(const TCFeature* bFeatures, int bCount,
                                     const ui32* cindex, int dsSize,
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
                    bFeatures, bCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, bCount
            );
        } else
        {
            ComputeSplitPropertiesBImpl < BLOCK_SIZE, false,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    bFeatures, bCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, bCount
            );
        }
    };

    void ComputeHist2Binary(const TCFeature* bFeatures, int bCount,
                            const ui32* cindex, int dsSize,
                            const float* target, const float* weight, const ui32* indices,
                            const TDataPartition* partition, ui32 partsCount, ui32 foldCount,
                            float* binSums, bool fullPass,
                            TCudaStream stream)
    {
        dim3 numBlocks;
        numBlocks.x = (bCount + 31) / 32;
        const int histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = histCount;

        numBlocks.z = foldCount;

        const int blockSize = 768;
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, dsSize), 64);
        numBlocks.x *= multiplier;

        if (bCount)
        {

            if (multiplier == 1)
            {
                RunComputeHist2BinaryKernel<blockSize, 1>(bFeatures, bCount, cindex, dsSize, target, weight, indices,
                                                          partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 2)
            {
                RunComputeHist2BinaryKernel<blockSize, 2>(bFeatures, bCount, cindex, dsSize, target, weight, indices,
                                                          partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 4)
            {
                RunComputeHist2BinaryKernel<blockSize, 4>(bFeatures, bCount, cindex, dsSize, target, weight, indices,
                                                          partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 8)
            {
                RunComputeHist2BinaryKernel<blockSize, 8>(bFeatures, bCount, cindex, dsSize, target, weight, indices,
                                                          partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 16)
            {
                RunComputeHist2BinaryKernel<blockSize, 16>(bFeatures, bCount, cindex, dsSize, target, weight, indices,
                                                           partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 32)
            {
                RunComputeHist2BinaryKernel<blockSize, 32>(bFeatures, bCount, cindex, dsSize, target, weight, indices,
                                                           partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 64)
            {
                RunComputeHist2BinaryKernel<blockSize, 64>(bFeatures, bCount, cindex, dsSize, target, weight, indices,
                                                           partition, binSums, fullPass, stream, numBlocks);
            } else
            {
                exit(1);
            }

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
            const float* __restrict__ target, const float* __restrict__ weight, int dsSize,
            const ui32* __restrict__ indices,
            const TDataPartition* __restrict__ partition,
            float* __restrict__ binSums,
            const int totalFeatureCount)
    {


        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 8;
        cindex += feature->Offset * ((size_t) dsSize);
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
            const int OUTER_UNROLL = 2;
            #endif
            ComputeHistogram2 < BLOCK_SIZE, OUTER_UNROLL, 1, M, THist > (indices, partition->Offset, partition->Size, target, weight, cindex, smem);
        } else {

            #if __CUDA_ARCH__ <= 350
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
                                              const ui32* cindex, int dsSize,
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
                    nbFeatures, nbCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, binFeatureCount
            );

        } else
        {
            ComputeSplitPropertiesHalfByteImpl < BLOCK_SIZE, false,
                    BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, binFeatureCount);
        }

    }

    void ComputeHist2HalfByte(const TCFeature* halfByteFeatures, int halfByteFeaturesCount,
                              const ui32* cindex, int dsSize,
                              const float* target, const float* weight, const ui32* indices,
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
        const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, dsSize), 64);
        numBlocks.x *= multiplier;

        if (halfByteFeaturesCount)
        {

            if (multiplier == 1)
            {
                RunComputeHist2HalfByteKernel<blockSize, 1>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize,
                                                            target,
                                                            weight, indices, partition, binSums, binFeatureCount,
                                                            fullPass,
                                                            stream, numBlocks);
            } else if (multiplier == 2)
            {
                RunComputeHist2HalfByteKernel<blockSize, 2>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize,
                                                            target,
                                                            weight, indices, partition, binSums, binFeatureCount,
                                                            fullPass,
                                                            stream, numBlocks);
            } else if (multiplier == 4)
            {
                RunComputeHist2HalfByteKernel<blockSize, 4>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize,
                                                            target,
                                                            weight, indices, partition, binSums, binFeatureCount,
                                                            fullPass,
                                                            stream, numBlocks);
            } else if (multiplier == 8)
            {
                RunComputeHist2HalfByteKernel<blockSize, 8>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize,
                                                            target,
                                                            weight, indices, partition, binSums, binFeatureCount,
                                                            fullPass,
                                                            stream, numBlocks);
            } else if (multiplier == 16)
            {
                RunComputeHist2HalfByteKernel<blockSize, 16>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize,
                                                             target, weight, indices, partition, binSums,
                                                             binFeatureCount,
                                                             fullPass, stream, numBlocks);
            } else if (multiplier == 32)
            {
                RunComputeHist2HalfByteKernel<blockSize, 32>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize,
                                                             target, weight, indices, partition, binSums,
                                                             binFeatureCount,
                                                             fullPass, stream, numBlocks);
            } else if (multiplier == 64)
            {
                RunComputeHist2HalfByteKernel<blockSize, 64>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize,
                                                             target, weight, indices, partition, binSums,
                                                             binFeatureCount,
                                                             fullPass, stream, numBlocks);
            } else
            {
                exit(1);
            }

            const int scanBlockSize = 256;
            dim3 scanBlocks;
            scanBlocks.x = static_cast<ui32>((halfByteFeaturesCount * 32 + scanBlockSize - 1) / scanBlockSize);
            scanBlocks.y = static_cast<ui32>(histCount);
            scanBlocks.z = foldCount;
            const int scanOffset = fullPass ? 0 : ((partsCount / 2) * binFeatureCount * 2) * foldCount;
            ScanHistogramsImpl<scanBlockSize, 2> << < scanBlocks, scanBlockSize, 0, stream >> >
                                                          (halfByteFeatures, halfByteFeaturesCount, binFeatureCount,
                                                                  binSums + scanOffset);

            if (!fullPass)
            {
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
