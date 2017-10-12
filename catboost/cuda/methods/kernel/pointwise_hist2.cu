#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <cstdlib>


namespace NKernel {

    template <int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int BLOCK_SIZE>
    struct TPointHist {
        volatile float* Buffer;
        int BlockId;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 1024 * (threadIdx.x / 32);
            const int blocks = 4  >> INNER_HIST_BITS_COUNT;
            const int innerHistStart = (threadIdx.x & ((blocks - 1) << (INNER_HIST_BITS_COUNT + 3)));
            return warpOffset + innerHistStart;
        }

        __device__ TPointHist(float* buff)
        {
            const int HIST_SIZE = 32 * BLOCK_SIZE;
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE)
                buff[i] = 0;
            __syncthreads();

            Buffer = buff + SliceOffset();
            BlockId = (threadIdx.x / 32) & ((1 << OUTER_HIST_BITS_COUNT) - 1);
        }

        __device__ void AddPoint(ui32 ci, const float t, const float w) {
            const bool flag = threadIdx.x & 1;

#pragma unroll
            for (int i = 0; i < 4; i++) {
                short f = (threadIdx.x + (i << 1)) & 6;
                short bin = bfe(ci, 24 - (f << 2), 8);
                short pass = (bin >> (5 + INNER_HIST_BITS_COUNT)) == BlockId;
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
        __device__ void Reduce()
        {

            Buffer -= SliceOffset();

            const int innerHistCount = 4 >> INNER_HIST_BITS_COUNT;
            const int warpCount = BLOCK_SIZE >> 5;
            const int warpHistCount = warpCount >> OUTER_HIST_BITS_COUNT;
            const int fold = (threadIdx.x >> 3) & 31;

            const int mask = (1 << INNER_HIST_BITS_COUNT) - 1;
            const int binOffset = ((fold >> INNER_HIST_BITS_COUNT) << 5) + 8 * (fold & mask);
            const int offset = (threadIdx.x & 7) + binOffset;


#pragma unroll
            for (int outerBits = 0; outerBits < 1 << (OUTER_HIST_BITS_COUNT); ++outerBits)
            {
                for (int innerBits = 0; innerBits < (1 << (INNER_HIST_BITS_COUNT)); ++innerBits)
                {
                    float sum = 0.0;

                    const int innerOffset = innerBits << (10 - INNER_HIST_BITS_COUNT);
                    if (threadIdx.x < 256)
                    {
#pragma unroll
                        for (int hist = 0; hist < warpHistCount; ++hist)
                        {
                            const int warpOffset = ((hist << OUTER_HIST_BITS_COUNT) + outerBits) * 1024;

#pragma unroll
                            for (int inWarpHist = 0; inWarpHist < innerHistCount; ++inWarpHist)
                            {
                                sum += Buffer[offset + warpOffset + innerOffset +
                                              (inWarpHist << (3 + INNER_HIST_BITS_COUNT))];
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

    template <int STRIPE_SIZE, int HIST_BLOCK_COUNT, int N, int BLOCKS_PER_FEATURE, typename THist>
    __forceinline__ __device__ void ComputeHistogram(
            const ui32* __restrict indices, int dsSize,
            const float* __restrict target, const float* __restrict weight,
            const ui32* __restrict cindex, float* result)
    {

        indices += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        target += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        weight += (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE;
        dsSize = max(dsSize - (blockIdx.x % BLOCKS_PER_FEATURE) * STRIPE_SIZE, 0);
        const int stripe = STRIPE_SIZE * BLOCKS_PER_FEATURE;

        THist hist(result);
        if (dsSize)
        {
            int i = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
            int iteration_count = (dsSize - i + (stripe - 1)) / stripe;
            int blocked_iteration_count = ((dsSize - (i | 31) + (stripe - 1)) / stripe) / N;

            weight += i;
            target += i;
            indices += i;

#pragma unroll 4
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



    template <int BLOCK_SIZE, int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int N, int BLOCKS_PER_FEATURE>
    __forceinline__ __device__ void ComputeSplitPropertiesPass(const TCFeature* __restrict feature, const ui32* __restrict cindex,
                                                               const float* __restrict target, const float* __restrict weight, const ui32* __restrict indices,
                                                               const TDataPartition* __restrict partition, int fCount,
                                                               float* binSumsForPart,
                                                               float* smem) {

        using THist = TPointHist < OUTER_HIST_BITS_COUNT, INNER_HIST_BITS_COUNT, BLOCK_SIZE >;
        const int stripeSize = BLOCK_SIZE >> OUTER_HIST_BITS_COUNT;
        const int histBlockCount =  1 << OUTER_HIST_BITS_COUNT;

        ComputeHistogram<stripeSize, histBlockCount, N,  BLOCKS_PER_FEATURE, THist >(indices + partition->Offset,
                partition->Size, target + partition->Offset, weight + partition->Offset, cindex, smem);

        __syncthreads();



        int fid = (threadIdx.x / 64);
        int fold = (threadIdx.x / 2) & 31;


        for (int upperBits = 0; upperBits < (1 << (OUTER_HIST_BITS_COUNT + INNER_HIST_BITS_COUNT)); ++upperBits) {
            const int binOffset = upperBits << 5;

            if (fid < fCount && fold < min((int)feature[fid].Folds - binOffset, 32)) {
                int w = threadIdx.x & 1;
                if (BLOCKS_PER_FEATURE > 1) {
                    atomicAdd(binSumsForPart + (feature[fid].FirstFoldIndex + fold + binOffset) * 2 + w, smem[fold * 8 + 2 * fid + w + 256 * upperBits]);
                } else {
                    binSumsForPart[(feature[fid].FirstFoldIndex + fold + binOffset) * 2 + w] = smem[fold * 8 + 2 * fid + w + 256 * upperBits];
                }
            }
        }

        __syncthreads();


    }



#define DECLARE_PASS(O, I, N, M) \
    ComputeSplitPropertiesPass<BLOCK_SIZE, O, I, N, M>(feature, cindex, target, weight, indices, partition, fCount, binSums, &counters[0]);


    template <int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesNBImpl(
            const TCFeature* __restrict feature, int fCount, const ui32* __restrict cindex,
            const float* __restrict target, const float* __restrict weight, int dsSize,
            const ui32* __restrict indices,
            const TDataPartition* partition,
            float* binSums,
            const int totalFeatureCount) {


        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 4;
        cindex += feature->Offset * ((size_t)dsSize);
        fCount = min(fCount - (blockIdx.x / M)  * 4, 4);

//
        __shared__ float counters[32 * BLOCK_SIZE];
        const int maxBinCount = GetMaxBinCount(feature, fCount, (int*) &counters[0]);
        __syncthreads();


        if (partition->Size) {
            if (maxBinCount <= 32) {
                DECLARE_PASS(0, 0, 8,  M);
            }
            else if (maxBinCount <= 64) {
                DECLARE_PASS(0, 1, 4, M);
            } else if (maxBinCount <= 128) {
                DECLARE_PASS(0, 2, 4, M);
            } else {
                DECLARE_PASS(1, 2, 4, M);
            }
        }
    }



    template <int BLOCK_SIZE>
    struct TPointHistHalfByte {
        volatile float* Buffer;

        __forceinline__ __device__ int SliceOffset() {
            const int warpOffset = 512 * (threadIdx.x / 32);
            const int innerHistStart =  threadIdx.x & 16;
            return warpOffset + innerHistStart;
        }

        __device__  TPointHistHalfByte(float* buff)
        {
            const int HIST_SIZE = 16 * BLOCK_SIZE;
            for (int i = threadIdx.x; i < HIST_SIZE; i += BLOCK_SIZE)
                buff[i] = 0;
            __syncthreads();

            Buffer = buff + SliceOffset();
        }

        __device__ void AddPoint(ui32 ci, const float t, const float w) {

            const bool flag = threadIdx.x & 1;

#pragma unroll
            for (int i = 0; i < 8; i++) {
                const short f = (threadIdx.x + (i << 1)) & 14;
                short bin = bfe(ci, 28 - (f << 1), 4);
                bin <<= 5;
                bin += f;
                const int offset0 = bin + flag;
                const int offset1 = bin + !flag;
                Buffer[offset0] += flag ? t : w;
                Buffer[offset1] += flag ? w : t;
            }
        }

        __device__ void Reduce()
        {
            Buffer -= SliceOffset();

            {
                const int warpCount = BLOCK_SIZE >> 5;
                const int fold = (threadIdx.x >> 5) & 15;
                const int sumOffset = threadIdx.x & 31;


                float sum = 0.0;
                if (threadIdx.x < 512)
                {
                    #pragma unroll
                    for (int warpId = 0; warpId < warpCount; ++warpId)
                    {
                        const int warpOffset = 512 * warpId;
                        sum += Buffer[warpOffset + sumOffset + 32 * fold];
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

            if (threadIdx.x < 256) {
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

    inline constexpr __device__ __host__ int GetUnrollFactorHalfByteHist() {
        #if __CUDA_ARCH__ >= 600
        return 1;
        #elif __CUDA_ARCH__ >= 520
        return 2;
        #else
        return 4;
        #endif
    }

    template <int BLOCK_SIZE,  bool FULL_PASS, int M>
    #if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
    #else
    __launch_bounds__(BLOCK_SIZE, 1)
    #endif
    __global__ void ComputeSplitPropertiesBImpl(
            const TCFeature* __restrict feature, int fCount, const ui32* __restrict cindex,
            const float* __restrict target, const float* __restrict weight, int dsSize, const ui32* __restrict indices,
            const TDataPartition* partition, float* binSums, int totalFeatureCount)
    {

        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 32;
        cindex += feature->Offset * ((size_t)dsSize);
        fCount = min(fCount - (blockIdx.x / M) * 32, 32);

        __shared__ float counters[16 * BLOCK_SIZE];

        if (partition->Size)
        {

            using THist = TPointHistHalfByte<BLOCK_SIZE>;

            ComputeHistogram < BLOCK_SIZE, 1, GetUnrollFactorHalfByteHist(), M, THist > (indices + partition->Offset,
                    partition->Size, target + partition->Offset, weight + partition->Offset, cindex, &counters[0]);

            ui32 w = threadIdx.x & 1;
            ui32 fid = (threadIdx.x >> 1);

            if (fid < fCount)
            {
                const int groupId = fid / 4;
                uchar fMask = 1 << (3 - (fid & 3));

                float sum = 0.f;
                #pragma uroll
                for (int i = 0; i < 16; i++) {
                    if (!(i & fMask))
                        sum += counters[i * 16 + 2 * groupId + w];
                }

                if (M > 1) {
                    atomicAdd(binSums + (feature[fid].FirstFoldIndex) * 2 + w, sum);
                } else {
                    binSums[(feature[fid].FirstFoldIndex) * 2 + w] = sum;
                }
            }
        }
    }

    template <int BLOCK_SIZE,
              int BLOCKS_PER_FEATURE_COUNT>
    inline void RunComputeHist2NonBinaryKernel(const TCFeature* nbFeatures, int nbCount,
                                              const ui32* cindex, int dsSize,
                                              const float* target, const float* weight,  const ui32* indices,
                                              const TDataPartition* partition,
                                              float* binSums, const int binFeatureCount,
                                              bool fullPass,
                                              TCudaStream stream,
                                              dim3 numBlocks) {

        if (fullPass)
        {
            ComputeSplitPropertiesNBImpl < BLOCK_SIZE, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, binFeatureCount
            );

        } else {
                ComputeSplitPropertiesNBImpl < BLOCK_SIZE, false, BLOCKS_PER_FEATURE_COUNT> << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                        nbFeatures, nbCount, cindex, target, weight, dsSize,
                                indices, partition, binSums, binFeatureCount);
        }

    }

    inline ui32 EstimateBlockPerFeatureMultiplier(dim3 numBlocks, ui32 dsSize) {
        ui32 multiplier = 1;
        while ((numBlocks.x * numBlocks.y * min(numBlocks.z, 4) * multiplier < TArchProps::SMCount()) && ((dsSize / multiplier) > 10000) && (multiplier < 64)) {
            multiplier *= 2;
        }
        return multiplier;
    }

    void ComputeHist2NonBinary(const TCFeature* nbFeatures, int nbCount,
                               const ui32* cindex, int dsSize,
                               const float* target, const float* weight,  const ui32* indices,
                               const TDataPartition* partition, ui32 partCount, ui32 foldCount,
                               float* binSums, const int binFeatureCount,
                               bool fullPass,
                               TCudaStream stream)
    {
        if (nbCount) {
            dim3 numBlocks;
            numBlocks.x = (nbCount + 3) / 4;
            const int histPartCount = (fullPass ? partCount : partCount / 2);
            numBlocks.y = histPartCount;
            numBlocks.z = foldCount;

            const int blockSize = 384;
            const ui32 multiplier = min(EstimateBlockPerFeatureMultiplier(numBlocks, dsSize), 64);
            numBlocks.x *= multiplier;

            if (multiplier == 1) {
                RunComputeHist2NonBinaryKernel<blockSize, 1>(nbFeatures, nbCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
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
            } else {
                exit(1);
            }

            const int scanBlockSize = 256;
            dim3 scanBlocks;
            scanBlocks.x = (nbCount * 32 + scanBlockSize - 1) / scanBlockSize;
            scanBlocks.y = histPartCount;
            scanBlocks.z = foldCount;
            const int scanOffset = fullPass ? 0 : ((partCount / 2) * binFeatureCount * 2) * foldCount;
            ScanHistogramsImpl<scanBlockSize, 2><<<scanBlocks, scanBlockSize, 0, stream>>>(nbFeatures, nbCount, binFeatureCount, binSums + scanOffset);

            if (!fullPass) {
                UpdatePointwiseHistograms(binSums,  binFeatureCount, partCount, foldCount, 2, partition, stream);
            }
        }
    }

    template <int BLOCK_SIZE, int BLOCKS_PER_FEATURE_COUNT>
    void RunComputeHist2BinaryKernel(const TCFeature* bFeatures, int bCount,
                                     const ui32* cindex, int dsSize,
                                     const float* target, const float* weight, const ui32* indices,
                                     const TDataPartition* partition,
                                     float* binSums, bool fullPass,
                                     TCudaStream stream,
                                     dim3 numBlocks) {
        if (fullPass)
        {
            ComputeSplitPropertiesBImpl < BLOCK_SIZE, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    bFeatures, bCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, bCount
            );
        } else {
            ComputeSplitPropertiesBImpl < BLOCK_SIZE, false, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
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

        if (bCount) {

            if (multiplier == 1) {
                RunComputeHist2BinaryKernel<blockSize, 1>(bFeatures, bCount, cindex, dsSize, target, weight, indices, partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 2) {
                RunComputeHist2BinaryKernel<blockSize, 2>(bFeatures, bCount, cindex, dsSize, target, weight, indices, partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 4) {
                RunComputeHist2BinaryKernel<blockSize, 4>(bFeatures, bCount, cindex, dsSize, target, weight, indices, partition,  binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 8) {
                RunComputeHist2BinaryKernel<blockSize, 8>(bFeatures, bCount, cindex, dsSize, target, weight, indices, partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 16) {
                RunComputeHist2BinaryKernel<blockSize, 16>(bFeatures, bCount, cindex, dsSize, target, weight, indices, partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 32) {
                RunComputeHist2BinaryKernel<blockSize, 32>(bFeatures, bCount, cindex, dsSize, target, weight, indices, partition, binSums, fullPass, stream, numBlocks);
            } else if (multiplier == 64) {
                RunComputeHist2BinaryKernel<blockSize, 64>(bFeatures, bCount, cindex, dsSize, target, weight, indices, partition, binSums, fullPass, stream, numBlocks);
            } else {
                exit(1);
            }

            if (!fullPass) {
                UpdatePointwiseHistograms(binSums, bCount, partsCount, foldCount, 2, partition, stream);
            }
        }
    }



    template <int BLOCK_SIZE, bool FULL_PASS, int M>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesHalfByteImpl(
            const TCFeature* __restrict feature, int fCount, const ui32* __restrict cindex,
            const float* __restrict target, const float* __restrict weight, int dsSize,
            const ui32* __restrict indices,
            const TDataPartition* partition,
            float* binSums,
            const int totalFeatureCount) {


        TPartOffsetsHelper helper(gridDim.z);
        helper.ShiftPartAndBinSumsPtr(partition, binSums, totalFeatureCount, FULL_PASS);

        feature += (blockIdx.x / M) * 8;
        cindex += feature->Offset * ((size_t)dsSize);
        fCount = min(fCount - (blockIdx.x / M) * 8, 8);

//
        __shared__ float smem[16 * BLOCK_SIZE];


        using THist = TPointHistHalfByte<BLOCK_SIZE>;


        ComputeHistogram<BLOCK_SIZE, 1, GetUnrollFactorHalfByteHist(),  M, THist >(indices + partition->Offset, partition->Size,
                                                       target + partition->Offset, weight + partition->Offset,
                                                       cindex, smem);

        __syncthreads();

        const int fid = (threadIdx.x / 32);
        const int fold = (threadIdx.x / 2) & 15;
        const int w = threadIdx.x & 1;


        if (fid < fCount && fold < feature[fid].Folds) {
            if (M > 1) {
                atomicAdd(binSums + (feature[fid].FirstFoldIndex + fold) * 2 +  w, smem[fold * 16 + 2 * fid + w]);
            } else {
                binSums[(feature[fid].FirstFoldIndex + fold) * 2 +  w] = smem[fold * 16 + 2 * fid + w];
            }
        }

    }


    template <int BLOCK_SIZE,
              int BLOCKS_PER_FEATURE_COUNT>
    inline void RunComputeHist2HalfByteKernel(const TCFeature* nbFeatures, int nbCount,
                                              const ui32* cindex, int dsSize,
                                              const float* target, const float* weight,  const ui32* indices,
                                              const TDataPartition* partition,
                                              float* binSums, const int binFeatureCount,
                                              bool fullPass,
                                              TCudaStream stream,
                                              dim3 numBlocks) {

        if (fullPass)
        {
            ComputeSplitPropertiesHalfByteImpl < BLOCK_SIZE, true, BLOCKS_PER_FEATURE_COUNT > << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, binFeatureCount
            );

        } else {
            ComputeSplitPropertiesHalfByteImpl < BLOCK_SIZE, false, BLOCKS_PER_FEATURE_COUNT> << <numBlocks, BLOCK_SIZE, 0, stream>>>(
                    nbFeatures, nbCount, cindex, target, weight, dsSize,
                            indices, partition, binSums, binFeatureCount);
        }

    }

    void ComputeHist2HalfByte(const TCFeature* halfByteFeatures, int halfByteFeaturesCount,
                              const ui32* cindex, int dsSize,
                              const float* target, const float* weight,  const ui32* indices,
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

        if (halfByteFeaturesCount) {

            if (multiplier == 1) {
                RunComputeHist2HalfByteKernel<blockSize, 1>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 2) {
                RunComputeHist2HalfByteKernel<blockSize, 2>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 4) {
                RunComputeHist2HalfByteKernel<blockSize, 4>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize, target, weight, indices, partition,  binSums, binFeatureCount,  fullPass, stream, numBlocks);
            } else if (multiplier == 8) {
                RunComputeHist2HalfByteKernel<blockSize, 8>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount, fullPass, stream, numBlocks);
            } else if (multiplier == 16) {
                RunComputeHist2HalfByteKernel<blockSize, 16>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount,  fullPass, stream, numBlocks);
            } else if (multiplier == 32) {
                RunComputeHist2HalfByteKernel<blockSize, 32>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount,  fullPass, stream, numBlocks);
            } else if (multiplier == 64) {
                RunComputeHist2HalfByteKernel<blockSize, 64>(halfByteFeatures, halfByteFeaturesCount, cindex, dsSize, target, weight, indices, partition, binSums, binFeatureCount,  fullPass, stream, numBlocks);
            } else {
                exit(1);
            }

            const int scanBlockSize = 256;
            dim3 scanBlocks;
            scanBlocks.x = static_cast<ui32>((halfByteFeaturesCount * 32 + scanBlockSize - 1) / scanBlockSize);
            scanBlocks.y = static_cast<ui32>(histCount);
            scanBlocks.z = foldCount;
            const int scanOffset = fullPass ? 0 : ((partsCount / 2) * binFeatureCount * 2) * foldCount;
            ScanHistogramsImpl<scanBlockSize, 2><<<scanBlocks, scanBlockSize, 0, stream>>>(halfByteFeatures, halfByteFeaturesCount, binFeatureCount, binSums + scanOffset);

            if (!fullPass) {
                UpdatePointwiseHistograms(binSums,  binFeatureCount, partsCount, foldCount, 2, partition, stream);
            }
        }
    }




    __global__ void UpdateBinsImpl(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                                   ui32 loadBit, ui32 foldBits) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            const ui32 idx = LdgWithFallback(docIndices, i);
            const ui32 bit = (LdgWithFallback(bins, idx) >> loadBit) & 1;
            dstBins[i] =  dstBins[i] | (bit << (loadBit + foldBits));
        }
    }

    void UpdateFoldBins(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                        ui32 loadBit, ui32 foldBits, TCudaStream stream) {


        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        UpdateBinsImpl<<<numBlocks, blockSize, 0, stream>>>(dstBins, bins, docIndices, size, loadBit, foldBits);
    }

}
