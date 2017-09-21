#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"

#include <catboost/cuda/cuda_lib/kernel/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>


namespace NKernel {


    texture<float, cudaTextureType1D, cudaReadModeElementType> weight_tex_ref;
    texture<float, cudaTextureType1D, cudaReadModeElementType> target_tex_ref;

    void BindPointwiseTextureData(const float* targets, 
                                  const float* weights,
                                  ui32 size) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaBindTexture(0, target_tex_ref, targets, channelDesc, size * sizeof(float));
        cudaBindTexture(0, weight_tex_ref, weights, channelDesc, size * sizeof(float));
    }

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
                short f = ((threadIdx.x & 7) + (i << 1)) & 6;
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

    template <int STRIPE_SIZE, int HIST_BLOCK_COUNT, int N, typename THist>
    __forceinline__ __device__ void ComputeHistogram(
            const ui32* __restrict indices, ui32 dsSize,
            const float* __restrict target, const float* __restrict weight,
            const ui32* __restrict cindex, float* result, ui32 textureOffset)
    {

        THist hist(result);

        int i = (threadIdx.x & 31) + (threadIdx.x / 32 / HIST_BLOCK_COUNT) * 32;
        int iteration_count = (dsSize - i + (STRIPE_SIZE - 1)) / STRIPE_SIZE;
        int blocked_iteration_count = ((dsSize - (i | 31) + (STRIPE_SIZE - 1)) / STRIPE_SIZE) / N;

        weight += i;
        target += i;
        indices += i;

#pragma unroll 4
        for(int j = 0; j < blocked_iteration_count; ++j) {
            ui32 local_index[N];
#pragma unroll
            for(int k = 0; k < N; k++) {
#if __CUDA_ARCH__ >= 350
                local_index[k] = __ldg(indices + STRIPE_SIZE * k);
#else
                local_index[k] = indices[STRIPE_SIZE * k];
#endif
            }

            ui32 local_ci[N];
            float local_w[N];
            float local_wt[N];

#pragma unroll
            for(int k = 0; k < N; ++k) {
#if __CUDA_ARCH__ >= 350
            local_ci[k] = __ldg(cindex + local_index[k]);
            local_w[k] = __ldg(weight + STRIPE_SIZE * k);
            local_wt[k]= __ldg(target + STRIPE_SIZE * k);
#else
                local_ci[k] =  cindex[local_index[k]];
                local_w[k] =  tex1Dfetch(weight_tex_ref, textureOffset + i + STRIPE_SIZE * k);
                local_wt[k] =  tex1Dfetch(target_tex_ref, textureOffset + i + STRIPE_SIZE * k);
#endif
            }

#pragma unroll
            for(int k = 0; k < N; ++k) {
                hist.AddPoint(local_ci[k], local_wt[k], local_w[k]);
            }

            i += STRIPE_SIZE * N;
            indices += STRIPE_SIZE * N;
            target += STRIPE_SIZE * N;
            weight += STRIPE_SIZE * N;
        }

        for(int k = blocked_iteration_count * N; k < iteration_count; ++k) {
#if __CUDA_ARCH__ >= 350
            const int index = __ldg(indices);
            ui32 ci = __ldg(cindex + index);
            float w = __ldg(weight);
            float wt = __ldg(target);
#else
            const int index = indices[0];
            ui32 ci = cindex[index];
            float w = tex1Dfetch(weight_tex_ref, i + textureOffset);
            float wt = tex1Dfetch(target_tex_ref, i + textureOffset);
#endif
            hist.AddPoint(ci, wt, w);
            i += STRIPE_SIZE;
            indices += STRIPE_SIZE;
            target += STRIPE_SIZE;
            weight += STRIPE_SIZE;
        }
        __syncthreads();

        hist.Reduce();

    }



    template <int BLOCK_SIZE, int OUTER_HIST_BITS_COUNT, int INNER_HIST_BITS_COUNT, int N>
    __forceinline__ __device__ void ComputeSplitPropertiesPass(const TCFeature* __restrict feature, const ui32* __restrict cindex,
                                                   const float* __restrict target, const float* __restrict weight, const ui32* __restrict indices,
                                                   const TDataPartition* __restrict partition, int fCount,
                                                   volatile float* binSumsForPart,
                                                   float* smem) {

        using THist = TPointHist < OUTER_HIST_BITS_COUNT, INNER_HIST_BITS_COUNT, BLOCK_SIZE >;
        const int stripeSize = BLOCK_SIZE >> OUTER_HIST_BITS_COUNT;
        const int histBlockCount =  1 << OUTER_HIST_BITS_COUNT;

        ComputeHistogram<stripeSize, histBlockCount, N, THist >(indices + partition->Offset,
                partition->Size, target + partition->Offset, weight + partition->Offset, cindex, smem,  partition->Offset);

        __syncthreads();



        int fid = (threadIdx.x / 64);
        int fold = (threadIdx.x / 2) & 31;


        for (int upperBits = 0; upperBits < (1 << (OUTER_HIST_BITS_COUNT + INNER_HIST_BITS_COUNT)); ++upperBits) {
            const int binOffset = upperBits << 5;

            if (fid < fCount && fold < min((int)feature[fid].Folds - binOffset, 32)) {
                int w = threadIdx.x & 1;
                binSumsForPart[(feature[fid].FirstFoldIndex + fold + binOffset) * 2 + w] = smem[fold * 8 + 2 * fid + w + 256 * upperBits];
            }
        }

        __syncthreads();


    }



#define DECLARE_PASS(O, I, N) \
    ComputeSplitPropertiesPass<BLOCK_SIZE, O, I, N>(feature, cindex, target, weight, indices, partition, fCount, binSums, &counters[0]);


    template <int BLOCK_SIZE,
            bool FULL_PASS>
#if __CUDA_ARCH__ >= 520
    __launch_bounds__(BLOCK_SIZE, 2)
#else
    __launch_bounds__(BLOCK_SIZE, 1)
#endif
    __global__ void ComputeSplitPropertiesNBImpl(
            const TCFeature* __restrict feature, int fCount, const ui32* __restrict cindex,
            const float* __restrict target, const float* __restrict weight, int dsSize,
            const ui32* __restrict indices,
            const TDataPartition* __restrict partition,
            float* binSums,
            const int totalFeatureCount) {


        TPartOffsetsHelper helper(gridDim.z);

        if (FULL_PASS)
        {
            partition += helper.GetDataPartitionOffset(blockIdx.y, blockIdx.z);
            binSums +=  helper.GetHistogramOffset(blockIdx.y, blockIdx.z) * 2 * totalFeatureCount;
        } else {
            const ui64 leftPartOffset =  helper.GetDataPartitionOffset(blockIdx.y, blockIdx.z);
            const ui64 rightPartOffset =  helper.GetDataPartitionOffset(gridDim.y | blockIdx.y, blockIdx.z);
            const int leftPartSize = partition[leftPartOffset].Size;
            const int rightPartSize = partition[rightPartOffset].Size;

            partition += (leftPartSize < rightPartSize) ? leftPartOffset : rightPartOffset;
            binSums += 2 * totalFeatureCount * helper.GetHistogramOffset(gridDim.y | blockIdx.y, blockIdx.z);
        }

        feature += blockIdx.x * 4;
        cindex += feature->Offset * ((size_t)dsSize);
        fCount = min(fCount - blockIdx.x * 4, 4);

//
        __shared__ float counters[32 * BLOCK_SIZE];
        const int maxBinCount = GetMaxBinCount(feature, fCount, (int*) &counters[0]);
        __syncthreads();


        if (partition->Size) {
            if (maxBinCount <= 32) {
#if __CUDA_ARCH__ >= 350
                DECLARE_PASS(0, 0, 8);
#else
                DECLARE_PASS(0, 0, 4);
#endif
            }
            else if (maxBinCount <= 64) {
#if __CUDA_ARCH__ >= 350
                DECLARE_PASS(0, 1, 4);
#else
                DECLARE_PASS(0, 1, 2);
#endif
            } else if (maxBinCount <= 128) {
#if __CUDA_ARCH__ >= 350
                DECLARE_PASS(0, 2, 4);
#else
                DECLARE_PASS(2, 0, 2);
#endif
            } else {
#if __CUDA_ARCH__ >= 350
                DECLARE_PASS(1, 2, 4);
#else
                DECLARE_PASS(2, 1, 2);
#endif
            }
        }
    }

    template <int BIN_COUNT, int BLOCK_SIZE,  bool FULL_PASS>
    __launch_bounds__(BLOCK_SIZE, 1)
    __global__ void ComputeSplitPropertiesBImpl(
            const TCFeature* __restrict feature, int fCount, const ui32* __restrict cindex,
            const float* __restrict target, const float* __restrict weight, int dsSize, const ui32* __restrict indices,
            const TDataPartition* __restrict partition, float* __restrict binSums, int totalFeatureCount)
    {

        TPartOffsetsHelper helper(gridDim.z);

        if (FULL_PASS)
        {
            partition += helper.GetDataPartitionOffset(blockIdx.y, blockIdx.z);
            binSums +=  helper.GetHistogramOffset(blockIdx.y, blockIdx.z) * 2 * totalFeatureCount;
        } else {
            const ui64 leftPartOffset =  helper.GetDataPartitionOffset(blockIdx.y, blockIdx.z);
            const ui64 rightPartOffset =  helper.GetDataPartitionOffset(gridDim.y | blockIdx.y, blockIdx.z);
            const int leftPartSize = partition[leftPartOffset].Size;
            const int rightPartSize = partition[rightPartOffset].Size;

            partition += (leftPartSize < rightPartSize) ? leftPartOffset : rightPartOffset;
            binSums += 2 * totalFeatureCount * helper.GetHistogramOffset(gridDim.y | blockIdx.y, blockIdx.z);
        }


        feature += blockIdx.x * 20;
        cindex += feature->Offset * ((size_t)dsSize);
        fCount = min(fCount - blockIdx.x * 20, 20);

        __shared__ float counters[BIN_COUNT * BLOCK_SIZE];

        if (partition->Size)
        {

            ComputeHistogram < BLOCK_SIZE, 1, 8, TPointHist<0, 0, BLOCK_SIZE> > (indices + partition->Offset,
                    partition->Size, target + partition->Offset, weight + partition->Offset,
                    cindex, &counters[0], partition->Offset);

            uchar fold = (threadIdx.x >> 1) & 1;
            uchar fid = (threadIdx.x >> 2);
            if (fid < fCount && fold < feature[fid].Folds)
            {
                uchar w = threadIdx.x & 1;
                uchar fMask = 1 << (4 - fid % 5);
                float sum = 0.f;
                #pragma uroll
                for (int i = 0; i < 32; i++)
                {
                    if (!(i & fMask) || fold)
                        sum += counters[i * 8 + 2 * (fid / 5) + w];
                }

                binSums[(feature[fid].FirstFoldIndex + fold) * 2 + w] = sum;
            }
        }
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
            if (fullPass)
            {
                ComputeSplitPropertiesNBImpl < blockSize, true > << <numBlocks, blockSize, 0, stream>>>(
                        nbFeatures, nbCount, cindex, target, weight, dsSize,
                                indices, partition, binSums, binFeatureCount
                );
            } else {
                ComputeSplitPropertiesNBImpl < blockSize, false > << <numBlocks, blockSize, 0, stream>>>(
                        nbFeatures, nbCount, cindex, target, weight, dsSize,
                                indices, partition, binSums, binFeatureCount
                );
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

    void ComputeHist2Binary(const TCFeature* bFeatures, int bCount,
                            const ui32* cindex, int dsSize,
                            const float* target, const float* weight, const ui32* indices,
                            const TDataPartition* partition, ui32 partsCount, ui32 foldCount,
                            float* binSums, bool fullPass,
                            TCudaStream stream)
    {
        const int blockSize = 384;

        dim3 numBlocks;
        numBlocks.x = (bCount + 19) / 20;
        const int histCount = fullPass ? partsCount : partsCount / 2;
        numBlocks.y = histCount;
        numBlocks.z = foldCount;

        if (bCount) {
            if (fullPass)
            {
                ComputeSplitPropertiesBImpl < 32, blockSize, true > << <numBlocks, blockSize, 0, stream>>>(
                        bFeatures, bCount, cindex, target, weight, dsSize,
                                indices, partition, binSums, bCount
                );
            } else {
                ComputeSplitPropertiesBImpl < 32, blockSize, false > << <numBlocks, blockSize, 0, stream>>>(
                        bFeatures, bCount, cindex, target, weight, dsSize,
                                indices, partition, binSums, bCount
                );
            }

            if (!fullPass) {
                UpdatePointwiseHistograms(binSums, bCount, partsCount, foldCount, 2, partition, stream);
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
