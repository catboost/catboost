#include "ctr_calcers.cuh"
#include <catboost/cuda/cuda_util/kernel/index_wrapper.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>


namespace NKernel {

    __global__ void GatherTrivialWeightsImpl(const ui32* indices, ui32 size,
                                             ui32 firstZeroIndex, bool writeSegmentStartFloatMask,
                                             float* dst) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            TIndexWrapper indexWrapper(StreamLoad(indices + i));
            const float val = (indexWrapper.Index() < firstZeroIndex ? 1.0f : 0.0f);
            dst[i] = (writeSegmentStartFloatMask && indexWrapper.IsSegmentStart()) ? -val : val;
        }
    }

    void GatherTrivialWeights(const ui32* indices, ui32 size,
                              ui32 firstZeroIndex, bool writeSegmentStartFloatMask,
                              float* dst,
                              TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        if (numBlocks) {
            GatherTrivialWeightsImpl <<<numBlocks, blockSize, 0, stream>>>(indices, size, firstZeroIndex, writeSegmentStartFloatMask, dst);
        }
    }


    __global__ void WriteMaskImpl(const ui32* indices, ui32 size,
                                  float* dst) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            TIndexWrapper indexWrapper(StreamLoad(indices + i));
            const float val = dst[i];
            dst[i] = indexWrapper.IsSegmentStart() ? -val : val;
        }
    }

    void WriteMask(const ui32* indices, ui32 size,
                   float* dst,
                   TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        if (numBlocks) {
            WriteMaskImpl <<<numBlocks, blockSize, 0, stream>>>(indices, size, dst);
        }
    }

    __global__ void WeightedBinFreqCtrsImpl(const ui32* writeIndices, const ui32* bins,
                                            const float* binSums,
                                            float totalWeight, float prior, float priorObservations,
                                            float* dst,
                                            ui32 size) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            ui32 dstIdx = writeIndices ? TIndexWrapper(writeIndices[i]).Index() : i;
            dst[dstIdx] = (binSums[bins[i]] + prior) / (totalWeight + priorObservations);
        }
    }

    template <int BLOCK_SIZE, int DOCS_PER_THREAD>
    __global__ void NonWeightedBinFreqCtrsImpl(const ui32* writeIndices,
                                               const ui32* bins, const ui32* binOffsets, ui32 size,
                                               float prior, float priorObservations,
                                               float* dst) {
        const ui32 i = blockIdx.x * BLOCK_SIZE * DOCS_PER_THREAD  + threadIdx.x;

        int dstIndices[DOCS_PER_THREAD];
        ui32 binsLocal[DOCS_PER_THREAD];

        #pragma unroll DOCS_PER_THREAD
        for (int j = 0; j < DOCS_PER_THREAD; ++j) {
            const ui32 idx = i + BLOCK_SIZE * j;
            dstIndices[j] = idx < size ? (int)(writeIndices ? TIndexWrapper(writeIndices[idx]).Index() : idx) : -1;
            binsLocal[j] =  idx < size ? bins[idx] : 0;
        }

        #pragma unroll DOCS_PER_THREAD
        for (int j = 0; j < DOCS_PER_THREAD; ++j)
        {
            const ui32 bin = binsLocal[j];
            const ui32 currentBinOffset = LdgWithFallback(binOffsets + bin, 0);
            const ui32 nextBinOffset = bin < size ? LdgWithFallback(binOffsets + bin + 1, 0) : size;
            binsLocal[j] = (nextBinOffset - currentBinOffset);
        }

        #pragma unroll DOCS_PER_THREAD
        for (int j = 0; j < DOCS_PER_THREAD; ++j)
        {
            if (dstIndices[j] != -1)
            {
                WriteThrough(dst + dstIndices[j], (binsLocal[j] + prior) / (size + priorObservations));
            }
        }
    }


    void ComputeWeightedBinFreqCtr(const ui32* writeIdx, const ui32* bins,
                                   const float* binSums,
                                   float totalWeight, float prior, float priorObservations,
                                   float* dst,
                                   ui32 size, TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        if (numBlocks) {
            WeightedBinFreqCtrsImpl <<<numBlocks, blockSize, 0, stream>>>(writeIdx, bins, binSums, totalWeight, prior, priorObservations, dst, size);
        }
    }

    void ComputeNonWeightedBinFreqCtr(const ui32* writeIdx, const ui32* bins,
                                      const ui32* binOffsets, ui32 size,
                                      float prior, float priorObservations,
                                      float* dst, TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 elementsPerThreads = 4;
        const ui32 numBlocks = CeilDivide(size, blockSize * elementsPerThreads);
        if (numBlocks) {
            NonWeightedBinFreqCtrsImpl<blockSize, elementsPerThreads> <<<numBlocks, blockSize, 0, stream>>>(writeIdx, bins, binOffsets, size, prior, priorObservations, dst);
        }
    }


    __global__ void UpdateBordersMaskImpl(const ui32* bins, const ui32* prevBins, ui32* indices, ui32 size) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            TIndexWrapper currentIndex(indices[i]);

            bool mask = currentIndex.IsSegmentStart();
            if (!mask) {
                mask |= (i == 0 || (bins[i] != bins[i - 1]));
            }
            if (!mask) {
                TIndexWrapper prevIndex(indices[i - 1]);
                const ui32 currentBin = LdgWithFallback(prevBins, currentIndex.Index());
                const ui32 prevBin = LdgWithFallback(prevBins, prevIndex.Index());
                mask |=  currentBin != prevBin;
            }
            currentIndex.UpdateMask(mask);
            indices[i] = currentIndex.Value();
        }
    }

    void UpdateBordersMask(const ui32* bins, const ui32* prevBins, ui32* indices, ui32 size, TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);

        if (numBlocks) {
            UpdateBordersMaskImpl<<<numBlocks, blockSize, 0, stream>>>(bins, prevBins, indices, size);
        }
    }


    __global__ void MergeBinsKernelImpl(ui32* bins, const ui32* prev, ui32 shift, ui32 size) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            bins[i] = (bins[i] << shift) | prev[i];
        }
    }

    void MergeBinsKernel(ui32* bins, const ui32* prev, ui32 shift, ui32 size, TCudaStream stream)  {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);

        if (numBlocks) {
            MergeBinsKernelImpl<<<numBlocks, blockSize, 0, stream>>>(bins, prev, shift, size);
        }
    }


    __global__ void ExtractBorderMasksStartImpl(const ui32* indices, ui32* dst, ui32 size) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            const bool isStart = TIndexWrapper(indices[i]).IsSegmentStart();
            dst[i] = isStart;
        }
    }

    __global__ void ExtractBorderMasksEndImpl(const ui32* indices, ui32* dst, ui32 size) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            const bool isEnd= (i + 1) < size ? TIndexWrapper(indices[i + 1]).IsSegmentStart() : true;
            dst[i] = isEnd;
        }
    }

    void ExtractBorderMasks(const ui32* indices, ui32* dst, ui32 size, bool startSegment, TCudaStream stream) {
        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        if (numBlocks) {
            if (startSegment) {
                ExtractBorderMasksStartImpl << < numBlocks, blockSize, 0, stream >> > (indices, dst, size);
            } else {
                ExtractBorderMasksEndImpl << < numBlocks, blockSize, 0, stream >> > (indices, dst, size);
            }
        }
    }


    template <bool IS_BORDERS, int N>
    __global__ void FillBinarizedTargetsStatsImpl(const ui8* binarizedTargets, const float* sampleWeight,
                                                  float* dst, ui32 size, ui32 binIndex) {

        const ui32 i = (blockIdx.x * blockDim.x + threadIdx.x) * N;

        float localSamples[N];

#pragma unroll
        for (int k = 0; k < N; ++k) {
            const int idx = i + k;

            if (idx < size) {
                const float weight = StreamLoad(sampleWeight + idx);
                localSamples[k]=  abs(weight) * (IS_BORDERS ? StreamLoad(binarizedTargets + idx) > binIndex
                                                            : StreamLoad(binarizedTargets + idx) == binIndex);
                localSamples[k] = ExtractSignBit(weight) ? -localSamples[k] : localSamples[k];
            }
        }

#pragma unroll
        for (int k = 0; k < N; ++k) {
            const int idx = i + k;
            if (idx < size) {
                dst[idx] = localSamples[k];
            }
        }
    }

    void FillBinarizedTargetsStats(const ui8* sample, const float* sampleWeights, ui32 size,
                                   float* sums, ui32 binIndex, bool borders,
                                   TCudaStream stream) {
        const ui32 blockSize = 256;
        const int N = 4;
        const ui32 numBlocks = CeilDivide(size, N * blockSize);
        if (numBlocks) {
            if (borders) {
                FillBinarizedTargetsStatsImpl<true, N> << < numBlocks, blockSize, 0, stream >> >(sample, sampleWeights, sums, size, binIndex);
            } else {
                FillBinarizedTargetsStatsImpl<false, N> << < numBlocks, blockSize, 0, stream >> >(sample, sampleWeights, sums, size, binIndex);
            }
        }
    }


    __global__ void MakeMeansImpl(float* sums, const float* weights, ui32 size,
                                  float sumPrior, float weightPrior) {
        ui32 tid = blockDim.x * blockIdx.x + threadIdx.x;

        if (tid < size) {
            sums[tid] = (sums[tid] + sumPrior) / (weights[tid] + weightPrior);
        }
    }

    void MakeMeans(float* sums, const float* weights, ui32 size,
                   float sumPrior, float weightPrior,
                   TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        if (numBlocks) {
            MakeMeansImpl <<<numBlocks, blockSize, 0, stream>>> (sums, weights, size, sumPrior, weightPrior);
        }
    }

    __global__ void MakeMeansAndScatterImpl(const float* sums, const float* weights, ui32 size,
                                            float sumPrior, float weightPrior,
                                            const ui32* map, ui32 mask,
                                            float* dst) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            const ui32 m = map ? StreamLoad(map + i) & mask : i;
            dst[m] = (sums[i] + sumPrior) / (weights[i] + weightPrior);
        }
    }

    void MakeMeansAndScatter(const float* sums, const float* weights, ui32 size,
                             float sumPrior, float weightPrior,
                             const ui32* map, ui32 mask,
                             float* dst,
                             TCudaStream stream) {

        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        if (numBlocks) {
            MakeMeansAndScatterImpl <<<numBlocks, blockSize, 0, stream>>> (sums, weights, size, sumPrior, weightPrior, map, mask, dst);
        }
    }
}
