#include "pointwise_hist2.cuh"
#include "split_properties_helpers.cuh"

#include <cooperative_groups.h>
#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>
#include <library/cpp/cuda/wrappers/arch.cuh>

using namespace cooperative_groups;

namespace NKernel
{


    __global__ void UpdateBinsImpl(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                                   ui32 loadBit, ui32 foldBits) {
        const ui32 i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            const ui32 idx = LdgWithFallback(docIndices, i);
            const ui32 bit = (LdgWithFallback(bins, idx) >> loadBit) & 1;
            dstBins[i] = dstBins[i] | (bit << (loadBit + foldBits));
        }
    }

    void UpdateFoldBins(ui32* dstBins, const ui32* bins, const ui32* docIndices, ui32 size,
                        ui32 loadBit, ui32 foldBits, TCudaStream stream) {


        const ui32 blockSize = 256;
        const ui32 numBlocks = CeilDivide(size, blockSize);
        UpdateBinsImpl << < numBlocks, blockSize, 0, stream >> > (dstBins, bins, docIndices, size, loadBit, foldBits);
    }



    template <int HIST_COUNT>
    __global__ void UpdatePointwiseHistogramsImpl(float* histogram,
                                                  const int firstBinFeature, int featuresCount,
                                                  const TDataPartition* parts,
                                                  const ui64 histLineSize) {

        TPointwisePartOffsetsHelper helper(gridDim.z);

        const int leftPartId = helper.GetDataPartitionOffset(blockIdx.y, blockIdx.z);
        const int rightPartId = helper.GetDataPartitionOffset(blockIdx.y | gridDim.y, blockIdx.z);
        const int binFeature = firstBinFeature + blockIdx.x * blockDim.x + threadIdx.x;

        if (binFeature < (firstBinFeature + featuresCount)) {
            const TDataPartition leftPart = parts[leftPartId];
            const TDataPartition rightPart = parts[rightPartId];

            const bool isLeftCalculated = leftPart.Size < rightPart.Size;


            const size_t leftOffset = HIST_COUNT * (helper.GetHistogramOffset(blockIdx.y, blockIdx.z) * histLineSize + binFeature);
            const size_t rightOffset = HIST_COUNT * (helper.GetHistogramOffset(blockIdx.y | gridDim.y, blockIdx.z) * histLineSize + binFeature);

            float calcVal[HIST_COUNT];
            float complementVal[HIST_COUNT];

#pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId) {
                calcVal[histId] = histogram[rightOffset + histId];
                complementVal[histId] = histogram[leftOffset + histId] - calcVal[histId];
            }

#pragma unroll
            for (int histId = 0; histId < HIST_COUNT; ++histId) {
                histogram[leftOffset + histId] = isLeftCalculated ? calcVal[histId] : complementVal[histId] ;
                histogram[rightOffset + histId] = isLeftCalculated ? complementVal[histId] : calcVal[histId];
            }
        }
    }

    void UpdatePointwiseHistograms(float* histograms,
                                   int firstBinFeature, int binFeatureCount,
                                   int partCount,
                                   int foldCount,
                                   int histCount,
                                   int histLineSize,
                                   const TDataPartition* parts,
                                   TCudaStream stream) {

        const int blockSize = 256;
        dim3 numBlocks;
        numBlocks.x = (binFeatureCount + blockSize - 1) / blockSize;
        numBlocks.y = partCount / 2;
        numBlocks.z = foldCount;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        if (histCount == 1) {
            UpdatePointwiseHistogramsImpl<1><<<numBlocks, blockSize, 0, stream>>>(histograms, firstBinFeature, binFeatureCount, parts, histLineSize);
        }
        else if (histCount == 2) {
            UpdatePointwiseHistogramsImpl<2><<<numBlocks, blockSize, 0, stream>>>(histograms, firstBinFeature, binFeatureCount, parts, histLineSize);
        } else {
            CB_ENSURE_INTERNAL(false, "histCount should be 1 or 2, not " << histCount);
        }
    }


    void ScanPointwiseHistograms(const TCFeature* features,
                                 int featureCount, int partCount, int foldCount,
                                 int histLineSize, bool fullPass,
                                 int histCount,
                                 float* binSums,
                                 TCudaStream stream) {
        const int scanBlockSize = 256;
        const int histPartCount = (fullPass ? partCount : partCount / 2);
        dim3 scanBlocks;
        scanBlocks.x = (featureCount * 32 + scanBlockSize - 1) / scanBlockSize;
        scanBlocks.y = histPartCount;
        scanBlocks.z = foldCount;
        if (IsGridEmpty(scanBlocks)) {
            return;
        }
        const int scanOffset = fullPass ? 0 : ((partCount / 2) * histLineSize * histCount) * foldCount;
        if (histCount == 1) {
            ScanHistogramsImpl<scanBlockSize, 1> << < scanBlocks, scanBlockSize, 0, stream >> > (features, featureCount, histLineSize, binSums + scanOffset);
        } else if (histCount == 2) {
            ScanHistogramsImpl<scanBlockSize, 2> << < scanBlocks, scanBlockSize, 0, stream >> >
                                                                                    (features, featureCount, histLineSize, binSums + scanOffset);
        } else {
            CB_ENSURE_INTERNAL(false, "histCount should be 1 or 2, not " << histCount);
        }
    }
}
