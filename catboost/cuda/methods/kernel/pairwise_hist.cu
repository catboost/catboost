#include "pairwise_hist.cuh"
#include "split_properties_helpers.cuh"
#include "pairwise_hist_one_byte_5bit.cuh"
#include "pairwise_hist_one_byte_6bit.cuh"
#include "pairwise_hist_one_byte_7bit.cuh"
#include "pairwise_hist_one_byte_8bit_atomics.cuh"

#include <library/cpp/cuda/wrappers/arch.cuh>
#include <catboost/cuda/cuda_util/kernel/instructions.cuh>
#include <catboost/cuda/cuda_util/kernel/kernel_helpers.cuh>

namespace NKernel {

    template <bool FULL_PASS>
    __global__ void BuildBinaryFeatureHistograms(const TCFeature* nbFeatures,
                                                 int featureCount,
                                                 const TDataPartition* partition,
                                                 const TPartitionStatistics* partitionStats,
                                                 const ui64 histLineSize,
                                                 float* histogram) {

        if (FULL_PASS) {
            partitionStats += blockIdx.y;
            histogram += blockIdx.y * histLineSize * 4;
        } else {
            const int depth = (int)log2((float)gridDim.y);
            int partId = GetPairwisePartIdToCalculate(partition);
            partitionStats += partId;
            histogram += (((blockIdx.z + 1) << depth) | blockIdx.y) * histLineSize * 4;
        }

        const int featuresPerBlock = blockDim.x / 32;
        const int featureId = blockIdx.x * featuresPerBlock + threadIdx.x / 32;
        nbFeatures += featureId;
        const float partWeight = partitionStats->Weight;

        if (featureId >= featureCount || partitionStats->Weight == 0) {
            return;
        }

        const int x = threadIdx.x & 31;
        const ui32 featureFolds = nbFeatures->Folds;
        const ui32 featureOffset = nbFeatures->FirstFoldIndex;

        if (nbFeatures->OneHotFeature) {
            for (ui32 fold = x; fold < featureFolds; fold += 32) {
                const ui32 offset = featureOffset + fold;
                const float hist0 = histogram[4 * offset];
                // const float hist1 = histogram[4 * offset + 1];
                const float hist2 = histogram[4 * offset + 2];
                const float hist3 = histogram[4 * offset + 3];

                const float w00 = max(hist0, 0.0f);
                const float w01 = max(hist2, 0.0f);
                const float w10 = max(hist3, 0.0f);
                const float w11 = max(partWeight - hist0 - hist2 - hist3, 0.0f);

                histogram[4 * offset] = w00;
                histogram[4 * offset + 1] = w01;
                histogram[4 * offset + 2] = w10;
                histogram[4 * offset + 3] = w11;
            }

        } else {
            for (ui32 fold = x; fold < featureFolds; fold += 32) {
                const ui32 offset = featureOffset + fold;
                const float hist0 = histogram[4 * offset];
                const float hist1 = histogram[4 * offset + 1];
                const float hist2 = histogram[4 * offset + 2];
                const float hist3 = histogram[4 * offset + 3];

                const float w00 = max(hist1 + hist2, 0.0f);
                const float w01 = max(hist0 - hist1, 0.0f);
                const float w10 = max(hist3 - hist2, 0.0f);
                const float w11 = max(partWeight - hist0 - hist3, 0.0f);

                histogram[4 * offset] = w00;
                histogram[4 * offset + 1] = w01;
                histogram[4 * offset + 2] = w10;
                histogram[4 * offset + 3] = w11;
            }
        }
    }


    void BuildBinaryFeatureHistograms(const TCFeature* features, ui32 featureCount,
                                      const TDataPartition* partition,
                                      const TPartitionStatistics* partitionStats,
                                      ui32 partCount,
                                      const ui64 histLineSize,
                                      bool fullPass,
                                      float* histogram,
                                      TCudaStream stream) {

        const int buildHistogramBlockSize = 256;

        dim3 numBlocks;
        numBlocks.x = (featureCount * 32 + buildHistogramBlockSize - 1) / buildHistogramBlockSize;
        numBlocks.y = fullPass ? partCount : partCount / 4;
        numBlocks.z = fullPass ? 1 : 3;
        if (IsGridEmpty(numBlocks)) {
            return;
        }

        if (fullPass) {
            BuildBinaryFeatureHistograms<true><< <numBlocks, buildHistogramBlockSize, 0, stream >> > (features, featureCount, partition, partitionStats, histLineSize, histogram);
        } else {
            BuildBinaryFeatureHistograms<false><< <numBlocks, buildHistogramBlockSize, 0, stream >> > (features, featureCount, partition, partitionStats, histLineSize, histogram);
        }
    }

    __global__ void UpdatePairwiseHistogramsImpl(ui32 firstFeatureId, ui32 featureCount,
                                                 const TDataPartition* parts,
                                                 const ui64 histLineSize,
                                                 float* histogram) {
        const int histCount = 4;

        const int depth = (int)log2((float)gridDim.y);
        int partIds[4];
        {
            int partSizes[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int partId = (i << depth) | blockIdx.y;
                partIds[i] = partId;
                partSizes[i] = parts[partId].Size;
            }//

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = i + 1; j < 4; ++j) {
                    if (partSizes[j] > partSizes[i]) {
                        const int tmpSize = partSizes[j];
                        const int tmpId = partIds[j];

                        partSizes[j] = partSizes[i];
                        partIds[j] = partIds[i];

                        partSizes[i] = tmpSize;
                        partIds[i] = tmpId;
                    }
                }
            }
        }

        const ui32 binFeature = firstFeatureId + blockIdx.x * blockDim.x + threadIdx.x;

        if (binFeature < (firstFeatureId + featureCount)) {

            float hists[histCount * 4];
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                const size_t srcPartIdx = (part << depth) | blockIdx.y;

                #pragma unroll
                for (int i = 0; i < histCount; ++i) {
                    hists[part * 4 + i] = histogram[histCount * (srcPartIdx * histLineSize + binFeature) + i];
                }
            }
            #pragma unroll
            for (int part = 1; part < 4; ++part) {
                #pragma unroll
                for (int i = 0; i < histCount; ++i) {
                    hists[i] -= hists[4 * part + i];
                }
            }

            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                const size_t destPartIdx = partIds[part];
                #pragma unroll
                for (int i = 0; i < histCount; ++i) {
                    histogram[histCount * (destPartIdx * histLineSize + binFeature) + i] = max(hists[part * 4 + i], 0.0f);
                }
            }
        }
    }

    void UpdatePairwiseHistograms(const ui32 firstFeatureId, const ui32 featureCount,
                                  const TDataPartition* dataParts, ui32 partCount,
                                  ui32 histLineSize,
                                  float* histograms,
                                  TCudaStream stream
    ) {
        const int blockSize = 256;
        dim3 numBlocks;
        numBlocks.x = (featureCount + blockSize - 1) / blockSize;
        numBlocks.y = partCount / 4;
        numBlocks.z = 1;
        if (IsGridEmpty(numBlocks)) {
            return;
        }
        UpdatePairwiseHistogramsImpl<< <numBlocks, blockSize, 0, stream>>>(firstFeatureId, featureCount, dataParts, histLineSize, histograms);
    }





    void ScanPairwiseHistograms(const TCFeature* features,
                                int featureCount, int partCount,
                                int histLineSize, bool fullPass,
                                float* binSums,
                                TCudaStream stream) {
        const size_t histOffset = fullPass ? 0 : (partCount / 4) * ((ui64) histLineSize * 4);

        const int scanBlockSize = 256;
        dim3 scanBlocks;

        scanBlocks.x = (featureCount * 32 + scanBlockSize - 1) / scanBlockSize;
        scanBlocks.y = fullPass ? partCount : partCount * 3 / 4;
        scanBlocks.z = 1;
        if (IsGridEmpty(scanBlocks)) {
            return;
        }

        ScanHistogramsImpl<scanBlockSize, 4> << < scanBlocks, scanBlockSize, 0, stream >> > (features, featureCount, histLineSize, binSums + histOffset);
    }


    void ComputePairwiseHistogramOneByte5Bits(const TCFeature* features,
                                              const TCFeature* featureCpu,
                                              const ui32 featureCount,
                                              const ui32 fiveBitsFeatureCount,
                                              const ui32* compressedIndex,
                                              const uint2* pairs, ui32 pairCount,
                                              const float* weight,
                                              const TDataPartition* partition,
                                              ui32 partCount,
                                              ui32 histLineSize,
                                              bool fullPass,
                                              float* histogram,
                                              int parallelStreamCount,
                                              TCudaStream stream) {


        ComputePairwiseHistogramOneByte5BitsImpl<false>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                        pairs, pairCount, weight, partition, partCount, histLineSize,
                                                        fullPass,
                                                        histogram,
                                                        parallelStreamCount,
                                                        stream);

        ComputePairwiseHistogramOneByte5BitsImpl<true>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                       pairs, pairCount, weight, partition, partCount, histLineSize,
                                                       fullPass,
                                                       histogram,
                                                       parallelStreamCount,
                                                       stream);
    }


    void ComputePairwiseHistogramOneByte6Bits(const TCFeature* features,
                                              const TCFeature* featureCpu,
                                              const ui32 featureCount,
                                              const ui32 fiveBitsFeatureCount,
                                              const ui32* compressedIndex,
                                              const uint2* pairs, ui32 pairCount,
                                              const float* weight,
                                              const TDataPartition* partition,
                                              ui32 partCount,
                                              ui32 histLineSize,
                                              bool fullPass,
                                              float* histogram,
                                              int parallelStreamCount,
                                              TCudaStream stream) {

       ComputePairwiseHistogramOneByte6BitsImpl<false>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                        pairs, pairCount, weight, partition, partCount, histLineSize,
                                                        fullPass,
                                                        histogram,
                                                        parallelStreamCount,
                                                        stream);

        ComputePairwiseHistogramOneByte6BitsImpl<true>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                       pairs, pairCount, weight, partition, partCount, histLineSize,
                                                       fullPass,
                                                       histogram,
                                                       parallelStreamCount, stream);
    }

    void ComputePairwiseHistogramOneByte7Bits(const TCFeature* features,
                                              const TCFeature* featureCpu,
                                              const ui32 featureCount,
                                              const ui32 fiveBitsFeatureCount,
                                              const ui32* compressedIndex,
                                              const uint2* pairs, ui32 pairCount,
                                              const float* weight,
                                              const TDataPartition* partition,
                                              ui32 partCount,
                                              ui32 histLineSize,
                                              bool fullPass,
                                              float* histogram,
                                              int parallelStreamCount,
                                              TCudaStream stream) {

        ComputePairwiseHistogramOneByte7BitsImpl<false>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                        pairs, pairCount, weight, partition, partCount, histLineSize,
                                                        fullPass,
                                                        histogram,
                                                        parallelStreamCount,
                                                        stream);

        ComputePairwiseHistogramOneByte7BitsImpl<true>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                       pairs, pairCount, weight, partition, partCount, histLineSize,
                                                       fullPass,
                                                       histogram,
                                                       parallelStreamCount,
                                                       stream);

    }

    void ComputePairwiseHistogramOneByte8BitAtomics(const TCFeature* features,
                                                    const TCFeature* featureCpu,
                                                    const ui32 featureCount,
                                                    const ui32 fiveBitsFeatureCount,
                                                    const ui32* compressedIndex,
                                                    const uint2* pairs, ui32 pairCount,
                                                    const float* weight,
                                                    const TDataPartition* partition,
                                                    ui32 partCount,
                                                    ui32 histLineSize,
                                                    bool fullPass,
                                                    float* histogram,
                                                    int parallelStreamCount,
                                                    TCudaStream stream) {

        ComputePairwiseHistogramOneByte8BitAtomicsImpl<false>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                              pairs, pairCount, weight, partition, partCount, histLineSize,
                                                              fullPass,
                                                              histogram,
                                                              parallelStreamCount,
                                                              stream);

        ComputePairwiseHistogramOneByte8BitAtomicsImpl<true>(features, featureCpu, featureCount, fiveBitsFeatureCount, compressedIndex,
                                                             pairs, pairCount, weight, partition, partCount, histLineSize,
                                                             fullPass,
                                                             histogram,
                                                             parallelStreamCount,
                                                             stream);
    }



}
