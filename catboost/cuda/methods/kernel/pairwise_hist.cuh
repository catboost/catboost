#pragma once


#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NKernel {

    void UpdatePairwiseHistograms(const ui32 firstFeatureId, const ui32 featureCount,
                                  const TDataPartition* dataParts, ui32 partCount,
                                  ui32 histLineSize,
                                  float* histograms,
                                  TCudaStream stream
    );

    void BuildBinaryFeatureHistograms(const TCFeature* features, ui32 featureCount,
                                      const TDataPartition* partition,
                                      const TPartitionStatistics* partitionStats,
                                      ui32 partCount,
                                      const ui64 histLineSize,
                                      bool fullPass,
                                      float* histogram,
                                      TCudaStream stream);

    void ScanPairwiseHistograms(const TCFeature* features,
                                int featureCount, int partCount,
                                int histLineSize, bool fullPass,
                                float* binSums,
                                TCudaStream stream);



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
                                              TCudaStream stream);

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
                                              TCudaStream stream);
//
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
                                              TCudaStream stream);
//
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
                                                    TCudaStream stream);


    void ComputePairwiseHistogramHalfByte(const TCFeature* features, const TCFeature*,
                                         const ui32 featureCount,
                                         const ui32 halfByteFeatureCount,/* for easier dispatch via macro */
                                         const ui32* compressedIndex,
                                         const uint2* pairs, ui32 pairCount,
                                         const float* weight,
                                         const TDataPartition* partition,
                                         ui32 partCount,
                                         ui32 histLineSize,
                                         bool fullPass,
                                         float* histogram,
                                         int parallelStreams,
                                         TCudaStream stream);

    void ComputePairwiseHistogramBinary(const TCFeature* features,const TCFeature*,
                                         const ui32 featureCount,
                                         const ui32 binaryFeatureCount, /* for easier dispatch via macro */
                                         const ui32* compressedIndex,
                                         const uint2* pairs, ui32 pairCount,
                                         const float* weight,
                                         const TDataPartition* partition,
                                         ui32 partCount,
                                         ui32 histLineSize,
                                         bool fullPass,
                                         float* histogram,
                                         int parallelStreams,
                                         TCudaStream stream);


}
