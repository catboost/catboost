#include "pairwise_hist_one_byte_8bit_atomics.cuh"

namespace NKernel {
    template
    void ComputePairwiseHistogramOneByte8BitAtomicsImpl<true>(const TCFeature* features,
                                                               const ui32 featureCount,
                                                               const ui32 sixBitsFeatureCount,
                                                               const ui32* compressedIndex,
                                                               const uint2* pairs, ui32 pairCount,
                                                               const float* weight,
                                                               const TDataPartition* partition,
                                                               ui32 partCount,
                                                               ui32 histLineSize,
                                                               bool fullPass,
                                                               float* histogram,
                                                               TCudaStream stream);


}
