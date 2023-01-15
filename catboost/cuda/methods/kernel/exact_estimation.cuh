#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel {

    void ComputeNeedWeights(const float* targets,
                            const float* weights,
                            ui32 objectsCount,
                            ui32 binCount,
                            const ui32* beginOffsets,
                            const ui32* endOffsets,
                            float* needWeights,
                            float alpha,
                            TCudaStream stream);

    void ComputeWeightsWithTargets(const float* targets,
                                   const float* weights,
                                   float* weightsWithTargets,
                                   ui32 objectsCount,
                                   TCudaStream stream);

    void ComputeWeightedQuantileWithBinarySearch(const float* targets,
                                                 const float* weightsPrefixSum,
                                                 ui32 objectsCount,
                                                 const float* needWeights,
                                                 const ui32* beginOffsets,
                                                 const ui32* endOffsets,
                                                 ui32 binCount,
                                                 float* point,
                                                 float alpha,
                                                 ui32 binarySearchIterations,
                                                 TCudaStream stream);

    void MakeEndOfBinsFlags(const ui32* beginOffsets,
                            const ui32* endOffsets,
                            ui32 binCount,
                            ui32* flags,
                            TCudaStream stream);
};
