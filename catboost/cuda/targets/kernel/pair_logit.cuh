#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel
{

    struct TPairLogitContext: public IKernelContext {
        TDevicePointer<float> GatheredPoint;
    };


    void PairLogitPointwiseTarget(const float* point,
                                 const uint2* pairs, const float* pairWeights,
                                 const ui32* writeMap,
                                 ui32 pairCount, int pairShift,
                                 float* functionValue,
                                 float* der,
                                 float* der2,
                                 ui32 docCount,
                                 TCudaStream stream);

    void MakePairWeights(const uint2* pairs, const float* pairWeights, ui32 pairCount,
                         float* weights, TCudaStream stream);

    void PairLogitPairwise(const float* point,
                           const uint2* pairs,
                           const float* pairWeights,
                           const ui32* scatterDerIndices,
                           float* value,
                           float* pointDer,
                           ui32 docCount,
                           float* pairDer2,
                           ui32 pairCount,
                           TCudaStream stream);



    void RemoveOffsetsBias(ui32 bias,
                           ui32 nzPairCount,
                           uint2* nzPairs,
                           TCudaStream stream);

}
