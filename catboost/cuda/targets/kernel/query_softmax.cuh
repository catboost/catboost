#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>

namespace NKernel
{
    struct TQuerySoftMaxContext: public IKernelContext {
        TDevicePointer<float> ApproxExp;
        TDevicePointer<float> QueryApprox;
        TDevicePointer<float> QuerySumWeightedTargets;
        TDevicePointer<ui32> Qids;
    };

    void ComputeGroupMaximals(const float* target, const float* weights,
                              const float* approxExp,
                              const ui32* qOffsets, ui32 qOffsetsBis,
                              const ui32* qSizes, ui32 qCount,
                              float* maximals, float* sumWeightedTargets,
                              TCudaStream stream);

    void ComputeQueryExponents(const float* weights,
                               const ui32* qids, ui32 size,
                               const float* maximals,
                               const ui32* writeMap,
                               float* approxExp,
                               float beta,
                               TCudaStream stream);

    void ComputeGroupSums(const float* data,
                          const ui32* qOffsets, ui32 qOffsetsBias,
                          const ui32* qSizes, ui32 qCount,
                          float* groupSums, TCudaStream stream);

    void ApproximateQuerySoftMax(const float* target, const float* weights,
                                 const float* approxExp,
                                 const ui32* qids,
                                 float lambdaReg, float beta, ui32 size,
                                 const float* approxExpSum,
                                 const float* sumWeightedTargets,
                                 const ui32* writeMap,
                                 float* functionValue,
                                 float* der,
                                 float* der2,
                                 TCudaStream stream);

}
