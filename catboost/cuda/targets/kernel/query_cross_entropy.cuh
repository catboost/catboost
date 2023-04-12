#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/private/libs/options/enums.h>


namespace NKernel {

    struct TQueryLogitContext : public IKernelContext {
        TDevicePointer<int> QidCursor;
    };

    void QueryCrossEntropy(int* qidCursor, const ui32 qCount,
                           const float alpha,
                           const float* targets,
                           const float* weights,
                           const float* values,
                           const ui32* qids,
                           const bool* isSingleClassQueries,
                           const ui32* qOffsets,
                           const float* approxScale,
                           const float defaultScale,
                           const ui32 approxScaleSize,
                           const ui32* trueClassCount,
                           const ui32 docCount,
                           float* functionValue,
                           float* ders,
                           float* ders2llp,
                           float* ders2llmax,
                           float* groupDers2,
                           TCudaStream stream);

    void ComputeQueryLogitMatrixSizes(const ui32* queryOffsets,
                                      const bool* isSingleClassQuery,
                                      ui32 qCount,
                                      ui32* matrixSize,
                                      TCudaStream stream);

    void MakeQueryLogitPairs(const ui32* qOffsets,
                             const ui64* matrixOffset,
                             const bool* isSingleQueryFlags,
                             double meanQuerySize,
                             ui32 qCount,
                             uint2* pairs,
                             TCudaStream stream);

    void MakeIsSingleClassFlags(const float* targets,
                                const ui32* loadIndices,
                                const ui32* queryOffsets,
                                ui32 queryCount,
                                double meanQuerySize,
                                bool* isSingleClassQuery,
                                ui32* trueClassCount,
                                TCudaStream stream);



    void FillPairDer2AndRemapPairDocuments(const float* ders2,
                                           const float* groupDers2,
                                           const ui32* docIds,
                                           const ui32* qids,
                                           ui64 pairCount,
                                           float* pairDer2,
                                           uint2* pairs,
                                           TCudaStream stream
    );
}
