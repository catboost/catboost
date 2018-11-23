#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NKernel {

    struct TRemoveQueryBiasContext: public IKernelContext {
        TDevicePointer<float> QueryBias;
    };


    void ComputeGroupIds(const ui32* qSizes, const ui32* qOffsets, ui32 offsetsBias, int qCount, ui32* dst,
                         TCudaStream stream);

    void ComputeGroupMeans(const float* target, const float* weights,
                           const ui32* qOffsets, ui32 qOffsetsBias,
                           const ui32* qSizes, ui32 qCount,
                           float* result, TCudaStream stream);


    void RemoveGroupBias(const float *queryMeans, const ui32 *qids, ui32 size, float *dst, TCudaStream stream);

    void ComputeGroupMeans(const float* target, const float* weights,
                           const ui32* qOffsets,  ui32 qCount,
                           float* result, TCudaStream stream);

    void ComputeGroupMax(const float* target,
                         const ui32* qOffsets,  ui32 qCount,
                         float* result, TCudaStream stream);


    void FillQueryEndMask(const ui32* qids, const ui32* docs, ui32 docCount, ui32* masks, TCudaStream stream);
    void CreateSortKeys(ui64* seeds, ui32 seedSize, const ui32* qids,  ui32 docCount, ui64* keys, TCudaStream stream);

    void FillTakenDocsMask(const float* takenQueryMasks,
                           const ui32* qids,
                           const ui32* docs, ui32 docCount,
                           const ui32* queryOffsets,
                           const ui32 queryOffsetsBias,
                           const ui32* querySizes,
                           const float docwiseSampleRate,
                           const ui32 maxQuerySize,
                           float* takenMask,
                           TCudaStream stream);
}
