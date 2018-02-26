#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NKernel {

    void ComputeQuantileBorders(const float* values, ui32 size, float* borders, ui32 bordersCount, TCudaStream stream);
    void ComputeUniformBorders(const float* values, ui32 size, float* borders, ui32 bordersCount, TCudaStream stream);
    void FastGpuBorders(const float* values, ui32 size, float* borders, ui32 bordersCount, TCudaStream stream);
    void BinarizeFloatFeature(const float* values, ui32 docCount,
                              const float* borders,
                              TCFeature feature,
                              ui32* dst,
                              const ui32* gatherIndex,
                              bool atomicUpdate,
                              TCudaStream stream);

    void WriteCompressedIndex(TCFeature feature,
                              const ui8* bins, ui32 docCount,
                              ui32* cindex,
                              TCudaStream stream);

}
