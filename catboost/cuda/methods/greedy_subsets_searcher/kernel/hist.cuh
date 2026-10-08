#pragma once

#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

/*
 * All routines here assume histograms are zeroed externally
 */

namespace NKernel {

    // Replicate numBlocks.x across the GPU. The histogram launchers inlined
    // `numBlocks.x *= CeilDivide(targetBlocks, numBlocks.x*numBlocks.y*numBlocks.z)`
    // and only checked IsGridEmpty *after* it, so a zero dimension (e.g.
    // partCount == 0) divided by zero -> host SIGFPE before the guard fired. The
    // NVIDIA path keeps the exact original arithmetic (this forced-inline helper
    // folds to the same multiply); only HIP adds the active>0 guard. The
    // divide-by-zero is a pre-existing all-platform bug that catboost's
    // TPointwiseMultiStatHistogramTest exercises.
    static __forceinline__ __host__ void ScaleBlockCountToOccupancy(dim3& numBlocks, int targetBlocks) {
        const int active = (int)(numBlocks.x * numBlocks.y * numBlocks.z);
#if defined(USE_HIP)
        if (active > 0) {
            numBlocks.x *= (targetBlocks + active - 1) / active;
        }
#else
        numBlocks.x *= CeilDivide(targetBlocks, active);
#endif
    }

    void ComputeHistBinary(const TFeatureInBlock* features,
                           const int fCount,
                           const TDataPartition* parts,
                           const ui32* partIds,
                           ui32 partCount,
                           const ui32* cindex,
                           const int* indices,
                           const float* stats,
                           ui32 numStats,
                           ui32 statLineSize,
                           float* histograms,
                           TCudaStream stream);

    void ComputeHistBinary(const TFeatureInBlock* features,
                           const int fCount,
                           const TDataPartition* parts,
                           const ui32* partIds,
                           ui32 partCount,
                           const ui32* bins,
                           ui32 binsLineSize,
                           const float* stats,
                           ui32 numStats,
                           ui32 statLineSize,
                           float* histograms,
                           TCudaStream stream);



    void ComputeHistOneByte(int maxBins,
                            const TFeatureInBlock* features,
                            const int fCount,
                            const TDataPartition* parts,
                            const ui32* partIds,
                            ui32 partCount,
                            const ui32* bins,
                            ui32 binsLineSize,
                            const float* stats,
                            ui32 numStats,
                            ui32 statLineSize,
                            float* histograms,
                            TCudaStream stream);


    void ComputeHistOneByte(int maxBins,
                            const TFeatureInBlock* groups,
                            const int fCount,
                            const TDataPartition* parts,
                            const ui32* partIds,
                            ui32 partCount,
                            const ui32* cindex,
                            const int* indices,
                            const float* stats,
                            ui32 numStats,
                            ui32 statLineSize,
                            float* histograms,
                            TCudaStream stream);




    void ComputeHistHalfByte(const TFeatureInBlock* features,
                             const int fCount,
                             const TDataPartition* parts,
                             const ui32* partIds,
                             ui32 partCount,
                             const ui32* bins,
                             ui32 binsLineSize,
                             const float* stats,
                             ui32 numStats,
                             ui32 statLineSize,
                             float* histograms,
                             TCudaStream stream);

    void ComputeHistHalfByte(const TFeatureInBlock* features,
                             const int fCount,
                             const TDataPartition* parts,
                             const ui32* partIds,
                             ui32 partCount,
                             const ui32* cindex,
                             const int* indices,
                             const float* stats,
                             ui32 numStats,
                             ui32 statLineSize,
                             float* histograms,
                             TCudaStream stream);














}
