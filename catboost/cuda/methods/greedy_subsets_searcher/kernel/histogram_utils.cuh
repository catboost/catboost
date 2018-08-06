#pragma once
#include <catboost/cuda/cuda_lib/kernel/kernel.cuh>
#include <catboost/cuda/cuda_util/gpu_data/partitions.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NKernel
{



    void CopyHistograms(const ui32* leftLeaves,
                        const ui32* rightLeaves,
                        const ui32 leavesCount,
                        ui32 numStats,
                        ui32 binFeaturesInHist,
                        float* histograms,
                        TCudaStream stream
    );


    void WriteReducesHistograms(int blockOffset,
                                int histBlockSize,
                                const ui32* histogramIds,
                                ui32 leafCount,
                                ui32 statCount,
                                const float* blockHistogram,
                                const int binFeatureCount,
                                float* dstHistogram,
                                TCudaStream stream);

    void ZeroHistograms(const ui32* histIds,
                        ui32 idsCount,
                        ui32 statCount,
                        const int binFeatureCount,
                        float* dstHistogram,
                        TCudaStream stream);


    void SubstractHistgorams(const ui32* fromIds,
                             const ui32* whatIds,
                             const int idsCount,
                             const int statCount,
                             const int binFeatureCount,
                             float* dstHistogram,
                             TCudaStream stream);



    void ScanHistgorams(const TBinarizedFeature* features, int fCount,
                        const ui32* ids,
                        const int idsCount,
                        const int statCount,
                        const int binFeatureCount,
                        float* histograms,
                        TCudaStream stream);


}
