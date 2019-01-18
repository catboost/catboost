#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>

namespace NCatboostCuda {
    template <class TMapping = NCudaLib::TMirrorMapping>
    struct TL2Target {
        TCudaBuffer<float, TMapping> WeightedTarget;
        TCudaBuffer<float, TMapping> Weights;
    };

    template <class TMapping>
    inline void GatherTarget(TCudaBuffer<float, TMapping>& weightedTarget,
                             TCudaBuffer<float, TMapping>& weights,
                             const TL2Target<TMapping>& from,
                             const TCudaBuffer<ui32, TMapping>& indices) {
        auto guard = NCudaLib::GetCudaManager().GetProfiler().Profile("Gather target and weights");

        weights.Reset(from.Weights.GetMapping());
        weightedTarget.Reset(from.WeightedTarget.GetMapping());

        CB_ENSURE(weights.GetObjectsSlice() == from.Weights.GetObjectsSlice());
        CB_ENSURE(weights.GetObjectsSlice() == indices.GetObjectsSlice());

        Gather(weightedTarget, from.WeightedTarget, indices);
        Gather(weights, from.Weights, indices);
    }

    template <class TMapping>
    inline TBestSplitProperties BestSplit(const TCudaBuffer<TBestSplitProperties, TMapping>& optimalSplits, ui32 stream = 0) {
        TVector<TBestSplitProperties> best;
        optimalSplits.Read(best, stream);
        TBestSplitProperties minScr = best[0];

        for (auto scr : best) {
            if (scr < minScr) {
                minScr = scr;
            }
        }
        return minScr;
    }

}
