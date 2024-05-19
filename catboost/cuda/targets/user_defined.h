#pragma once

#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/libs/metrics/metric.h>

namespace NKernelHost {

    class TUserDefinedMetricKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Targets;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const float> Results;
        TCudaBufferPtr<const float> ResultsWeight;
        const TCustomGpuMetricDescriptor Descriptor;

    public:

        static const size_t BlockSize = 256;
        static const size_t NumBlocks = 64;

        TUserDefinedMetricKernel() = default;

        TUserDefinedMetricKernel(TCudaBufferPtr<const float> targets,
                                 TCudaBufferPtr<const float> weights,
                                 TCudaBufferPtr<const float> predictions,
                                 TCudaBufferPtr<const float> results,
                                 TCudaBufferPtr<const float> results_weight,
                                 const TCustomGpuMetricDescriptor& descriptor
        )
        : Targets(targets)
        , Weights(weights)
        , Predictions(predictions)
        , Results(results)
        , ResultsWeight(results_weight)
        , Descriptor(descriptor)
        {
        }

        inline void Load(IInputStream* s) {
            Y_UNUSED(s);
            ythrow TCatBoostException() << "Distributed training and evaluation is not supported with user defined metrics";
        }

        inline void Save(IOutputStream* s) const {
            Y_UNUSED(s);
            ythrow TCatBoostException() << "Distributed training and evaluation is not supported with user defined metrics";
        }

        void Run(const TCudaStream& stream) const {    
            auto target_ptr = TConstArrayRef<float>(Targets.Get(), Targets.ObjectCount());
            auto weights_ptr = TConstArrayRef<float>(Weights.Get(), Weights.ObjectCount());
            auto cursor_ptr = TConstArrayRef<float>(Predictions.Get(), Predictions.ObjectCount());
            auto result_ptr = TConstArrayRef<float>(Results.Get(), Results.ObjectCount());
            auto result_weight_ptr = TConstArrayRef<float>(ResultsWeight.Get(), ResultsWeight.ObjectCount());
            (*(Descriptor.EvalFunc))(cursor_ptr, target_ptr, weights_ptr, result_ptr, result_weight_ptr, 0, target_ptr.size(), Descriptor.CustomData, stream.GetStream(), BlockSize, NumBlocks);
        }
    };
}