#pragma once

#include <catboost/cuda/cuda_lib/kernel.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/cuda/cuda_util/kernel/fill.cuh>

namespace NKernelHost {

    class TUserDefinedObjectiveKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Targets;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const float> ValResults;
        TCudaBufferPtr<const float> Der1Results;
        TCudaBufferPtr<const float> Der2Results;
        const TCustomObjectiveDescriptor Descriptor;
    public:
        static const size_t BlockSize = 1024;

        TUserDefinedObjectiveKernel() = default;

        TUserDefinedObjectiveKernel(const TCustomObjectiveDescriptor descriptor,
                                    TCudaBufferPtr<const float> targets,
                                    TCudaBufferPtr<const float> weights,
                                    TCudaBufferPtr<const float> predictions,
                                    TCudaBufferPtr<const float> valResult,
                                    TCudaBufferPtr<const float> der1Result,
                                    TCudaBufferPtr<const float> der2Result
        )
        : Targets(targets)
        , Weights(weights)
        , Predictions(predictions)
        , ValResults(valResult)
        , Der1Results(der1Result)
        , Der2Results(der2Result)
        , Descriptor(descriptor)
        {
        }

        inline void Load(IInputStream* s) {
            Y_UNUSED(s);
            CB_ENSURE(false, "Distributed training and evaluation is not supported with user defined metrics");
        }

        inline void Save(IOutputStream* s) const {
            Y_UNUSED(s);
            CB_ENSURE(false, "Distributed training and evaluation is not supported with user defined metrics");
        }

        void Run(const TCudaStream& stream) const {
            auto target_ptr = TConstArrayRef<float>(Targets.Get(), Targets.ObjectCount());
            auto weights_ptr = TConstArrayRef<float>(Weights.Get(), Weights.ObjectCount());
            auto predictions_ptr = TConstArrayRef<float>(Predictions.Get(), Predictions.ObjectCount());
            auto val_results_ptr = TConstArrayRef<float>(ValResults.Get(), ValResults.ObjectCount());
            auto der1_results_ptr = TConstArrayRef<float>(Der1Results.Get(), Der1Results.ObjectCount());
            auto der2_results_ptr = TConstArrayRef<float>(Der2Results.Get(), Der2Results.ObjectCount());
            size_t totalObjects = Targets.ObjectCount();
            size_t NumBlocks = (totalObjects + BlockSize - 1) / BlockSize;
            (*(Descriptor.GpuCalcDersRange))(predictions_ptr, target_ptr, weights_ptr, val_results_ptr, der1_results_ptr, der2_results_ptr, totalObjects, Descriptor.CustomData, stream.GetStream(), BlockSize, NumBlocks);
        }
    };

    class TUserDefinedMetricKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Targets;
        TCudaBufferPtr<const float> Weights;
        TCudaBufferPtr<const float> Predictions;
        TCudaBufferPtr<const float> Results;
        TCudaBufferPtr<const float> ResultsWeight;
        const TCustomMetricDescriptor Descriptor;

    public:

        static const size_t BlockSize = 256;
        static const size_t NumBlocks = 64;

        TUserDefinedMetricKernel() = default;

        TUserDefinedMetricKernel(TCudaBufferPtr<const float> targets,
                                 TCudaBufferPtr<const float> weights,
                                 TCudaBufferPtr<const float> predictions,
                                 TCudaBufferPtr<const float> results,
                                 TCudaBufferPtr<const float> results_weight,
                                 const TCustomMetricDescriptor& descriptor
        )
        : Targets(targets)
        , Weights(weights)
        , Predictions(predictions)
        , Results(results)
        , ResultsWeight(results_weight)
        , Descriptor(descriptor)
        {
        }

        inline void Load(IInputStream*) {
            CB_ENSURE(false, "Distributed training and evaluation is not supported with user defined metrics");
        }

        inline void Save(IOutputStream*) const {
            CB_ENSURE(false, "Distributed training and evaluation is not supported with user defined metrics");
        }

        void Run(const TCudaStream& stream) const {
            auto target_ptr = TConstArrayRef<float>(Targets.Get(), Targets.ObjectCount());
            auto weights_ptr = TConstArrayRef<float>(Weights.Get(), Weights.ObjectCount());
            auto cursor_ptr = TConstArrayRef<float>(Predictions.Get(), Predictions.ObjectCount());
            auto result_ptr = TConstArrayRef<float>(Results.Get(), Results.ObjectCount());
            auto result_weight_ptr = TConstArrayRef<float>(ResultsWeight.Get(), ResultsWeight.ObjectCount());
            (*(Descriptor.GpuEvalFunc))(cursor_ptr, target_ptr, weights_ptr, result_ptr, result_weight_ptr, 0, target_ptr.size(), Descriptor.CustomData, stream.GetStream(), BlockSize, NumBlocks);
        }
    };
}
