#pragma once

#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>

namespace NCatboostCuda {
    class IBinOptimizedModel {
    public:
        virtual ~IBinOptimizedModel() {
        }

        virtual ui32 BinCount() const = 0;

        virtual ui32 OutputDim() const = 0;

        virtual void Rescale(double scale) = 0;
        virtual void ShiftLeafValues(double shift) = 0;

        virtual void UpdateWeights(const TVector<double>& newWeights) = 0;
        virtual void UpdateLeaves(const TVector<float>& newLeaves) = 0;

        virtual void ComputeBins(const TDocParallelDataSet& dataSet,
                                 TStripeBuffer<ui32>* dst) const = 0;
    };
}
