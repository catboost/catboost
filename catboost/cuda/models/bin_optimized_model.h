#pragma once

#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

namespace NCatboostCuda {
    // squared mean of L2 norms of leaf values, used as the default MVS regularization (mvs_reg)
    inline TMaybe<float> CalcL1LeavesSum(TConstArrayRef<float> leafValues, ui32 dim) {
        if (leafValues.empty()) {
            return Nothing();
        }
        const auto numLeaves = leafValues.size() / dim;
        double sumOverLeaves = 0;
        for (auto leaf : xrange(numLeaves)) {
            double w2 = 0;
            for (auto d : xrange(dim)) {
                const double leafValue = leafValues[dim * leaf + d];
                w2 += leafValue * leafValue;
            }
            sumOverLeaves += sqrt(w2);
        }
        return Sqr(sumOverLeaves / numLeaves);
    }

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
