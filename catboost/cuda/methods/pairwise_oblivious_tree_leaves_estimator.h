#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/models/oblivious_model.h>

namespace NCatboostCuda {
    /*
     * Pairwise leaves estimator
     */
    class TPairwiseObliviousTreeLeavesEstimator {
    public:
        TPairwiseObliviousTreeLeavesEstimator() {
            Y_VERIFY(false, "Unimplemented yet");
        }

        template <class TTarget>
        TPairwiseObliviousTreeLeavesEstimator& AddEstimationTask(TTarget&& target,
                                                                 TStripeBuffer<const float>&& current,
                                                                 TObliviousTreeModel* dst) {
            Y_VERIFY(false, "Unimplemented yet");
            return *this;
        }

        void Estimate() {
            Y_VERIFY(false, "Unimplemented yet");
        }
    };
}
