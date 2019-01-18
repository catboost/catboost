#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>

namespace NCatboostCuda {
    enum class EAucType {
        Pessimistic,
        Optimistic
    };

    template <class TFloat, class TMapping>
    double ComputeAUC(
        const TCudaBuffer<TFloat, TMapping>& target,
        const TCudaBuffer<TFloat, TMapping>& weights,
        const TCudaBuffer<TFloat, TMapping>& cursor);
}
