#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>

namespace NCatboostCuda {
    namespace NDetail {
        template <typename I, typename T, typename TMapping>
        void MakeDcgDecay(
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets,
            NCudaLib::TCudaBuffer<T, TMapping>& decay,
            ui32 stream = 0);

        template <typename I, typename T, typename TMapping>
        void MakeDcgExponentialDecay(
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets,
            T base,
            NCudaLib::TCudaBuffer<T, TMapping>& decay,
            ui32 stream = 0);
    }
}
