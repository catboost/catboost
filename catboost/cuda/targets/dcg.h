#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>
#include <catboost/libs/options/enums.h>
#include <util/generic/maybe.h>

namespace NCatboostCuda {
    template <typename TMapping>
    float CalculateNdcg(
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
        const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TMaybe<float> exponentialDecay = Nothing(),
        ui32 stream = 0);

    template <typename TMapping>
    float CalculateIdcg(
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TMaybe<float> exponentialDecay = Nothing(),
        ui32 stream = 0);

    template <typename TMapping>
    float CalculateDcg(
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
        const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& biasedOffsets,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TMaybe<float> exponentialDecay = Nothing(),
        ui32 stream = 0);

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

        template <typename I, typename T, typename TMapping>
        void FuseUi32AndFloatIntoUi64(
            const NCudaLib::TCudaBuffer<I, TMapping>& ui32s,
            const NCudaLib::TCudaBuffer<T, TMapping>& floats,
            NCudaLib::TCudaBuffer<ui64, TMapping>& fused,
            bool negateFloats = false,
            ui32 stream = 0);

        template <typename I, typename T, typename TMapping>
        void FuseUi32AndTwoFloatsIntoUi64(
            const NCudaLib::TCudaBuffer<I, TMapping>& ui32s,
            const NCudaLib::TCudaBuffer<T, TMapping>& floats1,
            const NCudaLib::TCudaBuffer<T, TMapping>& floats2,
            NCudaLib::TCudaBuffer<ui64, TMapping>& fused,
            bool negateFloats1 = false,
            bool negateFloats2 = false,
            ui32 stream = 0);

        // TODO(yazevnul): this one doesn't seem to be needed
        template <typename T, typename U, typename TMapping>
        void GetBits(
            const NCudaLib::TCudaBuffer<T, TMapping>& src,
            NCudaLib::TCudaBuffer<U, TMapping>& dst,
            ui32 bitsOffset,
            ui32 bitsCount,
            ui32 stream = 0);
    }
}

