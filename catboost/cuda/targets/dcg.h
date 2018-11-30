#pragma once

#include <catboost/cuda/cuda_lib/fwd.h>
#include <catboost/libs/options/enums.h>

#include <util/generic/array_ref.h>
#include <util/generic/maybe.h>
#include <util/generic/utility.h>

namespace NCatboostCuda {
    // Calculate sum of per-query NDCGs.
    //
    // @param sizes             Array of per-query document counts.
    // @param offsets           Array of per-query offsets of documents (NOTE: offsets are "biased",
    //                          e.g. they are given within a device).
    // @param weights           Per-document weights (weight for each document withing query is
    //                          identical; if you have no weights initialize it with ones).
    // @param targets           Ideal document relevance (e.g. from a dataset)
    // @param approxes          Predicted document relevance (e.g. from a trained model)
    // @param type              How to treat relevances, if type is `Exp` relevance will be
    //                          exponentiated ($$2^relevance - 1$$), otherwise relevance will be
    //                          keps as-is.
    //
    // @return                  Weighted sums of per-query NDCGs
    //
    // NOTE: sum(sizes) == len(targets)
    template <typename TMapping>
    TVector<float> CalculateNdcg(
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& sizes,
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& offsets,
        const NCudaLib::TCudaBuffer<const float, TMapping>& weights,
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
        const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TConstArrayRef<ui32> topSizes = {},
        ui32 stream = 0);

    // Calculate sum of per-query IDCGs.
    //
    // @param sizes             Array of per-query document counts.
    // @param offsets           Array of per-query offsets of documents (NOTE: offsets are "biased",
    //                          e.g. they are given within a device).
    // @param weights           Per-document weights (weight for each document withing query is
    //                          identical; if you have no weights initialize it with ones).
    // @param targets           Ideal document relevance (e.g. from a dataset)
    // @param type              How to treat relevances, if type is `Exp` relevance will be
    //                          exponentiated ($$2^relevance - 1$$), otherwise relevance will be
    //                          keps as-is.
    // @param exponentialDecay  If defined instead of a classic decay ($$1/log2(position + 1)$$)
    //                          will use exponential decay ($$exponentialDecay^(position-1)$$).
    //
    // @return                  Weighted sums of per-query IDCGs
    //
    // NOTE: sum(sizes) == len(targets)
    template <typename TMapping>
    TVector<float> CalculateIdcg(
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& sizes,
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& offsets,
        const NCudaLib::TCudaBuffer<const float, TMapping>& weights,
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TMaybe<float> exponentialDecay = Nothing(),
        TConstArrayRef<ui32> topSizes = {},
        ui32 stream = 0);


    // Calculate sum of per-query DCGs.
    //
    // @param sizes             Array of per-query document counts.
    // @param offsets           Array of per-query offsets of documents (NOTE: offsets are "biased",
    //                          e.g. they are given within a device).
    // @param weights           Per-document weights (weight for each document withing query is
    //                          identical; if you have no weights initialize it with ones).
    // @param targets           Ideal document relevance (e.g. from a dataset)
    // @param approxes          Predicted document relevance (e.g. from a trained model)
    // @param type              How to treat relevances, if type is `Exp` relevance will be
    //                          exponentiated ($$2^relevance - 1$$), otherwise relevance will be
    //                          keps as-is.
    // @param exponentialDecay  If defined instead of a classic decay ($$1/log2(position + 1)$$)
    //                          will use exponential decay ($$exponentialDecay^(position-1)$$).
    //
    // @return                  Weighted sums of per-query DCGs
    //
    // NOTE: sum(sizes) == len(targets)
    template <typename TMapping>
    TVector<float> CalculateDcg(
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& sizes,
        const NCudaLib::TCudaBuffer<const ui32, TMapping>& offsets,
        const NCudaLib::TCudaBuffer<const float, TMapping>& weights,
        const NCudaLib::TCudaBuffer<const float, TMapping>& targets,
        const NCudaLib::TCudaBuffer<const float, TMapping>& approxes,
        ENdcgMetricType type = ENdcgMetricType::Base,
        TMaybe<float> exponentialDecay = Nothing(),
        TConstArrayRef<ui32> topSizes = {},
        ui32 stream = 0);

    namespace NDetail {
        template <typename I, typename T, typename TMapping>
        void MakeDcgDecays(
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets,
            NCudaLib::TCudaBuffer<T, TMapping>& decays,
            ui32 stream = 0);

        template <typename I, typename T, typename TMapping>
        void MakeDcgExponentialDecays(
            const NCudaLib::TCudaBuffer<I, TMapping>& biasedOffsets,
            T base,
            NCudaLib::TCudaBuffer<T, TMapping>& decays,
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

        // sizes = [2, 3, 4]
        // offsets = [0, 2, 5]
        // elementwiseOffsets = [0, 0, 2, 2, 2, 5, 5, 5, 5]
        template <typename T, typename TMapping>
        void MakeElementwiseOffsets(
            const NCudaLib::TCudaBuffer<T, TMapping>& sizes,
            const NCudaLib::TCudaBuffer<T, TMapping>& offsets,
            NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& elementwiseOffsets,
            ui32 stream = 0);

        // Equal to:
        //
        // for (size_t i = 0; i < sizes.size(); ++i) {
        //     if (i == 0) {
        //         endOfGroupMarkers[i] = 1;
        //     }
        //
        //     if (const auto j = offsets[i] + sizes[i]; j < endOfGroupMarkers.size()) {
        //         endOfGroupMarkers[j] = 1;
        //     }
        // }
        //
        // NOTE: endOfGroupMarkers must be initialized with zeroes
        template <typename T, typename TMapping>
        void MakeEndOfGroupMarkers(
            const NCudaLib::TCudaBuffer<T, TMapping>& sizes,
            const NCudaLib::TCudaBuffer<T, TMapping>& offsets,
            NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& endOfGroupMarkers,
            ui32 stream = 0);

        // Equal to:
        //
        // for (size_t i = 0; i < sizes.size(); ++i) {
        //     dst[i] = src[offsets[i] + min(sizes[i], maxSize) - 1];
        // }
        template <typename T, typename I, typename TMapping>
        void GatherBySizeAndOffset(
            const NCudaLib::TCudaBuffer<T, TMapping>& src,
            const NCudaLib::TCudaBuffer<I, TMapping>& sizes,
            const NCudaLib::TCudaBuffer<I, TMapping>& offsets,
            NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& dst,
            std::remove_const_t<I> maxSize = Max<std::remove_const_t<I>>(),
            ui32 stream = 0);

        // Equal to:
        //
        // for (size_t i = 0; i < sizes.size(); ++i) {
        //     const auto mean = accumulate(
        //          values.begin() + offsets[i],
        //          values.begin() + offsets[i] + sizes[i],
        //          0) / sizes[i];
        //     for_each(
        //          values.begin() + offsets[i],
        //          values.begin() + offsets[i] + sizes[i],
        //          [mean](auto& value) { value -= mean; });
        // }
        template <typename T, typename I, typename TMapping>
        void RemoveGroupMean(
            const NCudaLib::TCudaBuffer<T, TMapping>& values,
            const NCudaLib::TCudaBuffer<I, TMapping>& sizes,
            const NCudaLib::TCudaBuffer<I, TMapping>& offsets,
            NCudaLib::TCudaBuffer<std::remove_const_t<T>, TMapping>& normalized,
            ui32 stream = 0);
    }
}
