#include "auc.h"

#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>
#include <catboost/cuda/cuda_util/scan.h>
#include <catboost/cuda/cuda_util/reduce.h>
#include <catboost/cuda/cuda_util/transform.h>
#include <catboost/cuda/cuda_util/sort.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/helpers.h>

#include <util/generic/va_args.h>

using NCudaLib::TCudaBuffer;
using NCudaLib::TMirrorMapping;
using NCudaLib::TStripeMapping;

template <class TFloat, class TMapping>
static double ComputeAucImpl(
    const TCudaBuffer<TFloat, TMapping>& target,
    const TCudaBuffer<TFloat, TMapping>& weights,
    const TCudaBuffer<TFloat, TMapping>& cursor) {
    auto singleDevMapping = NCudaLib::TSingleMapping(0, target.GetObjectsSlice().Size());

    auto singleDevTarget = TSingleBuffer<float>::Create(singleDevMapping);
    Reshard(target, singleDevTarget);

    auto singleDevCursor = TSingleBuffer<float>::Create(singleDevMapping);
    Reshard(cursor, singleDevCursor);

    auto singleDevWeights = TSingleBuffer<float>::Create(singleDevMapping);
    Reshard(weights, singleDevWeights);

    double auc = 0;

    auto indices = TSingleBuffer<ui32>::Create(singleDevMapping);

    for (auto aucType : {NCatboostCuda::EAucType::Pessimistic, NCatboostCuda::EAucType::Optimistic}) {
        MakeSequence(indices);

        {
            auto tmp = TSingleBuffer<float>::CopyMapping(singleDevTarget);
            tmp.Copy(singleDevTarget);
            RadixSort(tmp, indices, aucType == NCatboostCuda::EAucType::Optimistic);
        }
        {
            auto tmp = TSingleBuffer<float>::CopyMapping(singleDevCursor);
            Gather(tmp, singleDevCursor, indices);
            RadixSort(tmp, indices);
        }

        auto sortedTarget = TSingleBuffer<float>::Create(singleDevMapping);
        Gather(sortedTarget, singleDevTarget, indices);

        auto sortedWeights = TSingleBuffer<float>::Create(singleDevMapping);
        Gather(sortedWeights, singleDevWeights, indices);

        auto weightsPositive = TSingleBuffer<float>::CopyMapping(sortedWeights);
        weightsPositive.Copy(sortedWeights);
        MultiplyVector(weightsPositive, sortedTarget);

        auto prefixWeightsNegative = TSingleBuffer<float>::CopyMapping(sortedWeights);

        //swap classes
        AddVector(sortedTarget, -1.0f);
        MultiplyVector(sortedTarget, -1.0f);

        MultiplyVector(sortedWeights, sortedTarget);

        ScanVector(sortedWeights, prefixWeightsNegative, true);

        const ui32 totalObservations = prefixWeightsNegative.GetObjectsSlice().Size();
        float negativePairsWeight = 0;

        if (totalObservations) {
            TVector<float> tmp;
            prefixWeightsNegative.SliceView(TSlice(totalObservations - 1, totalObservations)).Read(tmp);
            negativePairsWeight = tmp[0];
        }

        MultiplyVector(prefixWeightsNegative, weightsPositive);

        const float correctPairsWeights = ReduceToHost(prefixWeightsNegative);
        const float positivePairsWeight = ReduceToHost(weightsPositive);

        float denum = positivePairsWeight * negativePairsWeight;
        auc += denum > 0 ? correctPairsWeights / denum : 0;
    }

    return auc / 2;
}

#define Y_CATBOOST_CUDA_F_IMPL_PROXY(x) \
    Y_CATBOOST_CUDA_F_IMPL x

#define Y_CATBOOST_CUDA_F_IMPL(T, TMapping)               \
    template <>                                           \
    double NCatboostCuda::ComputeAUC<T, TMapping>(        \
        const TCudaBuffer<T, TMapping>& target,           \
        const TCudaBuffer<T, TMapping>& weights,          \
        const TCudaBuffer<T, TMapping>& cursor) {         \
        return ::ComputeAucImpl(target, weights, cursor); \
    }

Y_MAP_ARGS(
    Y_CATBOOST_CUDA_F_IMPL_PROXY,
    (float, TStripeMapping),
    (const float, TMirrorMapping),
    (const float, TStripeMapping));

#undef Y_CATBOOST_CUDA_F_IMPL
#undef Y_CATBOOST_CUDA_F_IMPL_PROXY
