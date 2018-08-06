#include "multiclass_targets.h"
#include "multiclass_kernels.h"
#include <catboost/cuda/cuda_util/fill.h>


namespace NCatboostCuda {

    TAdditiveStatistic TMultiClassificationTargets<NCudaLib::TStripeMapping>::ComputeStats(const TStripeBuffer<const float>& point) const {
        TVector<float> result;
        auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

        MultiLogitValueAndDer(GetTarget().GetTargets(),
                              GetTarget().GetWeights(),
                              point,
                             (const TBuffer<ui32>*)nullptr,
                             NumClasses,
                             &tmp,
                             (TVec*) nullptr);

        NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

        const double weight = GetTotalWeight();
        return MakeSimpleAdditiveStatistic(result[0], weight);
    }

    void TMultiClassificationTargets<NCudaLib::TStripeMapping>::StochasticDer(const TStripeBuffer<const float>& point,
                                                                              const TStripeBuffer<float>& sampledWeights,
                                                                              TStripeBuffer<ui32>&& sampledIndices,
                                                                              bool secondDerAsWeights,
                                                                              TOptimizationTarget* target) const {

        target->MultiLogitOptimization = true;
        CB_ENSURE(!secondDerAsWeights, "MultiClass losss doesn't support second derivatives in tree structure search currently");
        auto gatheredTarget = TVec::CopyMapping(sampledWeights);
        Gather(gatheredTarget, GetTarget().GetTargets(), sampledIndices);

        const ui32 numClasses = 1 + point.GetColumnCount();
        CB_ENSURE(numClasses == NumClasses, numClasses << " ≠ " << NumClasses);

        target->StatsToAggregate.Reset(sampledWeights.GetMapping(), 1 + numClasses - 1);
        auto weights = target->StatsToAggregate.ColumnsView(0);
        Gather(weights, GetTarget().GetWeights(), sampledIndices);
        MultiplyVector(weights, sampledWeights);

        auto ders = target->StatsToAggregate.ColumnsView(TSlice(1, numClasses));
        MultiLogitValueAndDer(gatheredTarget.ConstCopyView(), weights.ConstCopyView(), point, &sampledIndices, numClasses, (TVec*)nullptr, &ders);
        target->Indices = std::move(sampledIndices);
    }

    void TMultiClassificationTargets<NCudaLib::TStripeMapping>::ComputeValueAndFirstDer(const TStripeBuffer<const float>& target,
                                                                                        const TStripeBuffer<const float>& weights,
                                                                                        const TStripeBuffer<const float>& point,
                                                                                        TStripeBuffer<float>* value,
                                                                                        TStripeBuffer<float>* der,
                                                                                        ui32 stream) const {

        MultiLogitValueAndDer(target, weights, point, (const TStripeBuffer<ui32>*)nullptr, NumClasses, value, der, stream);
    }

    void TMultiClassificationTargets<NCudaLib::TStripeMapping>::ComputeSecondDerLine(const TStripeBuffer<const float>& target,
                                                                                     const TStripeBuffer<const float>& weights,
                                                                                     const TStripeBuffer<const float>& point,
                                                                                     ui32 row,
                                                                                     TStripeBuffer<float>* der,
                                                                                     ui32 stream) const {
        MultiLogitSecondDerRow(target, weights, point, NumClasses, row, der, stream);
    }

}
