#include "multiclass_targets.h"
#include "multiclass_kernels.h"

#include <catboost/cuda/cuda_util/algorithm.h>

namespace NCatboostCuda {
    TAdditiveStatistic TMultiClassificationTargets<NCudaLib::TStripeMapping>::ComputeStats(const TStripeBuffer<const float>& point) const {
        TVector<float> result;
        auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

        ComputeValueAndFirstDer(GetTarget().GetTargets(), GetTarget().GetWeights(), point, &tmp, (TVec*)nullptr);

        NCudaLib::TCudaBufferReader<TVec>(tmp)
            .SetFactorSlice(TSlice(0, 1))
            .SetReadSlice(TSlice(0, 1))
            .ReadReduce(result);

        const double weight = GetTotalWeight();
        return MakeSimpleAdditiveStatistic(-result[0], weight);
    }

    void TMultiClassificationTargets<NCudaLib::TStripeMapping>::StochasticDer(const TStripeBuffer<const float>& point,
                                                                              const TStripeBuffer<float>& sampledWeights,
                                                                              TStripeBuffer<ui32>&& sampledIndices,
                                                                              bool secondDerAsWeights,
                                                                              TOptimizationTarget* target) const {
        CB_ENSURE(!secondDerAsWeights, "MultiClass loss doesn't support second derivatives in tree structure search currently");
        auto gatheredTarget = TVec::CopyMapping(sampledWeights);
        Gather(gatheredTarget, GetTarget().GetTargets(), sampledIndices);
        ui32 statCount = 1 + NumClasses;
        if (Type == ELossFunction::MultiClass) {
            statCount -= 1;
            target->MultiLogitOptimization = true;
        }

        target->StatsToAggregate.Reset(sampledWeights.GetMapping(), statCount);
        auto weights = target->StatsToAggregate.ColumnsView(0);
        Gather(weights, GetTarget().GetWeights(), sampledIndices);
        MultiplyVector(weights, sampledWeights);

        auto ders = target->StatsToAggregate.ColumnsView(TSlice(1, statCount));
        if (Type == ELossFunction::MultiClass) {
            MultiLogitValueAndDer(gatheredTarget.ConstCopyView(), weights.ConstCopyView(), point, &sampledIndices,
                                  NumClasses, (TVec*)nullptr, &ders);
        } else if (Type == ELossFunction::MultiClassOneVsAll) {
            MultiClassOneVsAllValueAndDer(gatheredTarget.ConstCopyView(), weights.ConstCopyView(), point, &sampledIndices,
                                          NumClasses, (TVec*)nullptr, &ders);
        } else if (Type == ELossFunction::RMSEWithUncertainty) {
            CB_ENSURE(NumClasses == 2, "Expect two-dimensional predictions");
            RMSEWithUncertaintyValueAndDer(gatheredTarget.ConstCopyView(), weights.ConstCopyView(), point, &sampledIndices,
                                  (TVec*)nullptr, &ders);
        } else {
            CB_ENSURE(false, "Bug");
        }
        target->Indices = std::move(sampledIndices);
    }

    void TMultiClassificationTargets<NCudaLib::TStripeMapping>::ComputeValueAndFirstDer(const TStripeBuffer<const float>& target,
                                                                                        const TStripeBuffer<const float>& weights,
                                                                                        const TStripeBuffer<const float>& point,
                                                                                        TStripeBuffer<float>* value,
                                                                                        TStripeBuffer<float>* der,
                                                                                        ui32 stream) const {
        if (Type == ELossFunction::MultiClass) {
            MultiLogitValueAndDer(target, weights, point, (const TStripeBuffer<ui32>*)nullptr, NumClasses, value, der,
                                  stream);
        } else if (Type == ELossFunction::MultiClassOneVsAll) {
            MultiClassOneVsAllValueAndDer(target, weights, point, (const TStripeBuffer<ui32>*)nullptr, NumClasses, value, der,
                                          stream);
        } else if (Type == ELossFunction::RMSEWithUncertainty) {
            CB_ENSURE(NumClasses == 2, "Expect two-dimensional predictions");
            RMSEWithUncertaintyValueAndDer(target, weights, point, (const TStripeBuffer<ui32>*)nullptr, value, der,
                                  stream);
        } else {
            CB_ENSURE(false, "Unsupported loss " << Type);
        }
    }

    void TMultiClassificationTargets<NCudaLib::TStripeMapping>::ComputeSecondDerLine(const TStripeBuffer<const float>& target,
                                                                                     const TStripeBuffer<const float>& weights,
                                                                                     const TStripeBuffer<const float>& point,
                                                                                     ui32 row,
                                                                                     TStripeBuffer<float>* der,
                                                                                     ui32 stream) const {
        switch (Type) {
            case ELossFunction::MultiClass: {
                MultiLogitSecondDerRow(target, weights, point, NumClasses, row, der, stream);
                break;
            }
            case ELossFunction::MultiClassOneVsAll: {
                CB_ENSURE(row == 0, "THIS IS A BUG: report to catboost team");
                MultiClassOneVsAllSecondDer(target, weights, point, NumClasses, der, stream);
                break;
            }
            case ELossFunction::RMSEWithUncertainty: {
                CB_ENSURE(NumClasses == 2, "Expect two-dimensional predictions");
                RMSEWithUncertaintySecondDerRow(target, weights, point, row, der, stream);
                break;
            }
            default: {
                CB_ENSURE(false, "Unsupported loss " << Type);
            }
        }
    }

}
