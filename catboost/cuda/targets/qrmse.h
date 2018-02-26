#pragma once

#include "quality_metric_helpers.h"
#include "target_func.h"
#include "kernel.h"
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>

namespace NCatboostCuda {
    template <class TDocLayout,
              class TDataSet>
    class TQueryRMSE: public TQuerywiseTarget<TDocLayout, TDataSet> {
    public:
        using TParent = TQuerywiseTarget<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TQueryRMSE(const TDataSet& dataSet,
                   TRandom& random,
                   TSlice slice,
                   const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::QueryRMSE);
        }

        TQueryRMSE(const TDataSet& dataSet,
                   TRandom& random,
                   const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::QueryRMSE);
        }

        TQueryRMSE(const TQueryRMSE& target,
                   const TSlice& slice)
            : TParent(target,
                      slice) {
        }

        TQueryRMSE(const TQueryRMSE& target)
            : TParent(target)
        {
        }

        template <class TLayout>
        TQueryRMSE(const TQueryRMSE<TLayout, TDataSet>& basedOn,
                   TTarget<TMapping>&& target)
            : TParent(basedOn,
                      std::move(target)) {
        }

        TQueryRMSE(TQueryRMSE&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            const double weight = GetTotalWeight();
            TVector<float> result;
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

            ApproximateForPermutation(point, /*indices*/ nullptr, &tmp, nullptr, nullptr);
            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            return TAdditiveStatistic(result[0], weight);
        }

        static double Score(const TAdditiveStatistic& score) {
            return sqrt(-score.Sum / score.Weight);
        }

        double Score(const TConstVec& point) {
            return Score(ComputeStats(point));
        }

        void GradientAt(const TConstVec& point,
                        TVec& weightedDer,
                        TVec& weights,
                        ui32 stream = 0) const {
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &weightedDer,
                                      nullptr,
                                      stream);
            weights.Copy(GetTarget().GetWeights(), stream);
        }

        void NewtonAt(const TConstVec& point,
                      TVec& weightedDer,
                      TVec& weightedDer2,
                      ui32 stream = 0) const {
            //TODO(noxoomo): add der2 adjust for qrmse-like targets
            return GradientAt(point, weightedDer, weightedDer2, stream);
        }

        void ApproximateForPermutation(const TConstVec& point,
                                       const TBuffer<ui32>* indices,
                                       TVec* value,
                                       TVec* der,
                                       TVec* der2,
                                       ui32 stream = 0) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();
            ApproximateQueryRmse(samplesGrouping.GetSizes(),
                                 samplesGrouping.GetBiasedOffsets(),
                                 samplesGrouping.GetOffsetsBias(),
                                 GetTarget().GetTargets(),
                                 GetTarget().GetWeights(),
                                 point,
                                 indices,
                                 value,
                                 der,
                                 der2,
                                 stream);
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        static constexpr TStringBuf TargetName() {
            return "QueryRMSE";
        }
    };

}
