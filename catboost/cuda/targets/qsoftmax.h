#pragma once

#include "quality_metric_helpers.h"
#include "target_func.h"
#include "kernel.h"
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>

namespace NCatboostCuda {
    template <class TDocLayout,
              class TDataSet>
    class TQuerySoftMax: public TQuerywiseTarget<TDocLayout, TDataSet> {
    public:
        using TParent = TQuerywiseTarget<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TQuerySoftMax(const TDataSet& dataSet,
                      TRandom& random,
                      TSlice slice,
                      const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::QuerySoftMax);
        }

        TQuerySoftMax(const TDataSet& dataSet,
                      TRandom& random,
                      const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::QuerySoftMax);
        }

        TQuerySoftMax(const TQuerySoftMax& target,
                      const TSlice& slice)
            : TParent(target,
                      slice) {
        }

        TQuerySoftMax(const TQuerySoftMax& target)
            : TParent(target)
        {
        }

        template <class TLayout>
        TQuerySoftMax(const TQuerySoftMax<TLayout, TDataSet>& basedOn,
                      TTarget<TMapping>&& target)
            : TParent(basedOn,
                      std::move(target)) {
        }

        TQuerySoftMax(TQuerySoftMax&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            const double weight = GetTotalWeightedTarget();
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
            return -score.Sum / score.Weight;
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
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &weightedDer,
                                      &weightedDer2,
                                      stream);
        }

        void ApproximateForPermutation(const TConstVec& point,
                                       const TBuffer<ui32>* indices,
                                       TVec* value,
                                       TVec* der,
                                       TVec* der2,
                                       ui32 stream = 0) const {
            const auto& samplesGrouping = TParent::GetSamplesGrouping();
            ApproximateQuerySoftMax(samplesGrouping.GetSizes(),
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
            return "QuerySoftMax";
        }

        inline double GetTotalWeightedTarget() const {
            if (TotalWeightedTarget <= 0) {
                TotalWeightedTarget = DotProduct(GetTarget().GetTargets(),
                                                 GetTarget().GetWeights());
                if (TotalWeightedTarget <= 0) {
                    ythrow TCatboostException() << "Observation targets and weights should be greater or equal zero. Total weighted target should be greater, than zero";
                }
            }
            return TotalWeightedTarget;
        }

    private:
        mutable double TotalWeightedTarget = 0;
    };

}
