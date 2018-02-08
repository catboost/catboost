#pragma once

#include "quality_metric_helpers.h"
#include "target_base.h"
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

        TQuerySoftMax(const TQuerySoftMax& target,
                      const TSlice& slice)
            : TParent(target,
                      slice) {
        }

        template <class TLayout>
        TQuerySoftMax(const TQuerySoftMax<TLayout, TDataSet>& basedOn,
                      TCudaBuffer<const float, TMapping>&& target,
                      TCudaBuffer<const float, TMapping>&& weights,
                      TCudaBuffer<const ui32, TMapping>&& indices)
            : TParent(basedOn,
                      std::move(target),
                      std::move(weights),
                      std::move(indices)) {
        }

        TQuerySoftMax(TQuerySoftMax&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;
        using TParent::GetWeights;

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
                        TVec& dst,
                        TVec& weights,
                        ui32 stream = 0) const {
            ApproximateForPermutation(point,
                                      nullptr,
                                      nullptr,
                                      &dst,
                                      nullptr,
                                      stream);
            weights.Copy(GetWeights(), stream);
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
                                    GetTarget(),
                                    GetWeights(),
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
                TotalWeightedTarget = DotProduct(TTargetBase<TMapping, TDataSet>::Target,
                                                 TTargetBase<TMapping, TDataSet>::Weights);
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
