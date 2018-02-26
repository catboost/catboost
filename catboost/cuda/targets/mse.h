#pragma once

#include "target_func.h"
#include "kernel.h"
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/algorithm.h>

namespace NCatboostCuda {
    template <class TDocLayout,
              class TDataSet>
    class TL2: public TPointwiseTarget<TDocLayout, TDataSet> {
    public:
        using TParent = TPointwiseTarget<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TL2(const TDataSet& dataSet,
            TRandom& random,
            TSlice slice,
            const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::RMSE);
        }

        TL2(const TDataSet& dataSet,
            TRandom& random,
            const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            CB_ENSURE(targetOptions.GetLossFunction() == ELossFunction::RMSE);
        }

        TL2(const TL2& target,
            const TSlice& slice)
            : TParent(target,
                      slice) {
        }

        TL2(const TL2& target)
            : TParent(target)
        {
        }

        template <class TLayout>
        TL2(const TL2<TLayout, TDataSet>& basedOn,
            TTarget<TMapping>&& target)
            : TParent(basedOn.GetDataSet(),
                      basedOn.GetRandom(),
                      std::move(target)) {
        }

        TL2(TL2&& other)
            : TParent(std::move(other))
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            TVec tmp = TVec::CopyMapping(point);
            tmp.Copy(point);
            SubtractVector(tmp, GetTarget().GetTargets());
            auto& weights = GetTarget().GetWeights();
            const double sum2 = DotProduct(tmp, tmp, &weights);
            const double weight = GetTotalWeight();

            return TAdditiveStatistic(sum2, weight);
        }

        static double Score(const TAdditiveStatistic& score) {
            return sqrt(score.Sum / score.Weight);
        }

        double Score(const TConstVec& point) {
            return Score(ComputeStats(point));
        }

        void GradientAt(const TConstVec& point,
                        TVec& weightedDer,
                        TVec& weights,
                        ui32 stream = 0) const {
            const auto& weight = GetTarget().GetWeights();
            Approximate(GetTarget().GetTargets(),
                        weight,
                        point,
                        nullptr,
                        &weightedDer,
                        nullptr,
                        stream);
            weights.Copy(weight, stream);
        }

        void NewtonAt(const TConstVec& point,
                      TVec& dst,
                      TVec& weightederDer2,
                      ui32 stream = 0) const {
            return GradientAt(point, dst, weightederDer2, stream);
        }

        void Approximate(const TConstVec& target,
                         const TConstVec& weights,
                         const TConstVec& point,
                         TVec* value,
                         TVec* der,
                         TVec* der2,
                         ui32 stream = 0) const {
            ApproximateMse(target, weights, point, value, der, der2, stream);
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        static constexpr TStringBuf TargetName() {
            return "RMSE";
        }
    };
}
