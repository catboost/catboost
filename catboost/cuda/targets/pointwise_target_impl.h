#pragma once

#include "target_func.h"
#include "kernel.h"
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/algorithm.h>

namespace NCatboostCuda {
    template <class TDocLayout, class TDataSet>
    class TPointwiseTargetsImpl: public TPointwiseTarget<TDocLayout, TDataSet> {
    public:
        using TParent = TPointwiseTarget<TDocLayout, TDataSet>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        TPointwiseTargetsImpl(const TDataSet& dataSet,
                              TRandom& random,
                              TSlice slice,
                              const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            Init(targetOptions);
        }

        TPointwiseTargetsImpl(const TDataSet& dataSet,
                              TRandom& random,
                              const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random) {
            Init(targetOptions);
        }

        TPointwiseTargetsImpl(const TPointwiseTargetsImpl& target,
                              const TSlice& slice)
            : TParent(target,
                      slice)
            , Type(target.GetType())
            , Alpha(target.GetAlpha())
            , MetricName(target.TargetName())
        {
        }

        TPointwiseTargetsImpl(const TPointwiseTargetsImpl& target)
            : TParent(target)
            , Type(target.GetType())
            , Alpha(target.GetAlpha())
            , MetricName(target.TargetName())
        {
        }

        template <class TLayout>
        TPointwiseTargetsImpl(const TPointwiseTargetsImpl<TLayout, TDataSet>& basedOn,
                              TTarget<TMapping>&& target)
            : TParent(basedOn.GetDataSet(),
                      basedOn.GetRandom(),
                      std::move(target))
            , Type(basedOn.GetType())
            , Alpha(basedOn.GetAlpha())
            , MetricName(basedOn.TargetName())
        {
        }

        TPointwiseTargetsImpl(TPointwiseTargetsImpl&& other)
            : TParent(std::move(other))
            , Type(other.GetType())
            , Alpha(other.GetAlpha())
            , MetricName(other.TargetName())
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            TVector<float> result;
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

            Approximate(GetTarget().GetTargets(),
                        GetTarget().GetWeights(),
                        point,
                        &tmp,
                        nullptr,
                        nullptr);

            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            const double weight = GetTotalWeight();
            const double multiplier = (Type == ELossFunction::MAE ? 2.0 : 1.0);

            return TAdditiveStatistic(result[0] * multiplier, weight);
        }

        static double Score(const TAdditiveStatistic& score) {
            return -score.Sum / score.Weight;
        }

        double Score(const TConstVec& point) {
            return Score(ComputeStats(point));
        }

        void GradientAt(const TConstVec& point,
                        TVec& weightedDer,
                        TVec& weightedDer2,
                        ui32 stream = 0) const {
            Approximate(GetTarget().GetTargets(),
                        GetTarget().GetWeights(),
                        point,
                        nullptr,
                        &weightedDer,
                        nullptr,
                        stream);
            weightedDer2.Copy(GetTarget().GetWeights());
        }

        void NewtonAt(const TConstVec& point,
                      TVec& weightedDer,
                      TVec& weightedDer2,
                      ui32 stream = 0) const {
            Approximate(GetTarget().GetTargets(),
                        GetTarget().GetWeights(),
                        point,
                        nullptr,
                        &weightedDer,
                        &weightedDer2,
                        stream);
        }

        void Approximate(const TConstVec& target,
                         const TConstVec& weights,
                         const TConstVec& point,
                         TVec* value,
                         TVec* der,
                         TVec* der2,
                         ui32 stream = 0) const {
            ApproximatePointwise(target, weights, point, Type, Alpha, value, der, der2, stream);
        }

        TStringBuf TargetName() const {
            return MetricName;
        }

        static constexpr bool IsMinOptimal() {
            return true;
        }

        ELossFunction GetType() const {
            return Type;
        }

        double GetAlpha() const {
            return Alpha;
        }

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            Type = targetOptions.GetLossFunction();
            switch (targetOptions.GetLossFunction()) {
                case ELossFunction::Poisson:
                case ELossFunction::MAPE: {
                    break;
                }
                case ELossFunction::MAE: {
                    Alpha = 0.5;
                    break;
                }
                case ELossFunction::Quantile:
                case ELossFunction::LogLinQuantile: {
                    Alpha = NCatboostOptions::GetAlpha(targetOptions);
                    break;
                }
                default: {
                    ythrow TCatboostException() << "Unsupported loss function " << targetOptions.GetLossFunction();
                }
            }
            MetricName = ToString(targetOptions);
        }

    private:
        ELossFunction Type = ELossFunction::Custom;
        double Alpha = 0;
        TString MetricName;
    };

}
