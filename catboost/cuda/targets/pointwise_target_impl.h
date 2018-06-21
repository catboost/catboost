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
                              TGpuAwareRandom& random,
                              TSlice slice,
                              const NCatboostOptions::TLossDescription& targetOptions)
            : TParent(dataSet,
                      random,
                      slice) {
            Init(targetOptions);
        }

        TPointwiseTargetsImpl(const TDataSet& dataSet,
                              TGpuAwareRandom& random,
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
            , Border(target.GetBorder())
            , MetricName(target.ScoreMetricName())
        {
        }

        TPointwiseTargetsImpl(const TPointwiseTargetsImpl& target)
            : TParent(target)
            , Type(target.GetType())
            , Alpha(target.GetAlpha())
            , Border(target.GetBorder())
            , MetricName(target.ScoreMetricName())
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
            , Border(basedOn.GetBorder())
            , MetricName(basedOn.ScoreMetricName())
        {
        }

        TPointwiseTargetsImpl(TPointwiseTargetsImpl&& other)
            : TParent(std::move(other))
            , Type(other.GetType())
            , Alpha(other.GetAlpha())
            , Border(other.GetBorder())
            , MetricName(other.ScoreMetricName())
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point, double alpha) const {
            TVector<float> result;
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

            ApproximatePointwise(GetTarget().GetTargets(),
                                 GetTarget().GetWeights(),
                                 point,
                                 Type,
                                 alpha,
                                 &tmp,
                                 (TVec*)nullptr,
                                 (TVec*)nullptr);

            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            const double weight = GetTotalWeight();
            const double multiplier = (Type == ELossFunction::MAE ? 2.0 : 1.0);

            return MakeSimpleAdditiveStatistic(result[0] * multiplier, weight);
        }

        TAdditiveStatistic ComputeStats(const TConstVec& point) const {
            return ComputeStats(point, Alpha);
        }

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString>& params) const {
            return ComputeStats(point, NCatboostOptions::GetAlpha(params));
        }

        double Score(const TAdditiveStatistic& score) const {
            switch (Type) {
                case ELossFunction::RMSE: {
                    return sqrt(score.Stats[0] / score.Stats[1]);
                }
                default: {
                    return -score.Stats[0] / score.Stats[1];
                }

            }
        }

        double Score(const TConstVec& point) const {
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
            switch (Type) {
                case ELossFunction::CrossEntropy:
                case ELossFunction::Logloss: {
                    ApproximateCrossEntropy(target,
                                            weights,
                                            point,
                                            value,
                                            der,
                                            der2,
                                            UseBorder(),
                                            GetBorder(),
                                            stream);
                    break;
                }
                default: {
                    ApproximatePointwise(target,
                                         weights,
                                         point,
                                         Type,
                                         Alpha,
                                         value,
                                         der,
                                         der2,
                                         stream);
                    break;
                }
            }
        }

        TStringBuf ScoreMetricName() const {
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

        double GetBorder() const {
            return Border;
        }

        ELossFunction GetScoreMetricType() const {
            return Type;
        }

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            Type = targetOptions.GetLossFunction();
            switch (targetOptions.GetLossFunction()) {
                case ELossFunction::Poisson:
                case ELossFunction::MAPE:
                case ELossFunction::RMSE:
                case ELossFunction::CrossEntropy: {
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
                case ELossFunction::Logloss: {
                    Border = NCatboostOptions::GetLogLossBorder(targetOptions);
                    break;
                }
                default: {
                    ythrow TCatboostException() << "Unsupported loss function " << targetOptions.GetLossFunction();
                }
            }
            MetricName = ToString(targetOptions);
        }

        bool UseBorder() const {
            return Type == ELossFunction::Logloss;
        }

    private:
        ELossFunction Type = ELossFunction::Custom;
        double Alpha = 0;
        double Border = 0;
        TString MetricName;
    };

}
