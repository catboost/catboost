#pragma once

#include "target_func.h"
#include "kernel.h"
#include "oracle_type.h"
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/gpu_data/doc_parallel_dataset.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

namespace NCatboostCuda {
    template <class TDocLayout>
    class TPointwiseTargetsImpl: public TPointwiseTarget<TDocLayout> {
    public:
        using TParent = TPointwiseTarget<TDocLayout>;
        using TStat = TAdditiveStatistic;
        using TMapping = TDocLayout;
        CB_DEFINE_CUDA_TARGET_BUFFERS();

        template <class TDataSet>
        TPointwiseTargetsImpl(const TDataSet& dataSet,
                              TGpuAwareRandom& random,
                              TSlice slice,
                              const NCatboostOptions::TLossDescription& targetOptions,
                              const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor = Nothing())
            : TParent(dataSet,
                      random,
                      slice)
            , ObjectiveDescriptor(objectiveDescriptor) {
            Init(targetOptions);
        }

        template <class TDataSet>
        TPointwiseTargetsImpl(const TDataSet& dataSet,
                              TGpuAwareRandom& random,
                              const NCatboostOptions::TLossDescription& targetOptions,
                              const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor = Nothing())
            : TParent(dataSet,
                      random)
            , ObjectiveDescriptor(objectiveDescriptor)
        {
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
            , ObjectiveDescriptor(target.GetObjectiveDescriptor())
        {
        }

        TPointwiseTargetsImpl(const TPointwiseTargetsImpl& target)
            : TParent(target)
            , Type(target.GetType())
            , Alpha(target.GetAlpha())
            , Border(target.GetBorder())
            , MetricName(target.ScoreMetricName())
            , ObjectiveDescriptor(target.GetObjectiveDescriptor())
        {
        }

        //        template <class TLayout>
        TPointwiseTargetsImpl(const TPointwiseTargetsImpl<NCudaLib::TMirrorMapping>& basedOn,
                              TTarget<TMapping>&& target)
            : TParent(basedOn,
                      std::move(target))
            , Type(basedOn.GetType())
            , Alpha(basedOn.GetAlpha())
            , Border(basedOn.GetBorder())
            , MetricName(basedOn.ScoreMetricName())
            , ObjectiveDescriptor(basedOn.GetObjectiveDescriptor())
        {
        }

        TPointwiseTargetsImpl(TPointwiseTargetsImpl&& other)
            : TParent(std::move(other))
            , Type(other.GetType())
            , Alpha(other.GetAlpha())
            , Border(other.GetBorder())
            , MetricName(other.ScoreMetricName())
            , ObjectiveDescriptor(other.GetObjectiveDescriptor())
        {
        }

        using TParent::GetTarget;
        using TParent::GetTotalWeight;

        TAdditiveStatistic ComputeStats(const TConstVec& point,
                                        const TMap<TString, TString>& params) const {
            auto tmp = TVec::Create(point.GetMapping().RepeatOnAllDevices(1));

            Approximate(GetTarget().GetTargets(),
                        GetTarget().GetWeights(),
                        point,
                        NCatboostOptions::GetAlpha(params),
                        UseBorder(),
                        GetBorder(),
                        &tmp,
                        /*der*/nullptr,
                        /*der2Row*/0,
                        /*der2*/nullptr);
            TVector<float> result;
            NCudaLib::TCudaBufferReader<TVec>(tmp)
                .SetFactorSlice(TSlice(0, 1))
                .SetReadSlice(TSlice(0, 1))
                .ReadReduce(result);

            const double weight = GetTotalWeight();
            const double multiplier = (Type == ELossFunction::MAE ? 2.0 : 1.0);

            return MakeSimpleAdditiveStatistic(-result[0] * multiplier, weight);
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
                        0,
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
                        0,
                        &weightedDer2,
                        stream);
        }

        void Approximate(const TConstVec& target,
                         const TConstVec& weights,
                         const TConstVec& point,
                         TVec* value,
                         TVec* der,
                         ui32 der2Row,
                         TVec* der2,
                         ui32 stream = 0) const {
            Approximate(
                target,
                weights,
                point,
                GetAlpha(),
                UseBorder(),
                GetBorder(),
                value,
                der,
                der2Row,
                der2,
                stream);
        }

        void StochasticDer(const TConstVec& point,
                           const TVec& sampledWeights,
                           TBuffer<ui32>&& sampledIndices,
                           bool secondDerAsWeights,
                           TOptimizationTarget* target) const {
            auto targetForIndices = TVec::CopyMapping(sampledIndices);
            Gather(targetForIndices, GetTarget().GetTargets(), sampledIndices);

            auto weightsForIndices = TVec::CopyMapping(sampledIndices);
            Gather(weightsForIndices, GetTarget().GetWeights(), sampledIndices);
            MultiplyVector(weightsForIndices, sampledWeights);

            auto pointForIndices = TVec::CopyMapping(sampledIndices);
            Gather(pointForIndices, point, sampledIndices);

            const ui32 statCount = point.GetColumnCount() + 1;
            target->StatsToAggregate.Reset(sampledWeights.GetMapping(), statCount);

            auto weightsView = target->StatsToAggregate.ColumnView(0);
            auto dersView = target->StatsToAggregate.ColumnsView(TSlice(1, statCount));

            if (secondDerAsWeights) {
                CB_ENSURE(point.GetColumnCount() == 1, "Unimplemented for multidim targets");
                Approximate(targetForIndices.ConstCopyView(),
                            weightsForIndices.ConstCopyView(),
                            pointForIndices.ConstCopyView(),
                            nullptr,
                            &dersView,
                            0,
                            &weightsView);
            } else {
                Approximate(targetForIndices.ConstCopyView(),
                            weightsForIndices.ConstCopyView(),
                            pointForIndices.ConstCopyView(),
                            nullptr,
                            &dersView,
                            0,
                            nullptr);

                weightsView.Copy(weightsForIndices);
            }

            target->Indices = std::move(sampledIndices);
        }

        static constexpr EOracleType OracleType() {
            return EOracleType::Pointwise;
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
        EHessianType GetHessianType() const {
            return EHessianType::Symmetric;
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

        ui32 GetDim() const {
            return 1;
        }

        TMaybe<TCustomObjectiveDescriptor> GetObjectiveDescriptor() const {
            return ObjectiveDescriptor;
        }

    private:
        void Init(const NCatboostOptions::TLossDescription& targetOptions) {
            Type = targetOptions.GetLossFunction();
            switch (targetOptions.GetLossFunction()) {
                case ELossFunction::PythonUserDefinedPerObject:
                case ELossFunction::Poisson:
                case ELossFunction::MAPE:
                case ELossFunction::RMSE:
                case ELossFunction::CrossEntropy: {
                    break;
                }
                case ELossFunction::Lq: {
                    Alpha = FromString<double>(targetOptions.GetLossParamsMap().at("q"));
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
                case ELossFunction::Expectile: {
                    Alpha = NCatboostOptions::GetAlpha(targetOptions);
                    break;
                }
                case ELossFunction::Logloss: {
                    Border = NCatboostOptions::GetLogLossBorder(targetOptions);
                    break;
                }
                case ELossFunction::Tweedie: {
                    VariancePower = NCatboostOptions::GetTweedieParam(targetOptions);
                    break;
                }
                case ELossFunction::Huber: {
                    Alpha = NCatboostOptions::GetHuberParam(targetOptions);
                    break;
                }
                default: {
                    ythrow TCatBoostException() << "Unsupported loss function " << targetOptions.GetLossFunction();
                }
            }
            MetricName = ToString(targetOptions);
        }

        bool UseBorder() const {
            return Type == ELossFunction::Logloss;
        }

        void Approximate(const TConstVec& target,
                         const TConstVec& weights,
                         const TConstVec& point,
                         double alpha,
                         bool useBorder,
                         double border,
                         TVec* value,
                         TVec* der,
                         ui32 der2Row,
                         TVec* der2,
                         ui32 stream = 0) const {
            CB_ENSURE(der2Row == 0);

            switch (Type) {
                case ELossFunction::PythonUserDefinedPerObject: {
                    ApproximateUserDefined(target,
                                           weights,
                                           point,
                                           *ObjectiveDescriptor,
                                           value,
                                           der,
                                           der2,
                                           stream);
                    break;
                }
                case ELossFunction::CrossEntropy:
                case ELossFunction::Logloss: {
                    ApproximateCrossEntropy(target,
                                            weights,
                                            point,
                                            value,
                                            der,
                                            der2,
                                            useBorder,
                                            border,
                                            stream);
                    break;
                }
                default: {
                    ApproximatePointwise(target,
                                         weights,
                                         point,
                                         Type,
                                         alpha,
                                         value,
                                         der,
                                         der2,
                                         stream);
                    break;
                }
            }
        }

    private:
        ELossFunction Type = ELossFunction::PythonUserDefinedPerObject;
        double Alpha = 0;
        double Border = 0;
        double VariancePower = 1.5;
        TString MetricName;
        TMaybe<TCustomObjectiveDescriptor> ObjectiveDescriptor = {};
    };

}
