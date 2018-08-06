#include "gpu_metrics.h"
#include "kernel.h"
#include "query_cross_entropy_kernels.h"
#include "multiclass_kernels.h"
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_util/dot_product.h>
#include <catboost/cuda/cuda_util/algorithm.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/all_reduce.h>
#include <catboost/cuda/gpu_data/kernels.h>
#include <catboost/cuda/gpu_data/querywise_helper.h>

using namespace NCudaLib;

namespace NCatboostCuda {
    IGpuMetric::IGpuMetric(const NCatboostOptions::TLossDescription& description)
        : CpuMetric(std::move(CreateMetricFromDescription(description, 1)[0]))
        , MetricDescription(description)
    {
    }

    static inline TMetricHolder MakeSimpleAdditiveStatistic(double sum, double weight) {
        TMetricHolder stat(2);
        stat.Stats[0] = sum;
        stat.Stats[1] = weight;
        return stat;
    }

    template <class T, class TMapping>
    static inline double SumVector(const TCudaBuffer<T, TMapping>& vec) {
        using TVec = TCudaBuffer<std::remove_const_t<T>, TMapping>;
        auto tmp = TVec::CopyMapping(vec);
        FillBuffer(tmp, 1.0f);
        return DotProduct(tmp, vec);
    }

    class TGpuPointwiseMetric: public IGpuPointwiseMetric {
    public:
        explicit TGpuPointwiseMetric(const NCatboostOptions::TLossDescription& config)
            : IGpuPointwiseMetric(config)
        {
        }

        virtual TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TStripeBuffer<const float>& cursor) const final {
            return EvalOnGpu<NCudaLib::TStripeMapping>(target, weights, cursor);
        }

        virtual TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TMirrorBuffer<const float>& cursor) const final {
            return EvalOnGpu<NCudaLib::TMirrorMapping>(target, weights, cursor);
        }

    private:
        template <class TMapping>
        TMetricHolder EvalOnGpu(const TCudaBuffer<const float, TMapping>& target,
                                const TCudaBuffer<const float, TMapping>& weights,
                                const TCudaBuffer<const float, TMapping>& cursor) const {
            using TVec = TCudaBuffer<float, TMapping>;

            double totalWeight = SumVector(weights);
            auto metricType = GetMetricDescription().GetLossFunction();
            const auto& params = GetMetricDescription().GetLossParams();
            switch (metricType) {
                case ELossFunction::Logloss:
                case ELossFunction::CrossEntropy: {
                    float border = GetDefaultClassificationBorder();
                    bool useBorder = false;
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    if (metricType == ELossFunction::Logloss) {
                        useBorder = true;
                        if (params.has("border")) {
                            border = FromString<float>(params.at("border"));
                        }
                    }

                    ApproximateCrossEntropy(target,
                                            weights,
                                            cursor,
                                            &tmp,
                                            (TVec*)nullptr,
                                            (TVec*)nullptr,
                                            useBorder,
                                            border);

                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::RMSE: {
                    auto tmp = TVec::CopyMapping(cursor);
                    tmp.Copy(cursor);
                    SubtractVector(tmp, target);
                    const double sum2 = DotProduct(tmp, tmp, &weights);
                    return MakeSimpleAdditiveStatistic(sum2, totalWeight);
                }
                case ELossFunction::Quantile:
                case ELossFunction::MAE:
                case ELossFunction::LogLinQuantile:
                case ELossFunction::MAPE:
                case ELossFunction::Poisson: {
                    float alpha = 0.5;
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    if (params.has("alpha")) {
                        alpha = FromString<float>(params.at("alpha"));
                    }

                    ApproximatePointwise(target,
                                         weights,
                                         cursor,
                                         metricType,
                                         alpha,
                                         &tmp,
                                         (TVec*)nullptr,
                                         (TVec*)nullptr);

                    auto result = ReadReduce(tmp);
                    const double multiplier = (metricType == ELossFunction::MAE ? 2.0 : 1.0);
                    return MakeSimpleAdditiveStatistic(-result[0] * multiplier, totalWeight);
                }
                case ELossFunction::MultiClass: {
                    auto tmp = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));
                    const ui32 classCount = cursor.GetColumnCount() + 1;
                    MultiLogitValueAndDer(target, weights, cursor, (const TCudaBuffer<ui32,TMapping>*)nullptr, classCount, &tmp, (TVec*)nullptr);
                    const double sum = ReadReduce(tmp)[0];
                    return MakeSimpleAdditiveStatistic(sum, totalWeight);
                }
                default: {
                    CB_ENSURE(false, "Unsupported on GPU pointwise metric " << metricType);
                }
            }
        }
    };

    class TGpuQuerywiseMetric: public IGpuQuerywiseMetric {
    public:
        explicit TGpuQuerywiseMetric(const NCatboostOptions::TLossDescription& config)
            : IGpuQuerywiseMetric(config)
        {
        }

        virtual TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TStripeMapping>& samplesGrouping,
                                   const TStripeBuffer<const float>& cursor) const {
            return EvalOnGpu<NCudaLib::TStripeMapping>(target, weights, samplesGrouping, cursor);
        }

        virtual TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& samplesGrouping,
                                   const TMirrorBuffer<const float>& cursor) const {
            return EvalOnGpu<NCudaLib::TMirrorMapping>(target,
                                                       weights,
                                                       samplesGrouping,
                                                       cursor);
        }

    private:
        template <class TMapping>
        TMetricHolder EvalOnGpu(const TCudaBuffer<const float, TMapping>& target,
                                const TCudaBuffer<const float, TMapping>& weights,
                                const TGpuSamplesGrouping<TMapping>& samplesGrouping,
                                const TCudaBuffer<const float, TMapping>& cursor) const {
            using TVec = TCudaBuffer<float, TMapping>;
            auto value = TVec::Create(cursor.GetMapping().RepeatOnAllDevices(1));

            auto metricType = GetMetricDescription().GetLossFunction();
            switch (metricType) {
                case ELossFunction::QueryRMSE: {
                    double totalWeight = SumVector(weights);
                    ApproximateQueryRmse(samplesGrouping.GetSizes(),
                                         samplesGrouping.GetBiasedOffsets(),
                                         samplesGrouping.GetOffsetsBias(),
                                         target,
                                         weights,
                                         cursor,
                                         (TCudaBuffer<ui32, TMapping>*)nullptr,
                                         &value,
                                         (TVec*)nullptr,
                                         (TVec*)nullptr);
                    double sum = ReadReduce(value)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeight);
                }
                case ELossFunction::QuerySoftMax: {
                    double totalWeightedTarget = DotProduct(target,
                                                            weights);
                    ApproximateQuerySoftMax(samplesGrouping.GetSizes(),
                                            samplesGrouping.GetBiasedOffsets(),
                                            samplesGrouping.GetOffsetsBias(),
                                            NCatboostOptions::GetQuerySoftMaxLambdaReg(GetMetricDescription()),
                                            target,
                                            weights,
                                            cursor,
                                            (TCudaBuffer<ui32, TMapping>*)nullptr,
                                            &value,
                                            (TVec*)nullptr,
                                            (TVec*)nullptr);
                    double sum = ReadReduce(value)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalWeightedTarget);
                }
                case ELossFunction::PairLogitPairwise:
                case ELossFunction::PairLogit: {
                    double totalPairsWeight = SumVector(samplesGrouping.GetPairsWeights());
                    ApproximatePairLogit(samplesGrouping.GetPairs(),
                                         samplesGrouping.GetPairsWeights(),
                                         samplesGrouping.GetOffsetsBias(),
                                         cursor,
                                         (TCudaBuffer<ui32, TMapping>*)nullptr,
                                         &value,
                                         (TVec*)nullptr,
                                         (TVec*)nullptr);
                    double sum = ReadReduce(value)[0];
                    return MakeSimpleAdditiveStatistic(-sum, totalPairsWeight);
                }
                default: {
                    CB_ENSURE(false, "Unsupported on GPU pointwise metric " << metricType);
                }
            }
        }

    private:
    };

    TMetricHolder TCpuFallbackMetric::Eval(const TVector<TVector<double>>& approx,
                                           const TVector<float>& target,
                                           const TVector<float>& weight,
                                           const TVector<TQueryInfo>& queriesInfo) const {
        const IMetric& metric = GetCpuMetric();
        const int start = 0;
        const int end = static_cast<const int>(metric.GetErrorType() == EErrorType::PerObjectError ? target.size() : queriesInfo.size());
        CB_ENSURE(approx.size() >= 1);
        for (ui32 dim = 0; dim < approx.size(); ++dim) {
            CB_ENSURE(approx[dim].size() == target.size());
        }
        return metric.Eval(approx,
                           target,
                           weight,
                           queriesInfo,
                           start,
                           end,
                           NPar::LocalExecutor());
    }

    static THolder<IGpuMetric> CreateGpuMetricFromDescription(ELossFunction targetObjective,
                                                              const NCatboostOptions::TLossDescription& metricDescription) {
        auto metricType = metricDescription.GetLossFunction();
        switch (metricType) {
            case ELossFunction::Logloss:
            case ELossFunction::CrossEntropy:
            case ELossFunction::RMSE:
            case ELossFunction::Quantile:
            case ELossFunction::MAE:
            case ELossFunction::LogLinQuantile:
            case ELossFunction::MultiClass:
            case ELossFunction::MAPE:
            case ELossFunction::Poisson: {
                return new TGpuPointwiseMetric(metricDescription);
            }
            case ELossFunction::QueryRMSE:
            case ELossFunction::QuerySoftMax:
            case ELossFunction::PairLogit:
            case ELossFunction::PairLogitPairwise: {
                return new TGpuQuerywiseMetric(metricDescription);
            }
            case ELossFunction::QueryCrossEntropy: {
                CB_ENSURE(targetObjective == ELossFunction::QueryCrossEntropy, "Error: could compute QueryCrossEntropy metric on GPU only for QueryCrossEntropyObjective");
                return new TTargetFallbackMetric(metricDescription);
            }
            default: {
                THolder<IGpuMetric> metric = new TCpuFallbackMetric(metricDescription);
                MATRIXNET_WARNING_LOG << "Metric " << metric->GetCpuMetric().GetDescription() << " is not implemented on GPU. Will use CPU for metric computation, this could significantly affect learning time" << Endl;
                return metric;
            }
        }
    }

    TVector<THolder<IGpuMetric>> CreateGpuMetrics(const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
                                                  const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions) {
        TVector<THolder<IGpuMetric>> metrics;

        if (evalMetricOptions->EvalMetric.IsSet()) {
            if (evalMetricOptions->EvalMetric->GetLossFunction() == ELossFunction::Custom) {
                CB_ENSURE(false, "Error: GPU doesn't support custom metrics");
            } else {
                metrics.push_back(CreateGpuMetricFromDescription(lossFunctionOption->GetLossFunction(),
                                                                 evalMetricOptions->EvalMetric));
            }
        }

        CB_ENSURE(lossFunctionOption->GetLossFunction() != ELossFunction::Custom, "Error: GPU doesn't support custom loss");

        metrics.push_back(CreateGpuMetricFromDescription(lossFunctionOption->GetLossFunction(),
                                                         lossFunctionOption));

        for (const auto& description : evalMetricOptions->CustomMetrics.Get()) {
            metrics.push_back(CreateGpuMetricFromDescription(lossFunctionOption->GetLossFunction(),
                                                             description));
        }
        return metrics;
    }

}
