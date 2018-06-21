#pragma once

#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/options/enums.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/options/metric_options.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/cuda/gpu_data/samples_grouping_gpu.h>

namespace NCatboostCuda {
    class IGpuMetric {
    public:
        virtual ~IGpuMetric() {
        }

        explicit IGpuMetric(const NCatboostOptions::TLossDescription& config);

        const IMetric& GetCpuMetric() const {
            return *CpuMetric;
        }

        EErrorType GetErrorType() const {
            return GetCpuMetric().GetErrorType();
        }

        const NCatboostOptions::TLossDescription& GetMetricDescription() const {
            return MetricDescription;
        }

    private:
        THolder<IMetric> CpuMetric;
        NCatboostOptions::TLossDescription MetricDescription;
    };

    class IGpuPointwiseMetric: public IGpuMetric {
    public:
        explicit IGpuPointwiseMetric(const NCatboostOptions::TLossDescription& config)
            : IGpuMetric(config)
        {
        }

        virtual TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TStripeBuffer<const float>& cursor) const = 0;

        virtual TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TMirrorBuffer<const float>& cursor) const = 0;
    };

    class IGpuQuerywiseMetric: public IGpuMetric {
    public:
        explicit IGpuQuerywiseMetric(const NCatboostOptions::TLossDescription& config)
            : IGpuMetric(config)
        {
        }

        virtual TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TStripeMapping>& samplesGrouping,
                                   const TStripeBuffer<const float>& cursor) const = 0;

        virtual TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TGpuSamplesGrouping<NCudaLib::TMirrorMapping>& samplesGrouping,
                                   const TMirrorBuffer<const float>& cursor) const = 0;
    };

    class TTargetFallbackMetric: public IGpuMetric {
    public:
        explicit TTargetFallbackMetric(const NCatboostOptions::TLossDescription& config)
            : IGpuMetric(config)
        {
        }

        template <class TTarget, class TMapping>
        TMetricHolder Eval(const TTarget& target,
                           const TCudaBuffer<const float, TMapping>& point) const {
            auto metric = GetMetricDescription().GetLossFunction();
            CB_ENSURE(target.GetScoreMetricType() == metric, "Error: can't compute metric " << metric << " on GPU");
            TMetricHolder stats = target.ComputeStats(point,
                                                      GetMetricDescription().GetLossParams());
            stats.Stats[0] *= -1;
            return stats;
        };
    };

    class TCpuFallbackMetric: public IGpuMetric {
    public:
        explicit TCpuFallbackMetric(const NCatboostOptions::TLossDescription& config)
            : IGpuMetric(config)
        {
        }

        TMetricHolder Eval(const TVector<TVector<double>>& approx,
                           const TVector<float>& target,
                           const TVector<float>& weight,
                           const TVector<TQueryInfo>& queriesInfo) const;
    };

    TVector<THolder<IGpuMetric>> CreateGpuMetrics(const NCatboostOptions::TOption<NCatboostOptions::TLossDescription>& lossFunctionOption,
                                                  const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions);
}
