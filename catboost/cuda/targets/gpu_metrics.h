#pragma once

#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/loss_description.h>
#include <catboost/private/libs/options/metric_options.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/cuda/gpu_data/samples_grouping_gpu.h>
#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

#include <library/cpp/threading/local_executor/local_executor.h>

namespace NCatboostCuda {
    class IGpuMetric {
    public:
        virtual ~IGpuMetric() {
        }

        explicit IGpuMetric(const NCatboostOptions::TLossDescription& config, ui32 cpuApproxDim);

        explicit IGpuMetric(THolder<IMetric>&& cpuMetric, const NCatboostOptions::TLossDescription& config);

        const IMetric& GetCpuMetric() const {
            return *CpuMetric;
        }

        EErrorType GetErrorType() const {
            return GetCpuMetric().GetErrorType();
        }

        TMetricParam<bool>& GetUseWeights() const {
            return CpuMetric->UseWeights;
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
        explicit IGpuPointwiseMetric(const NCatboostOptions::TLossDescription& config, ui32 approxDim)
            : IGpuMetric(config, approxDim)
        {
        }

        explicit IGpuPointwiseMetric(THolder<IMetric>&& cpuMetric, const NCatboostOptions::TLossDescription& config)
            : IGpuMetric(std::move(cpuMetric), config)
        {
        }

        virtual TMetricHolder Eval(const TStripeBuffer<const float>& target,
                                   const TStripeBuffer<const float>& weights,
                                   const TStripeBuffer<const float>& cursor,
                                   TScopedCacheHolder* cache) const = 0;

        virtual TMetricHolder Eval(const TMirrorBuffer<const float>& target,
                                   const TMirrorBuffer<const float>& weights,
                                   const TMirrorBuffer<const float>& cursor,
                                   TScopedCacheHolder* cache) const = 0;
    };

    class IGpuQuerywiseMetric: public IGpuMetric {
    public:
        explicit IGpuQuerywiseMetric(const NCatboostOptions::TLossDescription& config, ui32 approxDim)
            : IGpuMetric(config, approxDim)
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
        explicit TTargetFallbackMetric(const NCatboostOptions::TLossDescription& config, ui32 approxDim)
            : IGpuMetric(config, approxDim)
        {
        }

        template <class TTarget, class TMapping>
        TMetricHolder Eval(const TTarget& target,
                           const TCudaBuffer<const float, TMapping>& point) const {
            auto metric = GetMetricDescription().GetLossFunction();
            CB_ENSURE(target.GetScoreMetricType() == metric, "Error: can't compute metric " << metric << " on GPU");
            TMetricHolder stats = target.ComputeStats(point,
                                                      GetMetricDescription().GetLossParamsMap());
            return stats;
        };
    };

    class TCpuFallbackMetric: public IGpuMetric {
    public:
        explicit TCpuFallbackMetric(const NCatboostOptions::TLossDescription& config, ui32 approxDim)
            : IGpuMetric(config, approxDim)
        {
        }

        explicit TCpuFallbackMetric(THolder<IMetric>&& metric, const NCatboostOptions::TLossDescription& config)
            : IGpuMetric(std::move(metric), config)
        {
        }

        TMetricHolder Eval(const TVector<TVector<double>>& approx,
                           const TVector<float>& target,
                           const TVector<float>& weight,
                           const TVector<TQueryInfo>& queriesInfo,
                           NPar::ILocalExecutor* localExecutor) const;
    };

    class TGpuCustomMetric : public IGpuMetric {
        TCustomMetricDescriptor Descriptor;
    public:
        explicit TGpuCustomMetric(
            const TCustomMetricDescriptor& metricDescriptor,
            const NCatboostOptions::TLossDescription& config
        )
            : IGpuMetric(MakeCustomMetric(metricDescriptor), config)
            , Descriptor(metricDescriptor)
        {
        }

        TMetricHolder Eval(
            const TStripeBuffer<const float>& target,
            const TStripeBuffer<const float>& weights,
            const TStripeBuffer<const float>& cursor,
            TScopedCacheHolder* cache,
            ui32 stream = 0
        ) const;

        TMetricHolder Eval(
            const TMirrorBuffer<const float>& target,
            const TMirrorBuffer<const float>& weights,
            const TMirrorBuffer<const float>& cursor,
            TScopedCacheHolder* cache,
            ui32 stream = 0
        ) const;


        template<class TMapping>
        TMetricHolder EvalImpl(
            const TCudaBuffer<const float, TMapping>& target,
            const TCudaBuffer<const float, TMapping>& weights,
            const TCudaBuffer<const float, TMapping>& cursor,
            TScopedCacheHolder* cache,
            ui32 stream
        ) const;

        double GetFinalError(TMetricHolder&& metricHolder) const;
    };

    TVector<THolder<IGpuMetric>> CreateGpuMetrics(
        const NCatboostOptions::TOption<NCatboostOptions::TMetricOptions>& evalMetricOptions,
        const ui32 cpuApproxDim,
        bool hasWeights,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor
    );
}
