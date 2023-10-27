#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/metrics/metric_holder.h>
#include <catboost/private/libs/options/dataset_reading_params.h>

#include <library/cpp/object_factory/object_factory.h>


namespace NCB {
    class ICalcMetricsImplementation {
    public:
        virtual ~ICalcMetricsImplementation() = default;
        virtual THashMap<TString, double> CalcMetrics(
            const TPathWithScheme& inputPath,
            const TVector<TString>& metricNamesList,
            const NCB::TDsvFormatOptions& dsvFormat,
            const THashMap<int, EColumn>& nonAuxiliaryColumnsDescription,
            ui32 threadCount
        ) const = 0;
    };

    using TCalcMetricsImplementationFactory
        = NObjectFactory::TParametrizedObjectFactory<ICalcMetricsImplementation, TString>;


    TVector<THolder<IMetric>> ConstructMetric(
        const TVector<TColumn>& columnsDescription,
        const TVector<TString>& metricNamesList
    );

    struct TNonAdditiveMetricData {
        TVector<TVector<double>> Approxes;
        TVector<TVector<float>> Target;
        TVector<float> Weights;

    public:
        void SaveProcessedData(const TDataProviderPtr datasetPart, NPar::TLocalExecutor* localExecutor);
    };

    TVector<TMetricHolder> ConsumeCalcMetricsData(
        const TVector<const IMetric*>& metrics,
        const TDataProviderPtr datasetPart,
        NPar::TLocalExecutor *localExecutor
    );

    TVector<TMetricHolder> CalculateNonAdditiveMetrics(
        const TNonAdditiveMetricData& nonAdditiveMetricData,
        const TVector<const IMetric*>& nonAdditiveMetrics,
        NPar::TLocalExecutor *localExecutor
    );

    TVector<double> GetMetricResultsFromMetricHolder(
        const TVector<TMetricHolder>& stats,
        const TVector<THolder<IMetric>>& metrics
    );

    TVector<double> CalcMetricsSingleHost(
        const NCatboostOptions::TDatasetReadingParams& datasetReadingParams,
        const TVector<THolder<IMetric>>& metrics,
        const TVector<TColumn>& columnsDescription,
        size_t threadCount
    );

    class TOpenSourceCalcMetricsImplementation : public ICalcMetricsImplementation {
    public:
        THashMap<TString, double> CalcMetrics(
            const TPathWithScheme& inputPath,
            const TVector<TString>& metricNamesList,
            const NCB::TDsvFormatOptions& dsvFormat,
            const THashMap<int, EColumn>& nonAuxiliaryColumnsDescription,
            ui32 threadCount
        ) const override;
    };

    void CheckColumnIndices(int columnCount, const THashMap<int, EColumn>& nonAuxiliaryColumnsDescription);
}
