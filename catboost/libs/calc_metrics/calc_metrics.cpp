#include "calc_metrics.h"

#include <catboost/libs/data/cb_dsv_loader.h>
#include <catboost/libs/data/proceed_pool_in_blocks.h>
#include <catboost/private/libs/target/data_providers.h>


namespace NCB {

    class TCalcMetricDataProvider {
    public:
        explicit TCalcMetricDataProvider(const TDataProviderPtr datasetPart)
            : DataProvider(datasetPart)
            {}

        TVector<TVector<double>> GetApproxes(NPar::TLocalExecutor* localExecutor) const;
        TVector<TSharedVector<float>> GetLabels(NPar::TLocalExecutor* localExecutor) const;
        const TWeights<float>& GetWeights() const;
        TMaybe<TSharedVector<TQueryInfo>> GetQueriesInfo() const;

        void ExtractApproxesToBackOfVector(TVector<TVector<double>>* approxesPtr, NPar::TLocalExecutor* localExecutor) const;

    private:
        const TDataProviderPtr DataProvider;
    };

    static TVector<TSharedVector<float>> ConvertTargetForCalcMetric(
        TMaybeData<TConstArrayRef<TRawTarget>> maybeRawTarget,
        bool isClass,
        bool isMultiClass,
        NPar::ILocalExecutor* localExecutor)
    {
        TVector<NJson::TJsonValue> outputClassLabels;
        return ConvertTarget(
            maybeRawTarget,
            ERawTargetType::Float,
            /* isRealTarget */ true,
            isClass,
            isMultiClass,
            /* targetBorder */ Nothing(),
            /* classCountUnknown */ true,
            /* inputClassLabels */ {},
            &outputClassLabels,
            localExecutor,
            /* classCount */ nullptr
        );
    }

    void TCalcMetricDataProvider::ExtractApproxesToBackOfVector(
        TVector<TVector<double>>* approxesPtr,
        NPar::TLocalExecutor* localExecutor
    ) const {
        const TRawObjectsDataProvider& rawObjectsData = dynamic_cast<TRawObjectsDataProvider& >(
            *(DataProvider->ObjectsData)
        );
        size_t approxDimension = DataProvider.Get()->ObjectsData->GetFeaturesLayout()->GetFloatFeatureCount();
        if (approxesPtr->empty()) {
            approxesPtr->resize(approxDimension);
        } else {
            Y_ASSERT(approxDimension == approxesPtr->size());
        }
        for (auto i : xrange(approxesPtr->size())) {
            auto maybeFactorData = rawObjectsData.GetFloatFeature(i);
            Y_ASSERT(maybeFactorData);
            auto factorData = maybeFactorData.GetRef()->ExtractValues(localExecutor);
            auto objectCount = DataProvider->GetObjectCount();
            (*approxesPtr)[i].insert(
                (*approxesPtr)[i].end(),
                factorData.begin(),
                factorData.begin() + objectCount
            );
        }
    }

    TVector<TVector<double>> TCalcMetricDataProvider::GetApproxes(
        NPar::TLocalExecutor* localExecutor
    ) const {
        TVector<TVector<double>> approx;
        ExtractApproxesToBackOfVector(&approx, localExecutor);
        return approx;
    }

    TVector<TSharedVector<float>> TCalcMetricDataProvider::GetLabels(
        NPar::TLocalExecutor* localExecutor
    ) const {
        const auto& targetData = DataProvider->RawTargetData;
        return ConvertTargetForCalcMetric(
            targetData.GetTarget(),
            /* isClass */ false,
            /* isMultiClass */ false,
            localExecutor
        );
    }

    const TWeights<float>& TCalcMetricDataProvider::GetWeights() const {
        return DataProvider->RawTargetData.GetWeights();
    }

    TMaybe<TSharedVector<TQueryInfo>> TCalcMetricDataProvider::GetQueriesInfo() const {
        const auto& targetData = DataProvider->RawTargetData;
        if (DataProvider->ObjectsData->GetGroupIds().Defined()) {
            return MakeGroupInfos(
                *targetData.GetObjectsGrouping(),
                DataProvider->ObjectsData->GetSubgroupIds(),
                targetData.GetWeights(),
                /* pairs */ Nothing()
            );
        }
        return Nothing();
    }

    TVector<THolder<IMetric>> ConstructMetric(
        const TVector<TColumn>& columnsDescription,
        const TVector<TString>& metricNamesList
    ) {
        size_t approxDimension = 0;
        for (auto& column : columnsDescription) {
            if (column.Type == EColumn::Num) {
                approxDimension += 1;
            }
        }
        return CreateMetricsFromDescription(metricNamesList, approxDimension);
    }

    TVector<double> GetMetricResultsFromMetricHolder(
        const TVector<TMetricHolder>& stats,
        const TVector<THolder<IMetric>>& metrics
    ) {
        TVector<double> metricResults;
        metricResults.reserve(metrics.size());
        for (auto metricIdx : xrange(metrics.size())) {
            metricResults.push_back(metrics[metricIdx]->GetFinalError(stats[metricIdx]));
        }
        return metricResults;
    }

    TVector<TMetricHolder> ConsumeCalcMetricsData(
        const TVector<const IMetric*>& metrics,
        const TDataProviderPtr dataProviderPtr,
        NPar::TLocalExecutor* localExecutor
    ) {
        for (const auto& metric : metrics) {
            CB_ENSURE_INTERNAL(metric->IsAdditiveMetric(), "ConsumeCalcMetricsData function support only additional metric");
        }

        TCalcMetricDataProvider dataProvider(dataProviderPtr);

        TVector<TVector<double>> approx = dataProvider.GetApproxes(localExecutor);
        TVector<TSharedVector<float>> labels = dataProvider.GetLabels(localExecutor);

        auto& weights = dataProvider.GetWeights();
        TMaybe<TSharedVector<TQueryInfo>> queriesInfo = dataProvider.GetQueriesInfo();
        TConstArrayRef<TQueryInfo> queriesInfoRef;
        if (queriesInfo.Defined()) {
            queriesInfoRef = *queriesInfo->Get();
        }

        TVector<TConstArrayRef<float>> labelRef(labels.size());
        for (auto targetIdx : xrange(labels.size())) {
            labelRef[targetIdx] = *labels[targetIdx].Get();
        }

        return EvalErrorsWithCaching(
            approx,
            /*approxDelts*/{},
            /*isExpApprox*/false,
            labelRef,
            weights.IsTrivial() ? TConstArrayRef<float>() : weights.GetNonTrivialData(),
            queriesInfoRef,
            metrics,
            localExecutor
        );
    }

    void TNonAdditiveMetricData::SaveProcessedData(
        const TDataProviderPtr dataProviderPtr,
        NPar::TLocalExecutor *localExecutor
    ) {
        TCalcMetricDataProvider dataProvider(dataProviderPtr);
        dataProvider.ExtractApproxesToBackOfVector(&Approxes, localExecutor);

        TVector<TSharedVector<float>> labels = dataProvider.GetLabels(localExecutor);
        auto& weights = dataProvider.GetWeights();
        if (!weights.IsTrivial()) {
            Weights.insert(
                Weights.end(),
                weights.GetNonTrivialData().begin(),
                weights.GetNonTrivialData().end()
            );
        }

        if (Target.empty()) {
            Target.resize(labels.size());
        } else {
            Y_ASSERT(Target.size() == labels.size());
        }
        for (auto targetIdx : xrange(labels.size())) {
            Target[targetIdx].insert(
                Target[targetIdx].end(),
                labels[targetIdx]->begin(),
                labels[targetIdx]->end());
        }
    }

    TVector<TMetricHolder> CalculateNonAdditiveMetrics(
        const TNonAdditiveMetricData& nonAdditiveMetricData,
        const TVector<const IMetric*>& nonAdditiveMetrics,
        NPar::TLocalExecutor *localExecutor
    ) {
        for (const auto& metric : nonAdditiveMetrics) {
            Y_ASSERT(!metric->IsAdditiveMetric());
        }
        if (!nonAdditiveMetrics.empty()) {
            return EvalErrorsWithCaching(
                nonAdditiveMetricData.Approxes,
                /*approxDelts*/{},
                /*isExpApprox*/false,
                To2DConstArrayRef<float>(nonAdditiveMetricData.Target),
                nonAdditiveMetricData.Weights,
                {},
                nonAdditiveMetrics,
                localExecutor
            );
        }
        return {};
    }

    TVector<double> CalcMetricsSingleHost(
        const NCatboostOptions::TDatasetReadingParams& datasetReadingParams,
        const TVector<THolder<IMetric>>& metrics,
        const TVector<TColumn>& columnsDescription,
        size_t threadCount
    ) {
        auto inputPath = datasetReadingParams.PoolPath;
        CB_ENSURE(inputPath.Scheme.Contains("dsv") || inputPath.Scheme == "", // "" is "dsv"
                  "Local metrics evaluation supports \"dsv\" and \"yt-dsv\" input file schemas.");

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(threadCount - 1);

        const int blockSize = 150000;
        TVector<TMetricHolder> stats;

        TNonAdditiveMetricData nonAdditiveMetricData;

        TVector<const IMetric*> additiveMetrics;
        TVector<const IMetric*> nonAdditiveMetrics;

        for (const auto& metric : metrics) {
            if (metric->IsAdditiveMetric()) {
                additiveMetrics.push_back(metric.Get());
            } else {
                nonAdditiveMetrics.push_back(metric.Get());
            }
        }

        ReadAndProceedPoolInBlocks(
            datasetReadingParams,
            blockSize,
            [&](const NCB::TDataProviderPtr datasetPart) {
                TSetLoggingVerbose inThisScope;
                auto subStats = ConsumeCalcMetricsData(
                    additiveMetrics,
                    datasetPart,
                    &executor);
                if (!nonAdditiveMetrics.empty()) {
                    nonAdditiveMetricData.SaveProcessedData(datasetPart, &executor);
                }
                if (stats.empty()) {
                    stats = std::move(subStats);
                } else {
                    Y_ASSERT(stats.size() == subStats.size());
                    for (auto ind : xrange(stats.size())) {
                        stats[ind].Add(subStats[ind]);
                    }
                }
            },
            &executor,
            MakeCdProviderFromArray(columnsDescription)
        );

        auto nonAdditiveStats = CalculateNonAdditiveMetrics(
            nonAdditiveMetricData,
            nonAdditiveMetrics,
            &executor
        );

        stats.insert(
            stats.end(),
            nonAdditiveStats.begin(),
            nonAdditiveStats.end()
        );

        auto metricResults = GetMetricResultsFromMetricHolder(stats, metrics);
        return metricResults;
    }

    THashMap<TString, double> TOpenSourceCalcMetricsImplementation::CalcMetrics(
        const TPathWithScheme& inputPath,
        const TVector<TString>& metricNamesList,
        const NCB::TDsvFormatOptions& dsvFormat,
        const THashMap<int, EColumn>& nonApproxColumnsDescription,
        ui32 threadCount
    ) const {
        NCatboostOptions::TDatasetReadingParams datasetReadingParams;
        datasetReadingParams.PoolPath = inputPath;
        datasetReadingParams.ColumnarPoolFormatParams.DsvFormat = dsvFormat;

        Y_ASSERT(inputPath.Scheme == "dsv" || inputPath.Scheme == "");
        ui32 columnCount = GetDsvColumnCount(inputPath, dsvFormat);

        TVector<TColumn> columnsDescription;
        columnsDescription.resize(columnCount, {EColumn::Num, TString()});
        for (auto [ind, column] : nonApproxColumnsDescription) {
            columnsDescription[ind] = TColumn{column, ""};
        }

        auto metrics = ConstructMetric(columnsDescription, metricNamesList);

        TVector<double> calculatedMetrics = CalcMetricsSingleHost(
            datasetReadingParams,
            metrics,
            columnsDescription,
            threadCount
        );
        THashMap<TString, double> metricResults;
        for (auto ind : xrange(metrics.size())) {
            metricResults.insert({metrics[ind]->GetDescription(), calculatedMetrics[ind]});
        }
        return metricResults;
    }

    namespace {
        TCalcMetricsImplementationFactory::TRegistrator<TOpenSourceCalcMetricsImplementation>
            OpenSourceCalcMetricsImplementationReg("dsv");
    }
}
