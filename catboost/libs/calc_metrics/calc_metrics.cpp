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
        TSharedWeights<float> GetWeights(NPar::TLocalExecutor* localExecutor) const;
        TMaybe<TSharedVector<TQueryInfo>> GetQueriesInfo() const;

        void ExtractApproxesToBackOfVector(
            TVector<TVector<double>>* approxesPtr,
            NPar::TLocalExecutor* localExecutor
        ) const;

    private:
        const TDataProviderPtr DataProvider;
    };

    static TVector<TSharedVector<float>> ConvertTargetForCalcMetric(
        TMaybeData<TConstArrayRef<TRawTarget>> maybeRawTarget,
        bool isClass,
        bool isMultiClass,
        bool isMultiLabel,
        NPar::ILocalExecutor* localExecutor
    ) {
        TVector<NJson::TJsonValue> outputClassLabels;
        return ConvertTarget(
            maybeRawTarget,
            ERawTargetType::Float,
            /* isRealTarget */ true,
            isClass,
            isMultiClass,
            isMultiLabel,
            /* targetBorder */ Nothing(),
            /* classCountUnknown */ true,
            /* inputClassLabels */ {},
            /* allowConstLabel */ true,
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

    TVector<TVector<double>> TCalcMetricDataProvider::GetApproxes(NPar::TLocalExecutor* localExecutor) const {
        TVector<TVector<double>> approx;
        ExtractApproxesToBackOfVector(&approx, localExecutor);
        return approx;
    }

    TVector<TSharedVector<float>> TCalcMetricDataProvider::GetLabels(NPar::TLocalExecutor* localExecutor) const {
        const auto& targetData = DataProvider->RawTargetData;
        return ConvertTargetForCalcMetric(
            targetData.GetTarget(),
            /* isClass */ false,
            /* isMultiClass */ false,
            /* isMultiLabel */ false,
            localExecutor
        );
    }

    TSharedWeights<float> TCalcMetricDataProvider::GetWeights(NPar::TLocalExecutor* localExecutor) const {
        const auto weights = NCB::MakeWeights(
            DataProvider->RawTargetData.GetWeights(),
            DataProvider->RawTargetData.GetGroupWeights(),
            /*isForGpu*/ false,
            localExecutor
        );
        return weights;
    }

    TMaybe<TSharedVector<TQueryInfo>> TCalcMetricDataProvider::GetQueriesInfo() const {
        const auto& targetData = DataProvider->RawTargetData;
        if (DataProvider->ObjectsData->GetGroupIds().Defined()) {
            return MakeGroupInfos(
                *targetData.GetObjectsGrouping(),
                DataProvider->ObjectsData->GetSubgroupIds(),
                targetData.GetGroupWeights(),
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
            CB_ENSURE_INTERNAL(metric->IsAdditiveMetric(), "ConsumeCalcMetricsData function supports only additive metrics");
        }

        TCalcMetricDataProvider dataProvider(dataProviderPtr);

        TVector<TVector<double>> approx = dataProvider.GetApproxes(localExecutor);
        TVector<TSharedVector<float>> labels = dataProvider.GetLabels(localExecutor);

        auto weightsPtr = dataProvider.GetWeights(localExecutor);
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
            weightsPtr->IsTrivial() ? TConstArrayRef<float>() : weightsPtr->GetNonTrivialData(),
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
        auto weightsPtr = dataProvider.GetWeights(localExecutor);
        if (!weightsPtr->IsTrivial()) {
            Weights.insert(
                Weights.end(),
                weightsPtr->GetNonTrivialData().begin(),
                weightsPtr->GetNonTrivialData().end()
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
                labels[targetIdx]->end()
            );
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

        NPar::TLocalExecutor executor;
        executor.RunAdditionalThreads(threadCount - 1);

        const int blockSize = 150000;
        TVector<TMetricHolder> additiveStats;

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
                    &executor
                );
                if (!nonAdditiveMetrics.empty()) {
                    nonAdditiveMetricData.SaveProcessedData(datasetPart, &executor);
                }
                if (additiveStats.empty()) {
                    additiveStats = std::move(subStats);
                } else {
                    Y_ASSERT(additiveStats.size() == subStats.size());
                    for (auto ind : xrange(additiveStats.size())) {
                        additiveStats[ind].Add(subStats[ind]);
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

        TVector<TMetricHolder> stats;
        auto additiveStatsPtr = additiveStats.begin();
        auto nonAdditiveStatsPtr = nonAdditiveStats.begin();
        for (const auto& metric : metrics) {
            if (metric->IsAdditiveMetric()) {
                stats.emplace_back(std::move(*(additiveStatsPtr++)));
            } else {
                stats.emplace_back(std::move(*(nonAdditiveStatsPtr++)));
            }
        }
        Y_ASSERT(additiveStatsPtr == additiveStats.end());
        Y_ASSERT(nonAdditiveStatsPtr == nonAdditiveStats.end());

        auto metricResults = GetMetricResultsFromMetricHolder(stats, metrics);
        return metricResults;
    }

    THashMap<TString, double> TOpenSourceCalcMetricsImplementation::CalcMetrics(
        const TPathWithScheme& inputPath,
        const TVector<TString>& metricNamesList,
        const NCB::TDsvFormatOptions& dsvFormat,
        const THashMap<int, EColumn>& nonAuxiliaryColumnsDescription,
        ui32 threadCount
    ) const {
        NCatboostOptions::TDatasetReadingParams datasetReadingParams;
        datasetReadingParams.PoolPath = inputPath;
        datasetReadingParams.ColumnarPoolFormatParams.DsvFormat = dsvFormat;

        Y_ASSERT(inputPath.Scheme == "dsv" || inputPath.Scheme == "");
        ui32 columnCount = GetDsvColumnCount(inputPath, dsvFormat);

        CheckColumnIndices(columnCount, nonAuxiliaryColumnsDescription);

        TVector<TColumn> columnsDescription;
        columnsDescription.resize(columnCount, {EColumn::Auxiliary, TString()});
        for (auto [ind, column] : nonAuxiliaryColumnsDescription) {
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

    void CheckColumnIndices(int columnCount, const THashMap<int, EColumn>& nonAuxiliaryColumnsDescription) {
        for (auto [ind, column] : nonAuxiliaryColumnsDescription) {
            CB_ENSURE(
                ind < columnCount,
                "Index of " << ToString(column)
                    << " column (" << ind << ") is invalid, and should belong to [0, column count)"
            );
        }
    }

    namespace {
        TCalcMetricsImplementationFactory::TRegistrator<TOpenSourceCalcMetricsImplementation>
            OpenSourceCalcMetricsImplementationReg("dsv");
    }
}
