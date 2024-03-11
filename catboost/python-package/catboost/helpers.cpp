#include "helpers.h"

#include <catboost/libs/data/feature_names_converter.h>
#include <catboost/libs/data/sampler.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/matrix.h>
#include <catboost/libs/helpers/permutation.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/options/dataset_reading_params.h>
#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/plain_options_helper.h>
#include <catboost/private/libs/options/split_params.h>
#include <catboost/private/libs/target/data_providers.h>

#include <library/cpp/json/json_reader.h>

#include <util/system/guard.h>
#include <util/system/info.h>
#include <util/system/mutex.h>


using namespace NCB;


extern "C" PyObject* PyCatboostExceptionType;

void ProcessException() {
    try {
        throw;
    } catch (const TCatBoostException& exc) {
        PyErr_SetString(PyCatboostExceptionType, exc.what());
    } catch (const TInterruptException& exc) {
        PyErr_SetString(PyExc_KeyboardInterrupt, exc.what());
    } catch (const std::exception& exc) {
        PyErr_SetString(PyCatboostExceptionType, exc.what());
    }
}

void PyCheckInterrupted() {
    TGilGuard guard;
    if (PyErr_CheckSignals() == -1) {
        throw TInterruptException();
    }
}

void SetPythonInterruptHandler() {
    SetInterruptHandler(PyCheckInterrupted);
}

void ResetPythonInterruptHandler() {
    ResetInterruptHandler();
}

void ThrowCppExceptionWithMessage(const TString& message) {
    ythrow TCatBoostException() << message;
}

TVector<TVector<double>> EvalMetrics(
    const TFullModel& model,
    const NCB::TDataProvider& srcData,
    const TVector<TString>& metricsDescription,
    int begin,
    int end,
    int evalPeriod,
    int threadCount,
    const TString& resultDir,
    const TString& tmpDir
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);

    TRestorableFastRng64 rand(0);

    auto metricLossDescriptions = CreateMetricLossDescriptions(metricsDescription);
    auto metrics = CreateMetrics(metricLossDescriptions, model.GetDimensionsCount());
    TMetricsPlotCalcer plotCalcer = CreateMetricCalcer(
        model,
        begin,
        end,
        evalPeriod,
        /*processedIterationsStep=*/50,
        tmpDir,
        metrics,
        &executor
    );

    auto processedDataProvider = NCB::CreateModelCompatibleProcessedDataProvider(
        srcData,
        metricLossDescriptions,
        model,
        NCB::GetMonopolisticFreeCpuRam(),
        &rand,
        &executor
    );

    if (plotCalcer.HasAdditiveMetric()) {
        plotCalcer.ProceedDataSetForAdditiveMetrics(processedDataProvider);
    }
    if (plotCalcer.HasNonAdditiveMetric()) {
        while (!plotCalcer.AreAllIterationsProcessed()) {
            plotCalcer.ProceedDataSetForNonAdditiveMetrics(processedDataProvider);
            plotCalcer.FinishProceedDataSetForNonAdditiveMetrics();
        }
    }

    TVector<TVector<double>> metricsScore = plotCalcer.GetMetricsScore();

    plotCalcer.SaveResult(resultDir, /*metricsFile=*/"", /*saveMetrics*/ false, /*saveStats=*/true).ClearTempFiles();
    return metricsScore;
}

TVector<TString> GetMetricNames(const TFullModel& model, const TVector<TString>& metricsDescription) {
    auto metrics = CreateMetricsFromDescription(metricsDescription, model.GetDimensionsCount());
    TVector<TString> metricNames;
    metricNames.reserve(metrics.ysize());
    for (auto& metric : metrics) {
        metricNames.push_back(metric->GetDescription());
    }
    return metricNames;
}

TVector<double> EvalMetricsForUtils(
    TConstArrayRef<TVector<float>> label,   // [dimensionIdx][objectIdx]
    const TVector<TVector<double>>& approx, // [dimensionIdx][objectIdx]
    const TString& metricName,
    const TVector<float>& weight,
    const TVector<TGroupId>& groupId,
    const TVector<float>& groupWeight,
    const TVector<TSubgroupId>& subgroupId,
    const TVector<TPair>& pairs,
    int threadCount
) {
    auto objectCount = label[0].size();
    CB_ENSURE(objectCount > 0, "Cannot evaluate metric on empty data");

    CB_ENSURE(!IsGroupwiseMetric(metricName) || !groupId.empty(), "Metric \"" << metricName << "\" requires group data");

    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);
    const int approxDimension = approx.ysize();
    TVector<THolder<IMetric>> metrics = CreateMetricsFromDescription({metricName}, approxDimension);
    if (!weight.empty()) {
        for (auto& metric : metrics) {
            metric->UseWeights.SetDefaultValue(true);
        }
    }
    NCB::TObjectsGrouping objectGrouping = NCB::CreateObjectsGroupingFromGroupIds<TGroupId>(
        objectCount,
        groupId.empty() ? Nothing() : NCB::TMaybeData<TConstArrayRef<TGroupId>>(groupId)
    );
    if (!pairs.empty()) {
        NCB::CheckPairs(pairs, objectGrouping);
    }
    TVector<TQueryInfo> queriesInfo;
    if (!groupId.empty()) {
        queriesInfo = *NCB::MakeGroupInfos(
            objectGrouping,
            subgroupId.empty() ? Nothing() : NCB::TMaybeData<TConstArrayRef<TSubgroupId>>(subgroupId),
            groupWeight.empty() ? NCB::TWeights(groupId.size()) : NCB::TWeights(TVector<float>(groupWeight)),
            TConstArrayRef<TPair>(pairs)
        ).Get();
    }
    TVector<double> metricResults;
    metricResults.reserve(metrics.size());

    TVector<const IMetric*> metricPtrs;
    metricPtrs.reserve(metrics.size());
    for (const auto& metric : metrics) {
        metricPtrs.push_back(metric.Get());
    }

    auto stats = EvalErrorsWithCaching(
        approx,
        /*approxDelts*/{},
        /*isExpApprox*/false,
        To2DConstArrayRef<float>(label),
        weight,
        queriesInfo,
        metricPtrs,
        &executor
    );

    for (auto metricIdx : xrange(metricPtrs.size())) {
        metricResults.push_back(metricPtrs[metricIdx]->GetFinalError(stats[metricIdx]));
    }
    return metricResults;
}

NJson::TJsonValue GetTrainingOptions(
    const NJson::TJsonValue& plainJsonParams,
    const NCB::TDataMetaInfo& trainDataMetaInfo,
    const TMaybe<NCB::TDataMetaInfo>& testDataMetaInfo
) {
    NJson::TJsonValue trainOptionsJson;
    NJson::TJsonValue outputFilesOptionsJson;
    NCatboostOptions::PlainJsonToOptions(plainJsonParams, &trainOptionsJson, &outputFilesOptionsJson);
    ConvertParamsToCanonicalFormat(trainDataMetaInfo, &trainOptionsJson);
    NCatboostOptions::TCatBoostOptions catboostOptions(NCatboostOptions::LoadOptions(trainOptionsJson));
    NCatboostOptions::TOutputFilesOptions outputOptions;
    outputOptions.UseBestModel.SetDefault(false);
    SetDataDependentDefaults(
        trainDataMetaInfo,
        testDataMetaInfo,
        /*continueFromModel*/ false,
        /*learningContinuation*/ false,
        &outputOptions,
        &catboostOptions
    );
    NJson::TJsonValue catboostOptionsJson;
    catboostOptions.Save(&catboostOptionsJson);
    return catboostOptionsJson;
}

size_t GetNumPairs(const NCB::TDataProvider& dataProvider) noexcept {
    size_t result = 0;
    const NCB::TMaybeData<NCB::TRawPairsData>& maybePairsData = dataProvider.RawTargetData.GetPairs();
    if (maybePairsData) {
        std::visit([&](const auto& pairs) { result = pairs.size(); }, *maybePairsData);
    }
    return result;
}

TConstArrayRef<TPair> GetUngroupedPairs(const NCB::TDataProvider& dataProvider) {
    TConstArrayRef<TPair> result;
    const NCB::TMaybeData<NCB::TRawPairsData>& maybePairsData = dataProvider.RawTargetData.GetPairs();
    if (maybePairsData) {
        CB_ENSURE(
            std::holds_alternative<TFlatPairsInfo>(*maybePairsData),
            "Cannot get ungrouped pairs: pairs data is grouped"
        );
        result = std::get<TFlatPairsInfo>(*maybePairsData);
    }
    return result;
}

void TrainEvalSplit(
    const NCB::TDataProvider& srcDataProvider,
    NCB::TDataProviderPtr* trainDataProvider,
    NCB::TDataProviderPtr* evalDataProvider,
    const TTrainTestSplitParams& splitParams,
    bool saveEvalDataset,
    int threadCount,
    ui64 cpuUsedRamLimit
) {
    NPar::TLocalExecutor executor;
    executor.RunAdditionalThreads(threadCount - 1);

    bool shuffle = splitParams.Shuffle && srcDataProvider.ObjectsData->GetOrder() != NCB::EObjectsOrder::RandomShuffled;
    NCB::TObjectsGroupingSubset postShuffleGroupingSubset;
    if (shuffle) {
        TRestorableFastRng64 rand(splitParams.PartitionRandSeed);
        postShuffleGroupingSubset = NCB::Shuffle(srcDataProvider.ObjectsGrouping, 1, &rand);
    } else {
        postShuffleGroupingSubset = NCB::GetSubset(
            srcDataProvider.ObjectsGrouping,
            NCB::TArraySubsetIndexing<ui32>(NCB::TFullSubset<ui32>(srcDataProvider.ObjectsGrouping->GetGroupCount())),
            NCB::EObjectsOrder::Ordered
        );
    }
    auto postShuffleGrouping = postShuffleGroupingSubset.GetSubsetGrouping();

    // for groups
    NCB::TArraySubsetIndexing<ui32> postShuffleTrainIndices;
    NCB::TArraySubsetIndexing<ui32> postShuffleTestIndices;

    if (splitParams.Stratified) {
        auto maybeOneDimensionalTarget = srcDataProvider.RawTargetData.GetOneDimensionalTarget();
        CB_ENSURE(maybeOneDimensionalTarget, "Cannot do stratified split without one-dimensional target data");

        auto doStratifiedSplit = [&](auto targetArrayRef) {
            typedef std::remove_const_t<typename decltype(targetArrayRef)::value_type> TDst;
            TVector<TDst> shuffledTarget;
            if (shuffle) {
                shuffledTarget = NCB::GetSubset<TDst>(targetArrayRef, postShuffleGroupingSubset.GetObjectsIndexing(), &executor);
                targetArrayRef = shuffledTarget;
            }
            NCB::StratifiedTrainTestSplit(
                *postShuffleGrouping,
                targetArrayRef,
                splitParams.TrainPart,
                &postShuffleTrainIndices,
                &postShuffleTestIndices
            );
        };

        std::visit(
            TOverloaded{
                [&](const NCB::ITypedSequencePtr<float>& floatTarget) { doStratifiedSplit(TConstArrayRef<float>(NCB::ToVector(*floatTarget))); },
                [&](const TVector<TString>& stringTarget) { doStratifiedSplit(TConstArrayRef<TString>(stringTarget)); }
            },
            **maybeOneDimensionalTarget
        );
    } else {
        NCB::TrainTestSplit(*postShuffleGrouping, splitParams.TrainPart, &postShuffleTrainIndices, &postShuffleTestIndices);
    }

    auto getSubset = [&](const NCB::TArraySubsetIndexing<ui32>& postShuffleIndexing) {
        return srcDataProvider.GetSubset(
            NCB::GetSubset(
                srcDataProvider.ObjectsGrouping,
                NCB::Compose(postShuffleGroupingSubset.GetGroupsIndexing(), postShuffleIndexing),
                shuffle ? NCB::EObjectsOrder::RandomShuffled : NCB::EObjectsOrder::Ordered
            ),
            cpuUsedRamLimit,
            &executor
        );
    };

    *trainDataProvider = getSubset(postShuffleTrainIndices);
    if (saveEvalDataset) {
        *evalDataProvider = getSubset(postShuffleTestIndices);
    }
}

TAtomicSharedPtr<NPar::TTbbLocalExecutor<false>> GetCachedLocalExecutor(int threadsCount) {
    static TMutex lock;
    static TAtomicSharedPtr<NPar::TTbbLocalExecutor<false>> cachedExecutor;

    CB_ENSURE(threadsCount == -1 || 0 < threadsCount, "threadsCount should be positive or -1");

    if (threadsCount == -1) {
        threadsCount = NSystemInfo::CachedNumberOfCpus();
    }

    with_lock (lock) {
        if (cachedExecutor && cachedExecutor->GetThreadCount() + 1 == threadsCount) {
            return cachedExecutor;
        }

        cachedExecutor.Reset();
        cachedExecutor = MakeAtomicShared<NPar::TTbbLocalExecutor<false>>(threadsCount);

        return cachedExecutor;
    }
}

size_t GetMultiQuantileApproxSize(const TString& lossFunctionDescription) {
    const auto& paramsMap = ParseLossParams(lossFunctionDescription).GetParamsMap();
    return NCatboostOptions::GetAlphaMultiQuantile(paramsMap).size();
}

void GetNumFeatureValuesSample(
    const TFullModel& model,
    const NCatboostOptions::TDatasetReadingParams& datasetReadingParams,
    int threadCount,
    const TVector<ui32>& sampleIndicesVector,
    const TVector<TString>& sampleIdsVector,
    TVector<TArrayRef<float>>* numFeaturesColumns
) {
    auto localExecutor = GetCachedLocalExecutor(threadCount).Get();

    auto sampler = GetProcessor<IDataProviderSampler, TDataProviderSampleParams>(
        datasetReadingParams.PoolPath,
        TDataProviderSampleParams {
            datasetReadingParams,
            /*OnlyFeaturesData*/ true,
            /*CpuUsedRamLimit*/ Max<ui64>(),
            localExecutor
        }
    );

    TDataProviderPtr dataProvider;
    if (!sampleIndicesVector.empty()) {
        dataProvider = sampler->SampleByIndices(sampleIndicesVector);
    } else if (!sampleIdsVector.empty()) {
        dataProvider = sampler->SampleBySampleIds(sampleIdsVector);
    } else {
        CB_ENSURE(false, "Neither indices nor sampleIds are provided");
    }
    auto objectsDataProvider = dataProvider->ObjectsData;
    auto rawObjectsDataProvider = dynamic_cast<const TRawObjectsDataProvider*>(objectsDataProvider.Get());
    CB_ENSURE(rawObjectsDataProvider, "Only non-quantized datasets are supported now");

    for (const auto& floatFeature : model.ModelTrees->GetFloatFeatures()) {
        auto values = (*rawObjectsDataProvider->GetFloatFeature(floatFeature.Position.Index))->ExtractValues(localExecutor);
        auto dst = (*numFeaturesColumns)[floatFeature.Position.FlatIndex];
        Copy(values.begin(), values.end(), dst.begin());
    }
}

TMetricsAndTimeLeftHistory GetTrainingMetrics(const TFullModel& model) {
    if (model.ModelInfo.contains("training"sv)) {
        NJson::TJsonValue trainingJson;
        ReadJsonTree(model.ModelInfo.at("training"sv), &trainingJson, /*throwOnError*/ true);
        const auto& trainingMap = trainingJson.GetMap();
        if (trainingMap.contains("metrics"sv)) {
            return TMetricsAndTimeLeftHistory::LoadMetrics(trainingMap.at("metrics"sv));
        }
    }

    return TMetricsAndTimeLeftHistory();
}
