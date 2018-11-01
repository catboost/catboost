#include "plot.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/options/loss_description.h>
#include <catboost/libs/model/model_pool_compatibility.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/generic/array_ref.h>
#include <util/generic/xrange.h>


TMetricsPlotCalcer::TMetricsPlotCalcer(
    const TFullModel& model,
    const TVector<THolder<IMetric>>& metrics,
    NPar::TLocalExecutor& executor,
    const TString& tmpDir,
    ui32 first,
    ui32 last,
    ui32 step,
    ui32 processIterationStep)
    : Model(model)
    , Executor(executor)
    , First(first)
    , Last(last)
    , Step(step)
    , TmpDir(tmpDir)
    , ProcessedIterationsCount(0)
    , ProcessedIterationsStep(processIterationStep)
{
    EnsureCorrectParams();
    for (ui32 iteration = First; iteration < Last; iteration += Step) {
        Iterations.push_back(iteration);
    }
    if (Iterations.back() != Last - 1) {
        Iterations.push_back(Last - 1);
    }
    for (int metricIndex = 0; metricIndex < metrics.ysize(); ++metricIndex) {
        const auto& metric = metrics[metricIndex];
        if (metric->IsAdditiveMetric()) {
            AdditiveMetrics.push_back(metric.Get());
            AdditiveMetricsIndices.push_back(metricIndex);
        }
        else {
            NonAdditiveMetrics.push_back(metric.Get());
            NonAdditiveMetricsIndices.push_back(metricIndex);
            CB_ENSURE(metric->GetErrorType() == EErrorType::PerObjectError,
                "Error: we don't support non-additive querywise and pairwise metrics currenty");
        }
    }
    AdditiveMetricPlots.resize(AdditiveMetrics.ysize(), TVector<TMetricHolder>(Iterations.ysize()));
    NonAdditiveMetricPlots.resize(NonAdditiveMetrics.ysize(), TVector<TMetricHolder>(Iterations.ysize()));
}

void TMetricsPlotCalcer::ComputeAdditiveMetric(
    const TVector<TVector<double>>& approx,
    const TVector<float>& target,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    ui32 plotLineIndex
) {
    for (ui32 metricId = 0; metricId < AdditiveMetrics.size(); ++metricId) {
        const auto& metric = *AdditiveMetrics[metricId];
        ELossFunction lossFunction = ParseLossType(metric.GetDescription());
        CheckTarget(target, lossFunction);

        const auto docCount = static_cast<int>(target.size());
        const auto queryCount = static_cast<int>(queriesInfo.size());
        TMetricHolder metricResult;
        if (metric.GetErrorType() == EErrorType::PerObjectError) {
            metricResult = metric.Eval(approx, target, weights, queriesInfo, 0, docCount, Executor);
        } else {
            CB_ENSURE(metric.GetErrorType() == EErrorType::QuerywiseError || metric.GetErrorType() == EErrorType::PairwiseError);
            metricResult = metric.Eval(approx, target, weights, queriesInfo, 0, queryCount, Executor);
        }
        AdditiveMetricPlots[metricId][plotLineIndex].Add(metricResult);
    }
}

void TMetricsPlotCalcer::Append(const TVector<TVector<double>>& approx, TVector<TVector<double>>* dst, int dstStartDoc) {
    const ui32 docCount = approx[0].size();

    for (ui32 dim = 0; dim < approx.size(); ++dim) {
        NPar::ParallelFor(Executor, 0, docCount, [&](int i) {
            (*dst)[dim][dstStartDoc + i] += approx[dim][i];
        });
    };
}

static void ResizePool(int size, const TPool& basePool, TPool* pool) {
    pool->Docs.Resize(
        size,
        basePool.Docs.GetEffectiveFactorCount(),
        basePool.Docs.GetBaselineDimension(),
        !basePool.Docs.QueryId.empty(),
        !basePool.Docs.SubgroupId.empty()
    );
}

TPool TMetricsPlotCalcer::ProcessBoundaryGroups(const TPool& rawPool) {
    TPool resultPool;
    resultPool.Swap(LastGroupPool);

    const int offset = resultPool.Docs.GetDocCount();
    const int rawPoolSize = rawPool.Docs.GetDocCount();
    ResizePool(offset + rawPoolSize, rawPool, &resultPool);
    for (int docId = 0; docId < rawPoolSize; ++docId) {
        resultPool.Docs.AssignDoc(offset + docId, rawPool.Docs, docId);
    }

    int lastQuerySize = 0;
    const TGroupId lastQueryId = rawPool.Docs.QueryId.back();
    for (auto queryIdIt = rawPool.Docs.QueryId.rbegin(); queryIdIt != rawPool.Docs.QueryId.rend(); ++queryIdIt) {
        if (lastQueryId == *queryIdIt) {
            ++lastQuerySize;
        } else {
            break;
        }
    }
    ResizePool(lastQuerySize, resultPool, &LastGroupPool);
    const int newResultPoolSize = offset + rawPoolSize - lastQuerySize;
    for (int docId = 0; docId < lastQuerySize; ++docId) {
        LastGroupPool.Docs.AssignDoc(docId, resultPool.Docs, newResultPoolSize + docId);
    }
    LastGroupPool.CatFeatures = rawPool.CatFeatures;

    if (resultPool.CatFeatures.size() != LastGroupPool.CatFeatures.size()) {
        resultPool.CatFeatures = LastGroupPool.CatFeatures;
    }

    ResizePool(newResultPoolSize, resultPool, &resultPool);
    CB_ENSURE(resultPool.Docs.GetDocCount() != 0, "The size of the queries should be less than block-size parameter.");
    return resultPool;
}

TMetricsPlotCalcer& TMetricsPlotCalcer::ProceedDataSetForAdditiveMetrics(const TPool& pool, bool isProcessBoundaryGroups) {
    ProceedDataSet(pool, 0, Iterations.ysize(), isProcessBoundaryGroups, /*isAdditiveMetrics=*/true);
    return *this;
}

TMetricsPlotCalcer& TMetricsPlotCalcer::FinishProceedDataSetForAdditiveMetrics() {
    if (LastGroupPool.Docs.GetDocCount() != 0) {
        ProceedDataSet(LastGroupPool, 0, Iterations.ysize(), /*isProcessBoundaryGroups=*/false, /*isAdditiveMetrics=*/true);
    }
    return *this;
}

TMetricsPlotCalcer& TMetricsPlotCalcer::ProceedDataSetForNonAdditiveMetrics(const TPool& pool) {
    if (ProcessedIterationsCount == 0) {
        const ui32 newPoolSize = NonAdditiveMetricsData.Target.size() + pool.Docs.Target.size();
        NonAdditiveMetricsData.Target.reserve(newPoolSize);
        NonAdditiveMetricsData.Weights.reserve(newPoolSize);
        NonAdditiveMetricsData.Target.insert(NonAdditiveMetricsData.Target.end(), pool.Docs.Target.begin(), pool.Docs.Target.end());
        NonAdditiveMetricsData.Weights.insert(NonAdditiveMetricsData.Weights.end(), pool.Docs.Weight.begin(), pool.Docs.Weight.end());
    }
    ui32 begin = ProcessedIterationsCount;
    ui32 end = Min<ui32>(ProcessedIterationsCount + ProcessedIterationsStep, Iterations.size());
    ProceedDataSet(pool, begin, end, /*isProcessBoundaryGroups=*/false, /*isAdditiveMetrics=*/false);
    return *this;
}

TMetricsPlotCalcer& TMetricsPlotCalcer::FinishProceedDataSetForNonAdditiveMetrics() {
    ui32 begin = ProcessedIterationsCount;
    ui32 end = Min<ui32>(ProcessedIterationsCount + ProcessedIterationsStep, Iterations.size());
    ComputeNonAdditiveMetrics(begin, end);
    ProcessedIterationsCount = end;
    if (AreAllIterationsProcessed()) {
        DeleteApprox(end - 1);
    } else {
        LastApproxes = MakeHolder<TIFStream>(GetApproxFileName(end - 1));
    }
    return *this;
}

static void Load(ui32 docCount, IInputStream* input, TVector<TVector<double>>* output) {
    TVector<double> line;
    for (ui32 i = 0; i < docCount; ++i) {
        ::Load(input, line);
        for (ui32 dim = 0; dim < output->size(); ++dim) {
            (*output)[dim][i] = line[dim];
        }
    }
}

static int GetDocCount(TConstArrayRef<TPool> poolParts) {
    int answer = 0;
    for (const auto& pool : poolParts) {
        answer += pool.Docs.GetDocCount();
    }
    return answer;
}

static void InitApproxBuffer(
    int approxDimension,
    TConstArrayRef<TPool> datasetParts,
    bool initBaselineIfAvailable,
    TVector<TVector<double>>* approxMatrix
) {
    approxMatrix->resize(approxDimension);
    if (datasetParts.empty())
        return;

    bool hasBaseline = false;
    if (initBaselineIfAvailable) {
        hasBaseline = !datasetParts[0].Docs.Baseline.empty();
        for (auto datasetPartIdx : xrange<size_t>(1, datasetParts.size())) {
            CB_ENSURE(
                !datasetParts[datasetPartIdx].Docs.Baseline.empty() == hasBaseline,
                "Inconsistent baseline specification between dataset parts: part 0 has "
                << (hasBaseline ? "" : "no ") << " baseline, but part " << datasetPartIdx << " has"
                << (hasBaseline ? " not" : "")
            );
        }
    }

    int docCount = GetDocCount(datasetParts);

    for (auto approxIdx : xrange(approxDimension)) {
        auto& approx = (*approxMatrix)[approxIdx];
        if (hasBaseline) {
            approx.reserve(docCount);
            for (const auto& datasetPart : datasetParts) {
                const auto& baselinePart = datasetPart.Docs.Baseline[approxIdx];
                approx.insert(approx.end(), baselinePart.begin(), baselinePart.end());
            }
            Y_ASSERT(approx.ysize() == docCount);
        } else {
            approx.resize(docCount);
        }
    }
}

static void ClearApproxBuffer(TVector<TVector<double>>* approxMatrix) {
    for (auto& approx : *approxMatrix) {
        approx.clear();
    }
}

TMetricsPlotCalcer& TMetricsPlotCalcer::ProceedDataSet(
    const TPool& rawPool,
    ui32 beginIterationIndex,
    ui32 endIterationIndex,
    bool isProcessBoundaryGroups,
    bool isAdditiveMetrics
) {
    TPool tmpPool;
    if (isProcessBoundaryGroups) {
        tmpPool = ProcessBoundaryGroups(rawPool);
    }
    const TPool& pool = isProcessBoundaryGroups ? tmpPool : rawPool;
    TModelCalcerOnPool modelCalcerOnPool(Model, pool, &Executor);
    TVector<TQueryInfo> queriesInfo;
    const TVector<float>& groupWeight = pool.MetaInfo.HasGroupWeight ? pool.Docs.Weight : TVector<float>();
    UpdateQueriesInfo(pool.Docs.QueryId, groupWeight, pool.Docs.SubgroupId, 0, pool.Docs.GetDocCount(), &queriesInfo);
    UpdateQueriesPairs(pool.Pairs, 0, pool.Pairs.ysize(), /*invertedPermutation=*/{}, &queriesInfo);
    const ui32 docCount = pool.Docs.GetDocCount();
    InitApproxBuffer(
        Model.ObliviousTrees.ApproxDimension,
        MakeArrayRef(&pool, 1),
        /*initBaselineIfAvailable*/ beginIterationIndex == 0,
        &CurApproxBuffer
    );

    ui32 begin, end;
    if (beginIterationIndex == 0) {
        begin = 0;
    } else {
        begin = Iterations[beginIterationIndex];
        Load(docCount, LastApproxes.Get(), &CurApproxBuffer);
    }

    for (ui32 iterationIndex = beginIterationIndex; iterationIndex < endIterationIndex; ++iterationIndex) {
        end = Iterations[iterationIndex] + 1;
        modelCalcerOnPool.ApplyModelMulti(EPredictionType::InternalRawFormulaVal, begin, end, &FlatApproxBuffer, &NextApproxBuffer);
        Append(NextApproxBuffer, &CurApproxBuffer);
        if (isAdditiveMetrics) {
            ComputeAdditiveMetric(CurApproxBuffer, pool.Docs.Target, pool.Docs.Weight, queriesInfo, iterationIndex);
        } else {
            SaveApproxToFile(iterationIndex, CurApproxBuffer);
        }
        begin = end;
    }
    ClearApproxBuffer(&CurApproxBuffer);
    ClearApproxBuffer(&NextApproxBuffer);

    return *this;
}

void TMetricsPlotCalcer::ComputeNonAdditiveMetrics(ui32 begin, ui32 end) {
    const auto& target = NonAdditiveMetricsData.Target;
    const auto& weights = NonAdditiveMetricsData.Weights;
    for (ui32 idx = begin; idx < end; ++idx) {
        auto approx = LoadApprox(idx);
        for (ui32 metricId = 0; metricId < NonAdditiveMetrics.size(); ++metricId) {
            NonAdditiveMetricPlots[metricId][idx] = NonAdditiveMetrics[metricId]->Eval(approx, target, weights, {}, 0, target.size(), Executor);
        }
        if (idx != 0) {
            DeleteApprox(idx - 1);
        }
    }
}

static TVector<float> BuildTargets(const TVector<TPool>& poolParts) {
    TVector<float> result;
    result.reserve(GetDocCount(poolParts));
    for (const auto& pool : poolParts) {
        result.insert(result.end(), pool.Docs.Target.begin(), pool.Docs.Target.end());
    }
    return result;
}

static TVector<float> BuildWeights(const TVector<TPool>& poolParts) {
    TVector<float> result;
    result.reserve(GetDocCount(poolParts));
    for (const auto& pool : poolParts) {
        result.insert(result.end(), pool.Docs.Weight.begin(), pool.Docs.Weight.end());
    }
    return result;
}

static TVector<int> GetStartDocIdx(const TVector<TPool>& poolParts) {
    TVector<int> result;
    result.reserve(poolParts.size());
    int start = 0;
    for (const auto& pool : poolParts) {
        result.push_back(start);
        start += pool.Docs.GetDocCount();
    }
    return result;
}

void TMetricsPlotCalcer::ComputeNonAdditiveMetrics(const TVector<TPool>& datasetParts) {
    TVector<float> allTargets = BuildTargets(datasetParts);
    TVector<float> allWeights = BuildWeights(datasetParts);

    TVector<TVector<double>> curApprox;
    InitApproxBuffer(
        Model.ObliviousTrees.ApproxDimension,
        datasetParts,
        /*initBaselineIfAvailable*/ true,
        &curApprox
    );

    int begin = 0;
    TVector<TModelCalcerOnPool> modelCalcers;
    for (const auto& pool : datasetParts) {
        modelCalcers.emplace_back(Model, pool, &Executor);
    }

    auto startDocIdx = GetStartDocIdx(datasetParts);
    for (ui32 iterationIndex = 0; iterationIndex < Iterations.size(); ++iterationIndex) {
        int end = Iterations[iterationIndex] + 1;
        for (int poolPartIdx = 0; poolPartIdx < modelCalcers.ysize(); ++poolPartIdx) {
            auto& calcer = modelCalcers[poolPartIdx];
            calcer.ApplyModelMulti(EPredictionType::InternalRawFormulaVal, begin, end, &FlatApproxBuffer, &NextApproxBuffer);
            Append(NextApproxBuffer, &curApprox, startDocIdx[poolPartIdx]);
        }

        for (ui32 metricId = 0; metricId < NonAdditiveMetrics.size(); ++metricId) {
            NonAdditiveMetricPlots[metricId][iterationIndex] = NonAdditiveMetrics[metricId]->Eval(curApprox, allTargets, allWeights, {}, 0, allTargets.size(), Executor);
        }
        begin = end;
    }
}

TString TMetricsPlotCalcer::GetApproxFileName(ui32 plotLineIndex) {
    const ui32 plotSize = plotLineIndex + 1;
    if (NonAdditiveMetricsData.ApproxFiles.size() < plotSize) {
        NonAdditiveMetricsData.ApproxFiles.resize(plotSize);
    }
    if (NonAdditiveMetricsData.ApproxFiles[plotLineIndex].Empty()) {
        if (!NFs::Exists(TmpDir)) {
            NFs::MakeDirectory(TmpDir);
            DeleteTmpDirOnExitFlag = true;
        }
        TString name = TStringBuilder() << CreateGuidAsString() << "_approx_" << plotLineIndex << ".tmp";
        auto path = JoinFsPaths(TmpDir, name);
        if (NFs::Exists(path)) {
            CATBOOST_INFO_LOG << "Path already exists " << path << ". Will overwrite file" << Endl;
            NFs::Remove(path);
        }
        NonAdditiveMetricsData.ApproxFiles[plotLineIndex] = path;
    }
    return NonAdditiveMetricsData.ApproxFiles[plotLineIndex];
}

void TMetricsPlotCalcer::SaveApproxToFile(ui32 plotLineIndex,
                                          const TVector<TVector<double>>& approx) {
    auto fileName = GetApproxFileName(plotLineIndex);
    ui32 docCount = approx[0].size();
    TFile file(fileName, EOpenModeFlag::ForAppend | EOpenModeFlag::OpenAlways);
    TOFStream out(file);
    TVector<double> line(approx.size());

    for (ui32 i = 0; i < docCount; ++i) {
        for (ui32 dim = 0; dim < approx.size(); ++dim) {
            line[dim] = approx[dim][i];
        }
        ::Save(&out, line);
    }
}

TVector<TVector<double>> TMetricsPlotCalcer::LoadApprox(ui32 plotLineIndex) {
    TIFStream input(GetApproxFileName(plotLineIndex));
    ui32 docCount = NonAdditiveMetricsData.Target.size();
    TVector<TVector<double>> result(Model.ObliviousTrees.ApproxDimension, TVector<double>(docCount));
    Load(docCount, &input, &result);
    return result;
}

void TMetricsPlotCalcer::DeleteApprox(ui32 plotLineIndex) {
    NFs::Remove(GetApproxFileName(plotLineIndex));
}

static inline ELossFunction ReadLossFunction(const TString& modelInfoParams) {
    return ParseLossType(ReadTJsonValue(modelInfoParams)["loss_function"]["type"].GetStringSafe());
}

TMetricsPlotCalcer CreateMetricCalcer(
    const TFullModel& model,
    int begin,
    int end,
    int evalPeriod,
    int processedIterationsStep,
    NPar::TLocalExecutor& executor,
    const TString& tmpDir,
    const TVector<THolder<IMetric>>& metrics
) {
    if (ReadTJsonValue(model.ModelInfo.at("params")).Has("loss_function")) {
        CheckMetrics(metrics, ReadLossFunction(model.ModelInfo.at("params")));
    }

    if (end == 0) {
        end = model.GetTreeCount();
    } else {
        end = Min<int>(end, model.GetTreeCount());
    }

    TMetricsPlotCalcer plotCalcer(model, metrics, executor, tmpDir, begin, end, evalPeriod, processedIterationsStep);

    return plotCalcer;
}

TVector<TVector<double>> TMetricsPlotCalcer::GetMetricsScore() {
    TVector<TVector<double>> metricsScore(AdditiveMetrics.size() + NonAdditiveMetrics.size(), TVector<double>(Iterations.size()));
    for (ui32 i = 0; i < Iterations.size(); ++i) {
        for (ui32 metricId = 0; metricId < AdditiveMetrics.size(); ++metricId) {
            metricsScore[AdditiveMetricsIndices[metricId]][i] = AdditiveMetrics[metricId]->GetFinalError(AdditiveMetricPlots[metricId][i]);
        }
        for (ui32 metricId = 0; metricId < NonAdditiveMetrics.size(); ++metricId) {
            metricsScore[NonAdditiveMetricsIndices[metricId]][i] = NonAdditiveMetrics[metricId]->GetFinalError(NonAdditiveMetricPlots[metricId][i]);
        }
    }
    return metricsScore;
}

static TLogger CreateLogger(
    const TString& token,
    const TString& resultDir,
    const TString& metricsFile,
    int iterationBegin,
    int iterationEnd,
    int iterationPeriod,
    bool saveMetrics,
    const TVector<const IMetric*>& metrics
) {
    TLogger logger(iterationBegin, iterationEnd - 1, iterationPeriod);
    if (saveMetrics) {
        logger.AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TErrorFileLoggingBackend(JoinFsPaths(resultDir, metricsFile))));
    }
    logger.AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TTensorBoardLoggingBackend(JoinFsPaths(resultDir, token))));

    const int iterationsCount = ceil((iterationEnd - iterationBegin) / (iterationPeriod + 0.0));
    auto metaJson = GetJsonMeta(iterationsCount, /*optionalExperimentName=*/"", metrics, /*learnSetNames=*/{}, {token}, ELaunchMode::Eval);
    logger.AddBackend(token, TIntrusivePtr<ILoggingBackend>(new TJsonLoggingBackend(JoinFsPaths(resultDir, "catboost_training.json"), metaJson)));
    return logger;
}

TMetricsPlotCalcer& TMetricsPlotCalcer::SaveResult(const TString& resultDir, const TString& metricsFile, bool saveMetrics, bool saveStats) {
    TFsPath trainDirPath(resultDir);
    if (!resultDir.empty() && !trainDirPath.Exists()) {
        trainDirPath.MkDirs();
    }

    if (saveStats) {
        TOFStream statsStream(JoinFsPaths(resultDir, "partial_stats.tsv"));
        const char sep = '\t';
        WriteHeaderForPartialStats(&statsStream, sep);
        WritePartialStats(&statsStream, sep);
    }


    TVector<const IMetric*> metrics(AdditiveMetrics.size() + NonAdditiveMetrics.size());
    for (ui32 metricId = 0; metricId < AdditiveMetrics.size(); ++metricId) {
        metrics[AdditiveMetricsIndices[metricId]] = AdditiveMetrics[metricId];
    }
    for (ui32 metricId = 0; metricId < NonAdditiveMetrics.size(); ++metricId) {
        metrics[NonAdditiveMetricsIndices[metricId]] = NonAdditiveMetrics[metricId];
    }

    TVector<TVector<double>> results = GetMetricsScore();

    const TString token = "eval_dataset";
    TLogger logger = CreateLogger(token, resultDir, metricsFile, First, Last, Step, saveMetrics, metrics);
    for (int iteration = 0; iteration < results[0].ysize(); ++iteration) {
        TOneInterationLogger oneIterLogger(logger);
        for (int metricIdx = 0; metricIdx < results.ysize(); ++metricIdx) {
            oneIterLogger.OutputMetric(token, TMetricEvalResult(metrics[metricIdx]->GetDescription(), results[metricIdx][iteration], false));
        }
    }
    return *this;
}
