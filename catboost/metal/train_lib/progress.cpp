#include "progress.h"
#include "query_cross_entropy.h"

#include <catboost/libs/eval_result/eval_result.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/vector_helpers.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/private/libs/algo/apply.h>
#include <util/generic/hash.h>

namespace NCB {
    namespace {
        TErrorTracker MakeErrorTracker(
            const NCatboostOptions::TOverfittingDetectorOptions& options,
            const IMetric& metric)
        {
            float bestValue = 0;
            EMetricBestValue bestValueType;
            metric.GetBestValue(&bestValueType, &bestValue);
            return CreateErrorTracker(options, bestValue, bestValueType, true);
        }

        TVector<TVector<double>> MakeInitialCursor(
            const TTrainingDataProvider& data,
            float bias,
            const TFullModel* initModel,
            const TObjectsDataProvider* initModelObjects,
            NPar::ILocalExecutor* executor,
            ui32 approxDimension,
            TConstArrayRef<ui32> baselineColumns)
        {
            TVector<TVector<double>> cursor(approxDimension,
                TVector<double>(data.GetObjectCount(), approxDimension == 1 ? bias : 0.0));
            if (initModel) {
                CB_ENSURE(initModel->GetDimensionsCount() == approxDimension,
                    "Initial model dimension differs from the Metal training approximation dimension");
                TVector<TVector<double>> initial;
                if (initModel->GetTreeCount() == 0) {
                    // A per-output initial bias is represented by a constant
                    // model. The shared evaluator rejects an empty tree range.
                    const auto& initialBias = initModel->GetScaleAndBias().GetBiasRef();
                    CB_ENSURE(initialBias.size() == approxDimension,
                        "Initial constant model bias dimension differs from Metal training");
                    for (double value : initialBias) {
                        initial.emplace_back(data.GetObjectCount(), value);
                    }
                } else {
                    initial = ApplyModelMulti(
                        *initModel,
                        initModelObjects ? *initModelObjects : *data.ObjectsData,
                        EPredictionType::InternalRawFormulaVal, 0, 0, executor);
                }
                CB_ENSURE(initial.size() == approxDimension,
                    "Initial model output dimension differs from the Metal training approximation dimension");
                for (ui32 dim = 0; dim < approxDimension; ++dim) {
                    CB_ENSURE(initial[dim].size() == cursor[dim].size(),
                        "Initial model apply pool size differs from the training pool");
                    for (size_t row = 0; row < cursor[dim].size(); ++row) {
                        cursor[dim][row] += initial[dim][row];
                    }
                }
            }
            const auto baseline = data.TargetData->GetBaseline();
            if (baseline) {
                for (ui32 dim = 0; dim < approxDimension; ++dim) {
                    const ui32 column = baseline->size() == approxDimension ? dim : baselineColumns[dim];
                    CB_ENSURE(column < baseline->size(),
                        "A mapped Metal training baseline column is missing");
                    CB_ENSURE((*baseline)[column].size() == cursor[dim].size(),
                        "Metal training requires one baseline value per object and dimension");
                    for (size_t row = 0; row < cursor[dim].size(); ++row) {
                        cursor[dim][row] += (*baseline)[column][row];
                    }
                }
            }
            // Metal, like CUDA, maintains its boosting cursor in float32.
            for (auto& dimension : cursor) {
                for (double& value : dimension) {
                    value = static_cast<float>(value);
                }
            }
            return cursor;
        }

        void AddTreeToCursor(
            const TFullModel& tree,
            const TTrainingDataProvider& data,
            NPar::ILocalExecutor* executor,
            TVector<TVector<double>>* cursor)
        {
            const auto delta = ApplyModelMulti(
                tree, *data.ObjectsData, EPredictionType::RawFormulaVal, 0, 1, executor);
            CB_ENSURE(delta.size() == cursor->size(),
                "Unexpected Metal training progress prediction dimensions");
            for (size_t dim = 0; dim < delta.size(); ++dim) {
                CB_ENSURE(delta[dim].size() == (*cursor)[dim].size(),
                    "Unexpected Metal training progress prediction object count");
                for (size_t row = 0; row < delta[dim].size(); ++row) {
                    (*cursor)[dim][row] = static_cast<float>((*cursor)[dim][row] + delta[dim][row]);
                }
            }
        }

        class TQueryCrossEntropyMetricWorkspace {
        public:
            TQueryCrossEntropyMetricWorkspace(const TTrainingDataProvider& data,
                const NCatboostOptions::TLossDescription& loss, TConstArrayRef<float> weights)
                : Point(data.GetObjectCount())
            {
                const auto target=data.TargetData->GetTarget();
                CB_ENSURE(target && target->size()==1,"QueryCrossEntropy metric requires scalar targets");
                const auto query=PrepareMetalQueryData(*data.ObjectsGrouping,loss);
                const auto scales=SelectMetalQueryCrossEntropyScales(loss,(*target)[0],query.Offsets);
                char error[2048]={};
                CB_ENSURE(cbm_query_cross_entropy_metric_create(data.GetObjectCount(),query.Options.group_count,
                    (*target)[0].data(),weights.empty() ? nullptr : weights.data(),query.Offsets.data(),scales.data(),
                    1ull<<30,&Handle,&Bytes,error,sizeof(error))==0,"Metal QueryCrossEntropy metric allocation failed: " << error);
            }
            ~TQueryCrossEntropyMetricWorkspace() { cbm_query_cross_entropy_metric_destroy(Handle); }
            ui64 AllocatedBytes() const { return Bytes; }
            double Evaluate(TConstArrayRef<double> cursor,float alpha) {
                CB_ENSURE(cursor.size()==Point.size(),"QueryCrossEntropy metric prediction size changed");
                for(size_t row=0;row<cursor.size();++row)Point[row]=static_cast<float>(cursor[row]);
                double value=0;uint64_t evaluations=0;char error[2048]={};
                CB_ENSURE(cbm_query_cross_entropy_metric_evaluate(Handle,Point.size(),Point.data(),alpha,
                    &value,&evaluations,error,sizeof(error))==0,"Metal QueryCrossEntropy metric failed: " << error);
                return value;
            }
        private:
            void* Handle=nullptr;
            uint64_t Bytes=0;
            TVector<float> Point;
        };

        class TQueryCrossEntropyMetricCache {
        public:
            double Evaluate(const TTrainingDataProvider& data,const NCatboostOptions::TLossDescription& loss,
                TConstArrayRef<float> weights,TConstArrayRef<double> cursor,float alpha)
            {
                const auto found=Workspaces.find(&data);
                if(found!=Workspaces.end())return found->second->Evaluate(cursor,alpha);
                auto workspace=MakeHolder<TQueryCrossEntropyMetricWorkspace>(data,loss,weights);
                const double result=workspace->Evaluate(cursor,alpha);
                // Bound retention across every learn/evaluation dataset. A
                // larger workspace is released after its one-shot evaluation.
                if(workspace->AllocatedBytes()<=(1ull<<28)-Bytes) {
                    Bytes+=workspace->AllocatedBytes();Workspaces.emplace(&data,std::move(workspace));
                }
                return result;
            }
        private:
            ui64 Bytes=0;
            THashMap<const TTrainingDataProvider*,THolder<TQueryCrossEntropyMetricWorkspace>> Workspaces;
        };

        double EvaluateMetric(
            const IMetric& metric,
            const TTrainingDataProvider& data,
            const TVector<TVector<double>>& cursor,
            NPar::ILocalExecutor* executor,
            const NCatboostOptions::TLossDescription& loss,
            TQueryCrossEntropyMetricCache& qceMetrics)
        {
            const auto target = data.TargetData->GetTarget();
            const auto groupInfo = data.TargetData->GetGroupInfo();
            const auto weights = GetWeights(*data.TargetData);
            TConstArrayRef<TQueryInfo> metricGroups = groupInfo ? *groupInfo : TConstArrayRef<TQueryInfo>();
            TVector<TQueryInfo> rankingGroups;
            const auto description = metric.GetDescription();
            const auto kind = TStringBuf(description).Before(':');
            const bool pairTarget = loss.GetLossFunction() == ELossFunction::PairLogit;
            if (kind == "QueryCrossEntropy" && loss.GetLossFunction() == ELossFunction::QueryCrossEntropy) {
                CB_ENSURE(target && target->size() == 1 && cursor.size() == 1,
                    "QueryCrossEntropy metric requires scalar targets and predictions");
                NCatboostOptions::TLossDescription metricOptions;
                metricOptions.Load(LossDescriptionToJson(description));
                // CUDA's TTargetFallbackMetric always uses target weights and
                // target scales; only alpha comes from the metric description.
                return qceMetrics.Evaluate(data,loss,weights,cursor[0],NCatboostOptions::GetAlphaQueryCrossEntropy(metricOptions));
            }
            if (kind == "NDCG" || kind == "MAP" || kind == "PFound") {
                // CUDA CacheQueryInfo takes the first already-prepared row
                // weight, or one for targets carrying pair weights. Preserve
                // competitors and subgroup IDs while changing only this mass.
                rankingGroups.assign(metricGroups.begin(), metricGroups.end());
                for (auto& group : rankingGroups) {
                    group.Weight = pairTarget || weights.empty() ? 1.0f : weights[group.Begin];
                }
                metricGroups = rankingGroups;
            }
            if (target && !target->empty()) {
                return metric.GetFinalError(EvalErrors(
                    To2DConstArrayRef<double>(cursor), {}, false, *target,
                    weights, metricGroups,
                    metric, executor));
            }
            // Pairwise metrics can evaluate a Pool without target columns.
            // The multi-target overload requires at least one target dimension.
            return metric.GetFinalError(EvalErrors(
                To2DConstArrayRef<double>(cursor), {}, false,
                TConstArrayRef<float>(),
                weights, metricGroups,
                metric, executor));
        }
    }

    class TMetalTrainingProgress::TImpl {
    public:
        TImpl(
            const NCatboostOptions::TCatBoostOptions& options,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TTrainingDataProviders& data,
            float bias,
            NPar::ILocalExecutor* executor,
            const TFullModel* initModel,
            const TDataProviders* initModelApplyCompatiblePools,
            bool forceCalcEvalMetricOnEveryIteration,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
            ui32 approxDimension,
            TConstArrayRef<ui32> baselineColumns)
            : Options(options)
            , OutputOptions(outputOptions)
            , Data(data)
            , Executor(executor)
            , ApproxDimension(approxDimension)
            , BaselineColumns(baselineColumns.begin(), baselineColumns.end())
            , InitialTreeCount(initModel ? initModel->GetTreeCount() : 0)
            , OutputFiles(outputOptions, "")
            , LearnToken(GetTrainModelLearnToken())
            , TestTokens(GetTrainModelTestTokens(data.Test.size()))
            , ProfileInfo(MakeHolder<TProfileInfo>(options.BoostingOptions->IterationCount))
        {
            CB_ENSURE(ApproxDimension > 0, "Metal training approximation dimension must be positive");
            CB_ENSURE(ApproxDimension == 1 || bias == 0,
                "Multidimensional Metal training requires zero scalar bias");
            CB_ENSURE(BaselineColumns.empty() || BaselineColumns.size() == ApproxDimension,
                "Metal baseline column mapping must match the approximation dimension");
            if (BaselineColumns.empty()) {
                BaselineColumns.resize(ApproxDimension);
                for (ui32 dim = 0; dim < ApproxDimension; ++dim) {
                    BaselineColumns[dim] = dim;
                }
            }
            CB_ENSURE(OutputOptions.GetMetricPeriod() > 0, "metric_period must be positive");
            InitializeEvalMetricIfNotSet(Options.LossFunctionDescription, &Options.MetricOptions->EvalMetric);
            Metrics = CreateMetrics(Options.MetricOptions, evalMetricDescriptor, ApproxDimension,
                data.Learn->MetaInfo.HasWeights);
            CheckMetrics(Metrics, Options.LossFunctionDescription->GetLossFunction());
            CB_ENSURE(!Metrics.empty(), "Eval metric is not defined");
            const auto metricPointers = GetConstPointers(Metrics);
            MetricDescriptions = GetMetricsDescription(metricPointers);
            SkipOnLearn = GetSkipMetricOnTrain(metricPointers);
            for (const auto& test : Data.Test) {
                SkipOnTest.push_back(GetSkipMetricOnTest(test->MetaInfo.TargetCount > 0, metricPointers));
            }

            if (!Data.Test.empty() && !SkipOnTest.back()[0]) {
                ErrorTracker = MakeHolder<TErrorTracker>(MakeErrorTracker(
                    Options.BoostingOptions->OverfittingDetector, *Metrics[0]));
                BestModelMinTreesTracker = MakeHolder<TErrorTracker>(MakeErrorTracker(
                    Options.BoostingOptions->OverfittingDetector, *Metrics[0]));
                CalcEvalMetricOnEveryIteration = forceCalcEvalMetricOnEveryIteration || ErrorTracker->IsActive();
                if (OutputOptions.GetMetricPeriod() > 1 && ErrorTracker->IsActive()) {
                    CATBOOST_WARNING_LOG << "Warning: Overfitting detector is active, thus evaluation metric is "
                        "calculated on every iteration. 'metric_period' is ignored for evaluation metric." << Endl;
                }
            } else if (!Data.Test.empty()) {
                CATBOOST_WARNING_LOG << "Warning: Eval metric " << Metrics[0]->GetDescription()
                    << " needs Target data, but the test dataset does not have it so it won't be calculated" << Endl;
            }

            if (initModelApplyCompatiblePools) {
                CB_ENSURE(initModelApplyCompatiblePools->Learn &&
                    initModelApplyCompatiblePools->Test.size() == Data.Test.size(),
                    "Initial model apply pools must match the training and evaluation pools");
            }
            LearnCursor = MakeInitialCursor(*Data.Learn, bias, initModel,
                initModelApplyCompatiblePools ? initModelApplyCompatiblePools->Learn->ObjectsData.Get() : nullptr,
                executor, ApproxDimension, BaselineColumns);
            for (size_t test = 0; test < Data.Test.size(); ++test) {
                TestCursor.push_back(MakeInitialCursor(*Data.Test[test], bias, initModel,
                    initModelApplyCompatiblePools ? initModelApplyCompatiblePools->Test[test]->ObjectsData.Get() : nullptr,
                    executor, ApproxDimension, BaselineColumns));
            }

            if (OutputOptions.AllowWriteFiles()) {
                InitializeFileLoggers(Options, OutputFiles, metricPointers, LearnToken, TestTokens,
                    OutputOptions.GetMetricPeriod(), &Logger);
            }
            AddConsoleLogger(LearnToken, TestTokens, true, OutputOptions.GetVerbosePeriod(),
                Options.BoostingOptions->IterationCount, &Logger);
        }

        void StartIteration() {
            CB_ENSURE(!IterationStarted, "Metal progress iteration already started");
            ProfileInfo->StartNextIteration();
            IterationStarted = true;
        }

        void LogIteration(ui32 iteration, bool outputMetrics, const TProfileResults& profile) {
            // Log() assumes every evaluation pool can calculate every metric.
            // Reuse its iteration logger directly so unlabeled pools can omit
            // target-dependent metrics while retaining the standard backends.
            TOneInterationLogger oneIteration(Logger);
            if (outputMetrics) {
                const auto& learnErrors = History.LearnMetricsHistory[iteration];
                for (size_t metric = 0; metric < MetricDescriptions.size(); ++metric) {
                    const auto& description = MetricDescriptions[metric];
                    if (learnErrors.contains(description)) {
                        oneIteration.OutputMetric(LearnToken,
                            TMetricEvalResult(description, learnErrors.at(description), metric == 0));
                    }
                }
                if (iteration < History.TestMetricsHistory.size()) {
                    for (size_t test = 0; test < Data.Test.size(); ++test) {
                        const auto& testErrors = History.TestMetricsHistory[iteration][test];
                        for (size_t metric = 0; metric < MetricDescriptions.size(); ++metric) {
                            const auto& description = MetricDescriptions[metric];
                            if (!testErrors.contains(description)) {
                                continue;
                            }
                            if (test + 1 == Data.Test.size() && ErrorTracker) {
                                oneIteration.OutputMetric(TestTokens[test], TMetricEvalResult(
                                    description, testErrors.at(description), ErrorTracker->GetBestError(),
                                    ErrorTracker->GetBestIteration(), metric == 0));
                            } else {
                                oneIteration.OutputMetric(TestTokens[test], TMetricEvalResult(
                                    description + ":" + ToString(test), testErrors.at(description), metric == 0));
                            }
                        }
                    }
                }
            }
            oneIteration.OutputProfile(profile);
        }

        bool OnIteration(
            ui32 iteration,
            const TFullModel& tree,
            TConstArrayRef<float> learnCursor,
            const TMetricsAndTimeLeftHistory* restoredHistory = nullptr)
        {
            CB_ENSURE(IterationStarted, "StartIteration must precede Metal progress updates");
            CB_ENSURE(iteration == History.TimeHistory.size(), "Metal progress iterations must be consecutive");
            CB_ENSURE(tree.GetTreeCount() == 1 && tree.GetDimensionsCount() == ApproxDimension,
                "Metal progress expects one tree with the configured approximation dimension per iteration");
            const auto& scaleAndBias = tree.GetScaleAndBias();
            CB_ENSURE(scaleAndBias.Scale == 1.0 && scaleAndBias.IsZeroBias(),
                "Metal progress iteration models must have unit scale and zero bias");
            ProfileInfo->AddOperation("Learn tree");
            if (restoredHistory) {
                CB_ENSURE(iteration < restoredHistory->LearnMetricsHistory.size() &&
                    iteration < restoredHistory->TimeHistory.size(),
                    "Saved Metal progress is missing an iteration");
                CB_ENSURE(Data.Test.empty() ||
                    (iteration < restoredHistory->TestMetricsHistory.size() &&
                    restoredHistory->TestMetricsHistory[iteration].size() == Data.Test.size()),
                    "Saved Metal progress evaluation pools do not match");
                RequireLearnCursor = true;
            } else if (learnCursor.empty()) {
                CB_ENSURE(!RequireLearnCursor,
                    "A live GPU learn cursor is required after replaying Metal progress");
                AddTreeToCursor(tree, *Data.Learn, Executor, &LearnCursor);
            } else {
                CB_ENSURE(learnCursor.size() == ui64(ApproxDimension) * LearnCursor[0].size(),
                    "Metal learn cursor size differs from the training pool");
                for (size_t row = 0; row < LearnCursor[0].size(); ++row) {
                    for (ui32 dim = 0; dim < ApproxDimension; ++dim) {
                        LearnCursor[dim][row] = learnCursor[row * ApproxDimension + dim];
                    }
                }
                RequireLearnCursor = false;
            }
            for (size_t test = 0; test < Data.Test.size(); ++test) {
                AddTreeToCursor(tree, *Data.Test[test], Executor, &TestCursor[test]);
            }
            ProfileInfo->AddOperation("Update approximations");

            // This is the CUDA tracker's metric-period and first-metric convention.
            const bool calcAllMetrics = iteration % OutputOptions.GetMetricPeriod() == 0 ||
                iteration + 1 == Options.BoostingOptions->IterationCount;
            const bool calcTrackerMetric = calcAllMetrics || CalcEvalMetricOnEveryIteration;
            History.LearnMetricsHistory.emplace_back();
            if (restoredHistory) {
                const auto& learnErrors = restoredHistory->LearnMetricsHistory[iteration];
                for (const auto& metric : Metrics) {
                    const auto error = learnErrors.find(metric->GetDescription());
                    if (error != learnErrors.end()) {
                        History.AddLearnError(*metric, error->second);
                    }
                }
            } else if (calcAllMetrics) {
                for (size_t metric = 0; metric < Metrics.size(); ++metric) {
                    if (!SkipOnLearn[metric]) {
                        History.AddLearnError(*Metrics[metric],
                            EvaluateMetric(*Metrics[metric], *Data.Learn, LearnCursor, Executor,
                                Options.LossFunctionDescription.Get(),QceMetrics));
                    }
                }
            }
            if (!Data.Test.empty()) {
                History.TestMetricsHistory.emplace_back(Data.Test.size());
                for (size_t test = 0; test < Data.Test.size(); ++test) {
                    for (size_t metric = 0; metric < Metrics.size(); ++metric) {
                        double error;
                        if (restoredHistory) {
                            const auto& testErrors = restoredHistory->TestMetricsHistory[iteration][test];
                            const auto savedError = testErrors.find(Metrics[metric]->GetDescription());
                            if (savedError == testErrors.end()) {
                                continue;
                            }
                            error = savedError->second;
                        } else {
                            if (SkipOnTest[test][metric] || (!calcAllMetrics && !(metric == 0 && calcTrackerMetric))) {
                                continue;
                            }
                            error = EvaluateMetric(*Metrics[metric], *Data.Test[test], TestCursor[test], Executor,
                                Options.LossFunctionDescription.Get(),QceMetrics);
                        }
                        const bool isTrackerMetric = metric == 0 && test + 1 == Data.Test.size();
                        History.AddTestError(test, *Metrics[metric], error, isTrackerMetric);
                        if (isTrackerMetric) {
                            ErrorTracker->AddError(error, iteration);
                            if (OutputOptions.UseBestModel && static_cast<int>(iteration + 1) >= OutputOptions.BestModelMinTrees) {
                                BestModelMinTreesTracker->AddError(error, iteration);
                                if (BestModelMinTreesTracker->GetBestIteration() == static_cast<int>(iteration)) {
                                    BestTestCursor = TestCursor;
                                }
                            }
                        }
                    }
                }
            }
            ProfileInfo->AddOperation("Calc errors");
            ProfileInfo->FinishIteration();
            IterationStarted = false;
            auto profile = ProfileInfo->GetProfileResults();
            if (restoredHistory) {
                const auto& savedTime = restoredHistory->TimeHistory[iteration];
                profile = TProfileResults(savedTime.PassedTime, savedTime.RemainingTime,
                    true, savedTime.IterationTime, iteration + 1);
            }
            History.TimeHistory.emplace_back(profile);
            const bool outputMetrics = calcAllMetrics || (restoredHistory &&
                !restoredHistory->LearnMetricsHistory[iteration].empty());
            LogIteration(iteration, outputMetrics, profile);
            const bool needStop = ErrorTracker && ErrorTracker->GetIsNeedStop();
            if (needStop) {
                CATBOOST_NOTICE_LOG << "Stopped by overfitting detector ("
                    << ErrorTracker->GetOverfittingDetectorIterationsWait() << " iterations wait)" << Endl;
            }
            return !needStop;
        }

        bool ReplayIteration(
            ui32 iteration,
            const TFullModel& tree,
            const TMetricsAndTimeLeftHistory& restoredHistory)
        {
            StartIteration();
            return OnIteration(iteration, tree, {}, &restoredHistory);
        }

        void RestoreTimeHistory(const TVector<TTimeInfo>& timeHistory) {
            CB_ENSURE(!IterationStarted, "Cannot restore Metal profile during an iteration");
            CB_ENSURE(timeHistory.size() == History.TimeHistory.size(),
                "Restored Metal profile must match the replayed iteration count");
            History.TimeHistory = timeHistory;
            ProfileInfo = MakeHolder<TProfileInfo>(Options.BoostingOptions->IterationCount);
            TProfileInfoData profileData;
            profileData.PassedIterations = timeHistory.size();
            profileData.PassedTime = timeHistory.empty() ? 0.0 : timeHistory.back().PassedTime;
            ProfileInfo->InitProfileInfo(std::move(profileData));
        }

        void Finish(TFullModel* model, const TVector<TEvalResult*>& evalResult) {
            CB_ENSURE(model, "Metal progress requires a model to finalize");
            CB_ENSURE(evalResult.empty() || evalResult.size() == TestCursor.size(),
                "Evaluation results must match the number of evaluation pools");
            if (ErrorTracker && ErrorTracker->GetBestIteration() >= 0) {
                CATBOOST_NOTICE_LOG << "bestTest = " << ErrorTracker->GetBestError() << Endl;
                CATBOOST_NOTICE_LOG << "bestIteration = " << ErrorTracker->GetBestIteration() << Endl;
            }
            bool useBestCursor = false;
            if (OutputOptions.ShrinkModelToBestIteration()) {
                if (!ErrorTracker) {
                    CATBOOST_INFO_LOG << "Warning: can't use-best-model without an evaluation metric; "
                        "will skip model shrinking" << Endl;
                } else if (BestModelMinTreesTracker->GetBestIteration() >= 0) {
                    const size_t bestNewTrees = BestModelMinTreesTracker->GetBestIteration() + 1;
                    const size_t bestTotalTrees = InitialTreeCount + bestNewTrees;
                    CB_ENSURE(bestTotalTrees <= model->GetTreeCount(), "Best iteration exceeds the Metal model tree count");
                    if (bestTotalTrees < model->GetTreeCount()) {
                        CATBOOST_NOTICE_LOG << "Shrink model to first " << bestTotalTrees << " iterations.";
                        if (ErrorTracker->GetBestIteration() + 1 < static_cast<int>(bestNewTrees)) {
                            CATBOOST_NOTICE_LOG << " (min iterations for best model = " << OutputOptions.BestModelMinTrees << ")";
                        }
                        CATBOOST_NOTICE_LOG << Endl;
                        model->Truncate(0, bestTotalTrees);
                        useBestCursor = true;
                    }
                }
            }
            auto& outputCursor = useBestCursor ? BestTestCursor : TestCursor;
            CB_ENSURE(!useBestCursor || outputCursor.size() == TestCursor.size(),
                "Best Metal evaluation approximations are missing");
            for (size_t test = 0; test < evalResult.size(); ++test) {
                if (evalResult[test]) {
                    evalResult[test]->SetRawValuesByMove(outputCursor[test]);
                }
            }
            if (Options.IsProfile) {
                LogAverages(ProfileInfo->GetProfileResults());
            }
        }

        NCatboostOptions::TCatBoostOptions Options;
        const NCatboostOptions::TOutputFilesOptions OutputOptions;
        const TTrainingDataProviders& Data;
        NPar::ILocalExecutor* const Executor;
        const ui32 ApproxDimension;
        TVector<ui32> BaselineColumns;
        const size_t InitialTreeCount;
        TOutputFiles OutputFiles;
        TMetricsAndTimeLeftHistory History;
        TLogger Logger;
        TVector<THolder<IMetric>> Metrics;
        TQueryCrossEntropyMetricCache QceMetrics;
        THolder<TErrorTracker> ErrorTracker;
        THolder<TErrorTracker> BestModelMinTreesTracker;
        const TString LearnToken;
        const TVector<const TString> TestTokens;
        THolder<TProfileInfo> ProfileInfo;
        TVector<TString> MetricDescriptions;
        TVector<bool> SkipOnLearn;
        TVector<TVector<bool>> SkipOnTest;
        TVector<TVector<double>> LearnCursor;
        TVector<TVector<TVector<double>>> TestCursor;
        TVector<TVector<TVector<double>>> BestTestCursor;
        bool CalcEvalMetricOnEveryIteration = false;
        bool IterationStarted = false;
        bool RequireLearnCursor = false;
    };

    TMetalTrainingProgress::TMetalTrainingProgress(
        const NCatboostOptions::TCatBoostOptions& options,
        const NCatboostOptions::TOutputFilesOptions& outputOptions,
        const TTrainingDataProviders& data,
        float bias,
        NPar::ILocalExecutor* executor,
        const TFullModel* initModel,
        const TDataProviders* initModelApplyCompatiblePools,
        bool forceCalcEvalMetricOnEveryIteration,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        ui32 approxDimension,
        TConstArrayRef<ui32> baselineColumns)
        : Impl(MakeHolder<TImpl>(options, outputOptions, data, bias, executor, initModel,
            initModelApplyCompatiblePools, forceCalcEvalMetricOnEveryIteration, evalMetricDescriptor,
            approxDimension, baselineColumns))
    {}

    TMetalTrainingProgress::~TMetalTrainingProgress() = default;

    void TMetalTrainingProgress::StartIteration() {
        Impl->StartIteration();
    }

    bool TMetalTrainingProgress::OnIteration(
        ui32 iteration,
        const TFullModel& singleTreeModel,
        TConstArrayRef<float> learnCursor)
    {
        return Impl->OnIteration(iteration, singleTreeModel, learnCursor);
    }

    const TMetricsAndTimeLeftHistory& TMetalTrainingProgress::GetHistory() const {
        return Impl->History;
    }

    bool TMetalTrainingProgress::ReplayIteration(
        ui32 iteration,
        const TFullModel& singleTreeModel,
        const TMetricsAndTimeLeftHistory& restoredHistory)
    {
        return Impl->ReplayIteration(iteration, singleTreeModel, restoredHistory);
    }

    void TMetalTrainingProgress::RestoreTimeHistory(const TVector<TTimeInfo>& timeHistory) {
        Impl->RestoreTimeHistory(timeHistory);
    }

    void TMetalTrainingProgress::Finish(TFullModel* model, const TVector<TEvalResult*>& evalResult) {
        Impl->Finish(model, evalResult);
    }
}
