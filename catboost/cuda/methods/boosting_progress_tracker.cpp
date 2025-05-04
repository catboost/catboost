#include "boosting_progress_tracker.h"
#include <catboost/cuda/cuda_lib/cuda_base.h>

namespace NCatboostCuda {
    static inline TErrorTracker CreateErrorTracker(const NCatboostOptions::TOverfittingDetectorOptions& odOptions,
                                                   const IMetric& metric,
                                                   bool hasTest) {
        float bestValue = 0;
        EMetricBestValue metricBestValueType;
        metric.GetBestValue(&metricBestValueType, &bestValue);
        return CreateErrorTracker(odOptions, static_cast<double>(bestValue), metricBestValueType, hasTest);
    }

    static inline TVector<const IMetric*> GetCpuMetrics(const TVector<THolder<IGpuMetric>>& metrics) {
        TVector<const IMetric*> cpuMetrics;
        for (size_t i = 0; i < metrics.size(); ++i) {
            cpuMetrics.push_back(&metrics[i]->GetCpuMetric());
        }
        return cpuMetrics;
    }

    TBoostingProgressTracker::TBoostingProgressTracker(const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                                       const NCatboostOptions::TOutputFilesOptions& outputFilesOptions,
                                                       bool forceCalcEvalMetricOnEveryIteration,
                                                       bool hasTest,
                                                       bool testHasTarget,
                                                       ui32 cpuApproxDim,
                                                       bool hasWeights,
                                                       TMaybe<ui32> learnAndTestCheckSum,
                                                       ITrainingCallbacks* trainingCallbacks,
                                                       const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor)
        : CatboostOptions(catBoostOptions)
        , OutputOptions(outputFilesOptions)
        , OutputFiles(outputFilesOptions, "")
        , Metrics(CreateGpuMetrics(catBoostOptions.MetricOptions, cpuApproxDim, hasWeights, evalMetricDescriptor))
        , ErrorTracker(CreateErrorTracker(catBoostOptions.BoostingOptions->OverfittingDetector, Metrics.at(0)->GetCpuMetric(), hasTest))
        , BestModelMinTreesTracker(CreateErrorTracker(catBoostOptions.BoostingOptions->OverfittingDetector, Metrics.at(0)->GetCpuMetric(), hasTest))
        , TrainingCallbacks(trainingCallbacks)
        , LearnToken(GetTrainModelLearnToken())
        , TestTokens(GetTrainModelTestTokens(hasTest ? 1 : 0))
        , ForceCalcEvalMetricOnEveryIteration(forceCalcEvalMetricOnEveryIteration)
        , HasTest(hasTest)
        , HasTestTarget(testHasTarget)
        , CpuApproxDim(cpuApproxDim)
        , ProfileInfo(catBoostOptions.BoostingOptions->IterationCount)
        , MetricDescriptions(GetMetricsDescription(GetCpuMetrics(Metrics)))
        , IsSkipOnTrainFlags(GetSkipMetricOnTrain(GetCpuMetrics(Metrics)))
        , IsSkipOnTestFlags(GetSkipMetricOnTest(testHasTarget, GetCpuMetrics(Metrics)))
        , CalcEvalMetricOnEveryIteration(forceCalcEvalMetricOnEveryIteration || ErrorTracker.IsActive())
        , HasWeights(hasWeights)
        , LearnAndTestQuantizedFeaturesCheckSum(learnAndTestCheckSum)
    {
        if (OutputOptions.AllowWriteFiles()) {
            InitializeFileLoggers(CatboostOptions,
                                  OutputFiles,
                                  GetCpuMetrics(Metrics),
                                  LearnToken,
                                  TestTokens,
                                  OutputOptions.GetMetricPeriod(),
                                  &Logger);
        }

        {
            NJson::TJsonValue options;
            CatboostOptions.Save(&options);
            CatBoostOptionsStr = ToString<NJson::TJsonValue>(options);
        }

        if (HasTest && IsSkipOnTestFlags[0]) {
            CATBOOST_WARNING_LOG << "Warning: Eval metric " << Metrics[0]->GetCpuMetric().GetDescription()
                << " needs Target data, but test dataset does not have it so it won't be calculated" << Endl;
        }
    }

    void TBoostingProgressTracker::OnFirstCall() {
        CB_ENSURE(FirstCall, "OnFirstCall is called twice");

        LastSnapshotTime = Now();

        AddConsoleLogger(
            LearnToken,
            TestTokens,
            /*hasTrain=*/true,
            OutputOptions.GetVerbosePeriod(),
            CatboostOptions.BoostingOptions->IterationCount,
            &Logger);
        FirstCall = false;
    }

    void TBoostingProgressTracker::FinishIteration() {
        const bool calcMetrics = ShouldCalcMetricOnIteration();

        ProfileInfo.FinishIteration();
        auto profileResults = ProfileInfo.GetProfileResults();

        History.TimeHistory.push_back(TTimeInfo(profileResults));

        constexpr size_t evalMetricIdx = 0;
        Log((int)Iteration,
            MetricDescriptions,
            History.LearnMetricsHistory,
            History.TestMetricsHistory,
            !IsSkipOnTestFlags[evalMetricIdx] ? TMaybe<double>(ErrorTracker.GetBestError()) : Nothing(),
            !IsSkipOnTestFlags[evalMetricIdx] ? TMaybe<int>(ErrorTracker.GetBestIteration()) : Nothing(),
            profileResults,
            LearnToken,
            TestTokens,
            calcMetrics,
            &Logger);

        ContinueTraining = TrainingCallbacks->IsContinueTraining(History);

        ++Iteration;
    }

    static inline double GetFinalErrorFromMetric(const IGpuMetric* metric, TMetricHolder&& holder) {
        if (dynamic_cast<const TGpuCustomMetric*>(metric)) {
            auto* customMetric = dynamic_cast<const TGpuCustomMetric*>(metric);
            return customMetric->GetFinalError(std::move(holder));
        } else {
            return metric->GetCpuMetric().GetFinalError(std::move(holder));
        }
    }

    void TBoostingProgressTracker::TrackLearnErrors(IMetricCalcer& metricCalcer) {
        History.LearnMetricsHistory.emplace_back();
        if (!ShouldCalcMetricOnIteration()) {
            return;
        }

        for (size_t i = 0; i < Metrics.size(); ++i) {
            if (!IsSkipOnTrainFlags[i]) {
                const auto& metric = Metrics[i].Get();
                auto metricValue = GetFinalErrorFromMetric(Metrics[i].Get(), metricCalcer.Compute(metric));
                History.AddLearnError(metric->GetCpuMetric(), metricValue);
            }
        }
    }

    void TBoostingProgressTracker::TrackTestErrors(IMetricCalcer& metricCalcer) {
        History.TestMetricsHistory.emplace_back(); // new iter

        const bool calcAllMetrics = ShouldCalcMetricOnIteration();
        const bool calcErrorTrackerMetric = calcAllMetrics || CalcEvalMetricOnEveryIteration;

        // Error tracker metric is first metric (explicitly set by option --eval-metric or loss function).
        // In case of changing the order it should be changed in CPU mode also.
        const int errorTrackerMetricIdx = calcErrorTrackerMetric ? 0 : -1;
        for (int i = 0; i < Metrics.ysize(); ++i) {
            if (!calcAllMetrics && (i != errorTrackerMetricIdx)) {
                continue;
            }
            if (IsSkipOnTestFlags[i]) {
                continue;
            }

            auto metricValue = GetFinalErrorFromMetric(Metrics[i].Get(), metricCalcer.Compute(Metrics[i].Get()));
            History.AddTestError(0 /*testIdx*/, Metrics[i]->GetCpuMetric(), metricValue, i == errorTrackerMetricIdx);

            if (i == errorTrackerMetricIdx) {
                ErrorTracker.AddError(metricValue, static_cast<int>(GetCurrentIteration()));
                if (OutputOptions.UseBestModel && static_cast<int>(GetCurrentIteration() + 1) >= OutputOptions.BestModelMinTrees) {
                    BestModelMinTreesTracker.AddError(metricValue, static_cast<int>(GetCurrentIteration()));
                }
            }
        }
    }

    void TBoostingProgressTracker::MaybeRestoreFromSnapshot(std::function<void(IInputStream*)> loader) {
        if (!HasSnapshot()) {
            return;
        }

        try {
            TProgressHelper(GpuProgressLabel()).CheckedLoad(OutputFiles.SnapshotFile, [&](TIFStream* in) {
                if (!TrainingCallbacks->OnLoadSnapshot(in)) {
                    return;
                }
                TString taskOptionsStr;
                ::Load(in, taskOptionsStr);
                const bool paramsCompatible = NCatboostOptions::IsParamsCompatible(CatBoostOptionsStr, taskOptionsStr);
                CB_ENSURE(paramsCompatible, "Saved model's params are different from current model's params");
                ::Load(in, History);

                TProfileInfoData profileData;
                ::Load(in, profileData);
                ProfileInfo.InitProfileInfo(std::move(profileData));

                TMaybe<ui32> learnAndTestCheckSum;
                ::Load(in, learnAndTestCheckSum);
                if (learnAndTestCheckSum.Defined() && LearnAndTestQuantizedFeaturesCheckSum.Defined()) {
                    CB_ENSURE(
                        learnAndTestCheckSum.GetRef() == LearnAndTestQuantizedFeaturesCheckSum.GetRef(),
                        "Saved model's learn and test checksum is different from current model's learn and test checksum");
                }

                loader(in);
            });
        } catch (const TCatBoostException&) {
            throw;
        } catch (...) {
            CATBOOST_WARNING_LOG << "Can't load progress from snapshot file: " << OutputFiles.SnapshotFile << " exception: "
                                 << CurrentExceptionMessage() << Endl;
            return;
        }

        Iteration = History.TimeHistory.size();

        // WriteHistory & update ErrorTracker
        for (ui64 iteration = 0; iteration < Iteration; ++iteration) {
            const int testIdxToLog = 0;
            if (ShouldCalcMetricOnIteration(iteration) && iteration < History.TestMetricsHistory.size()) {
                if (IsSkipOnTestFlags[testIdxToLog]) {
                    continue;
                }

                const int metricIdxToLog = 0;
                const TString& metricDescription = Metrics[metricIdxToLog]->GetCpuMetric().GetDescription();
                const double error = History.TestMetricsHistory[iteration][testIdxToLog].at(metricDescription);
                ErrorTracker.AddError(error, static_cast<int>(iteration));
                if (OutputOptions.UseBestModel && static_cast<int>(iteration + 1) >= OutputOptions.BestModelMinTrees) {
                    BestModelMinTreesTracker.AddError(error, static_cast<int>(iteration));
                }
            }

            Log(
                (int)iteration,
                MetricDescriptions,
                History.LearnMetricsHistory,
                History.TestMetricsHistory,
                !IsSkipOnTestFlags[testIdxToLog] ? TMaybe<double>(ErrorTracker.GetBestError()) : Nothing(),
                !IsSkipOnTestFlags[testIdxToLog] ? TMaybe<int>(ErrorTracker.GetBestIteration()) : Nothing(),
                TProfileResults(History.TimeHistory[iteration].PassedTime, History.TimeHistory[iteration].RemainingTime),
                LearnToken,
                TestTokens,
                /*outputErrors*/ ShouldCalcMetricOnIteration(iteration),
                &Logger);
        }
    }

    void TBoostingProgressTracker::MaybeSaveSnapshot(std::function<void(IOutputStream*)> saver) {
        if (IsTimeToSaveSnapshot()) {
            const auto snapshotBackup = OutputFiles.SnapshotFile + ".bak";
            TProgressHelper(GpuProgressLabel()).Write(snapshotBackup, [&](IOutputStream* out) {
                NJson::TJsonArray processors;
                const auto& devicesProps = NCudaLib::NCudaHelpers::GetDevicesProps();
                for (const auto& props : devicesProps) {
                    processors.AppendValue(props.GetName());
                }
                TrainingCallbacks->OnSaveSnapshot(processors, out);
                ::Save(out, CatBoostOptionsStr);
                ::Save(out, History);
                ::Save(out, ProfileInfo.DumpProfileInfo());
                ::Save(out, LearnAndTestQuantizedFeaturesCheckSum);
                saver(out);
            });
            TFsPath(snapshotBackup).ForceRenameTo(OutputFiles.SnapshotFile);
            LastSnapshotTime = Now();
        }
    }
}
