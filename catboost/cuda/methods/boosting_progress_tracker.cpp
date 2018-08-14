#include "boosting_progress_tracker.h"

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
                                                       bool hasTest)
        : CatboostOptions(catBoostOptions)
        , OutputOptions(outputFilesOptions)
        , OutputFiles(outputFilesOptions, "")
        , Metrics(CreateGpuMetrics(catBoostOptions.LossFunctionDescription,
                                   catBoostOptions.MetricOptions))
        , ErrorTracker(CreateErrorTracker(catBoostOptions.BoostingOptions->OverfittingDetector, Metrics.at(0)->GetCpuMetric(), hasTest))
        , LearnToken(GetTrainModelLearnToken())
        , TestTokens(GetTrainModelTestTokens(hasTest ? 1 : 0))
        , HasTest(hasTest)
        , ProfileInfo(catBoostOptions.BoostingOptions->IterationCount)
        , MetricDescriptions(GetMetricsDescription(GetCpuMetrics(Metrics)))
        , IsSkipOnTrainFlags(GetSkipMetricOnTrain(GetCpuMetrics(Metrics)))
    {
        if (OutputOptions.AllowWriteFiles()) {
            CreateMetaFile(OutputFiles,
                           OutputOptions,
                           GetCpuMetrics(Metrics),
                           CatboostOptions.BoostingOptions->IterationCount);

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
    }

    void TBoostingProgressTracker::OnFirstCall() {
        Y_VERIFY(FirstCall);

        LastSnapshotTime = Now();

        AddConsoleLogger(
            LearnToken,
            TestTokens,
            /*hasTrain=*/true,
            OutputOptions.GetMetricPeriod(),
            CatboostOptions.BoostingOptions->IterationCount,
            &Logger);
        FirstCall = false;
    }

    void TBoostingProgressTracker::FinishIteration() {
        const bool calcMetrics = ShouldCalcMetricOnIteration();

        ProfileInfo.FinishIteration();
        History.TimeHistory.push_back({ProfileInfo.GetProfileResults().PassedTime,
                                       ProfileInfo.GetProfileResults().RemainingTime});

        Log((int)Iteration,
            MetricDescriptions,
            IsSkipOnTrainFlags,
            History.LearnMetricsHistory,
            History.TestMetricsHistory,
            ErrorTracker.GetBestError(),
            ErrorTracker.GetBestIteration(),
            ProfileInfo.GetProfileResults(),
            LearnToken,
            TestTokens,
            calcMetrics,
            &Logger);

        ++Iteration;
    }

    void TBoostingProgressTracker::TrackLearnErrors(IMetricCalcer& metricCalcer) {
        History.LearnMetricsHistory.emplace_back();
        if (!ShouldCalcMetricOnIteration()) {
            return;
        }

        for (size_t i = 0; i < Metrics.size(); ++i) {
            if (!IsSkipOnTrainFlags[i]) {
                auto metricValue = Metrics[i]->GetCpuMetric().GetFinalError(metricCalcer.Compute(Metrics[i].Get()));
                History.LearnMetricsHistory.back().push_back(metricValue);
            }
        }
    }

    void TBoostingProgressTracker::TrackTestErrors(IMetricCalcer& metricCalcer) {
        History.TestMetricsHistory.emplace_back().emplace_back();

        const bool calcAllMetrics = ShouldCalcMetricOnIteration();
        const bool calcErrorTrackerMetric = calcAllMetrics || ErrorTracker.IsActive();

        // Error tracker metric is first metric (explicitly set by option --eval-metric or loss function).
        // In case of changing the order it should be changed in CPU mode also.
        const int errorTrackerMetricIdx = calcErrorTrackerMetric ? 0 : -1;
        for (size_t i = 0; i < Metrics.size(); ++i) {
            if (calcAllMetrics || i == errorTrackerMetricIdx) {
                auto metricValue = Metrics[i]->GetCpuMetric().GetFinalError(metricCalcer.Compute(Metrics[i].Get()));
                History.TestMetricsHistory.back()[0].push_back(metricValue);

                if (i == errorTrackerMetricIdx) {
                    ErrorTracker.AddError(metricValue, static_cast<int>(GetCurrentIteration()));
                }
            }
        }
    }

    void TBoostingProgressTracker::MaybeRestoreFromSnapshot(std::function<void(IInputStream*)> loader) {
        if (!HasSnapshot()) {
            return;
        }
        if (GetFileLength(OutputFiles.SnapshotFile) == 0) {
            MATRIXNET_DEBUG_LOG << "Empty snapshot file: can't restore from progress" << Endl;
            return;
        }

        TProgressHelper(GpuProgressLabel()).CheckedLoad(OutputFiles.SnapshotFile, [&](TIFStream* in) {
            TString taskOptionsStr;
            ::Load(in, taskOptionsStr);
            ::Load(in, History);

            TProfileInfoData profileData;
            ::Load(in, profileData);
            ProfileInfo.InitProfileInfo(std::move(profileData));

            loader(in);
        });

        auto testMetricHistory = History.TestMetricsHistory;
        const TVector<TTimeInfo>& timeHistory = History.TimeHistory;


        Iteration = History.TimeHistory.size();

        // WriteHistory & update ErrorTracker
        for (ui64 iteration = 0; iteration < Iteration; ++iteration) {
            if (ShouldCalcMetricOnIteration(iteration)) {
                const int testIdxToLog = 0;
                const int metricIdxToLog = 0;
                ErrorTracker.AddError(testMetricHistory[iteration][testIdxToLog][metricIdxToLog],
                                      static_cast<int>(iteration));
            }

            Log(
                (int)iteration,
                MetricDescriptions,
                IsSkipOnTrainFlags,
                History.LearnMetricsHistory,
                History.TestMetricsHistory,
                ErrorTracker.GetBestError(),
                ErrorTracker.GetBestIteration(),
                TProfileResults(timeHistory[iteration].PassedTime, timeHistory[iteration].RemainingTime),
                LearnToken,
                TestTokens,
                /*outputErrors*/ShouldCalcMetricOnIteration(iteration),
                &Logger
            );
        }
    }

    void TBoostingProgressTracker::MaybeSaveSnapshot(std::function<void(IOutputStream*)> saver) {
        if (IsTimeToSaveSnapshot()) {
            TProgressHelper(GpuProgressLabel()).Write(OutputFiles.SnapshotFile, [&](IOutputStream* out) {
                ::Save(out, CatBoostOptionsStr);
                ::Save(out, History);
                ::Save(out, ProfileInfo.DumpProfileInfo());
                saver(out);
            });
            LastSnapshotTime = Now();
        }
    }
}
