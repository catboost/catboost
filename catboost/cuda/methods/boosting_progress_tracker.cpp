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
        , TestTokens(GetTrainModelTestTokens(1))
        , HasTest(hasTest)
        , ProfileInfo(catBoostOptions.BoostingOptions->IterationCount)
        , MetricDescriptions(GetMetricsDescription(GetCpuMetrics(Metrics)))
        , IsSkipOnTrainFlags(GetSkipMetricOnTrain(GetCpuMetrics(Metrics))) {

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
                                      &Logger
                );
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
                &Logger
        );
        FirstCall = false;
    }

    void TBoostingProgressTracker::FinishIteration() {
        const bool skipMetrics = ShouldCalcMetricOnIteration();

        ProfileInfo.FinishIteration();
        History.TimeHistory.push_back({ProfileInfo.GetProfileResults().PassedTime,
                                       ProfileInfo.GetProfileResults().RemainingTime});

        Log(MetricDescriptions,
            IsSkipOnTrainFlags,
            History.LearnMetricsHistory,
            History.TestMetricsHistory,
            ErrorTracker.GetBestError(),
            ErrorTracker.GetBestIteration(),
            ProfileInfo.GetProfileResults(),
            LearnToken,
            TestTokens,
            skipMetrics,
            &Logger
        );

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

        if (!ShouldCalcMetricOnIteration()) {
            return;
        }

        for (size_t i = 0; i < Metrics.size(); ++i) {
            auto metricValue = Metrics[i]->GetCpuMetric().GetFinalError(metricCalcer.Compute(Metrics[i].Get()));
            History.TestMetricsHistory.back()[0].push_back(metricValue);

            if (i == 0) {
                ErrorTracker.AddError(metricValue, static_cast<int>(GetCurrentIteration()));
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

        WriteHistory(MetricDescriptions,
                     History,
                     LearnToken,
                     TestTokens,
                     &Logger
        );

        auto testMetricHistory = History.TestMetricsHistory;

        for (Iteration = 0; Iteration < testMetricHistory.size(); ++Iteration) {
            const int testIdxToLog = 0;
            const int metricIdxToLog = 0;
            if (ShouldCalcMetricOnIteration()) {
                ErrorTracker.AddError(testMetricHistory[Iteration][testIdxToLog][metricIdxToLog],
                                      static_cast<int>(Iteration));
            }
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
