#pragma once

#include "boosting_metric_calcer.h"
#include "serialization_helper.h"

#include <catboost/libs/overfitting_detector/overfitting_detector.h>
#include <catboost/cuda/targets/target_func.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/models/additive_model.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset.h>
#include <catboost/cuda/gpu_data/feature_parallel_dataset_builder.h>
#include <catboost/libs/overfitting_detector/error_tracker.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/libs/options/output_file_options.h>
#include <catboost/libs/loggers/logger.h>
#include <catboost/libs/options/catboost_options.h>
#include <catboost/libs/helpers/progress_helper.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>

#include <util/generic/maybe.h>


namespace NCatboostCuda {
    class TBoostingProgressTracker {
    public:
        TBoostingProgressTracker(const NCatboostOptions::TCatBoostOptions& catBoostOptions,
                                 const NCatboostOptions::TOutputFilesOptions& outputFilesOptions,
                                 bool hasTest,
                                 ui32 cpuApproxDim);

        const TErrorTracker& GetErrorTracker() const {
            return ErrorTracker;
        }

        const TErrorTracker& GetBestModelMinTreesTracker() const {
            return BestModelMinTreesTracker;
        }

        const TVector<TVector<double>>& GetBestTestCursor() const {
            return BestTestCursor;
        }

        bool NeedBestTestCursor() const {
            return HasTest && !OutputOptions.CreateEvalFullPath().empty();
        }

        size_t GetCurrentIteration() const {
            return History.TimeHistory.size();
        }

        bool ShouldStop() const {
            return (Iteration >= CatboostOptions.BoostingOptions->IterationCount) || ErrorTracker.GetIsNeedStop();
        }

        bool IsBestIteration() const {
            return static_cast<size_t>(ErrorTracker.GetBestIteration()) == GetCurrentIteration();
        }

        void SetBestTestCursor(const TVector<TVector<double>>& bestTestCursor) {
            BestTestCursor = bestTestCursor;
        }

        void MaybeRestoreFromSnapshot(std::function<void(IInputStream*)> loader);

        void MaybeSaveSnapshot(std::function<void(IOutputStream*)> saver);

        const TMetricsAndTimeLeftHistory& GetMetricsAndTimeLeftHistory() const {
            return this->History;
        }

    private:
        void OnFirstCall();

        bool IsTimeToSaveSnapshot() const {
            return OutputOptions.SaveSnapshot() &&
                ((Now() - LastSnapshotTime).SecondsFloat() > OutputOptions.GetSnapshotSaveInterval()
                || ShouldStop());
        }

        bool HasSnapshot() const {
            return OutputOptions.SaveSnapshot() && NFs::Exists(OutputFiles.SnapshotFile);
        }

        void StartIteration() {
            ProfileInfo.StartNextIteration();
        }

        void FinishIteration();

        void TrackLearnErrors(IMetricCalcer& metricCalcer);

        void TrackTestErrors(IMetricCalcer& metricCalcer);

        bool ShouldCalcMetricOnIteration(TMaybe<size_t> iteration = Nothing()) const {
            if (!iteration.Defined()) {
                iteration = GetCurrentIteration();
            }
            return (*iteration) % OutputOptions.GetMetricPeriod() == 0 || (*iteration) == CatboostOptions.BoostingOptions->IterationCount - 1;
        }

        friend class TOneIterationProgressTracker;

    private:
        const NCatboostOptions::TCatBoostOptions& CatboostOptions;
        const NCatboostOptions::TOutputFilesOptions& OutputOptions;

        TOutputFiles OutputFiles;

        TString CatBoostOptionsStr;
        TMetricsAndTimeLeftHistory History;
        TLogger Logger;

        TVector<THolder<IGpuMetric>> Metrics;
        TErrorTracker ErrorTracker;
        TErrorTracker BestModelMinTreesTracker;

        TString LearnToken;
        TVector<const TString> TestTokens;
        bool HasTest = false;
        TProfileInfo ProfileInfo;

        TVector<TString> MetricDescriptions;
        TVector<bool> IsSkipOnTrainFlags;
        TVector<TVector<double>> BestTestCursor;

        size_t Iteration = 0;
        TInstant LastSnapshotTime = Now();
        bool FirstCall = true;
    };

    class TOneIterationProgressTracker {
    public:
        TOneIterationProgressTracker(TBoostingProgressTracker& progressLogger)
            : Owner(progressLogger)
        {
            if (Owner.FirstCall) {
                Owner.OnFirstCall();
            }
            Owner.StartIteration();
        }

        ~TOneIterationProgressTracker() {
            Owner.FinishIteration();
        }

        void TrackLearnErrors(IMetricCalcer& metricCalcer) {
            CB_ENSURE(!LearnWasTracked, "Can't track learn more than once");
            Owner.TrackLearnErrors(metricCalcer);
            LearnWasTracked = true;
        }

        size_t GetCurrentIteration() const {
            return Owner.GetCurrentIteration();
        }

        bool IsBestTestIteration() const {
            CB_ENSURE(!Owner.HasTest || TestWasTracked, "Error: track test errors first");
            return Owner.HasTest && Owner.IsBestIteration();
        }

        template <class TSaver>
        void MaybeSaveSnapshot(TSaver&& saver) {
            Owner.MaybeSaveSnapshot(std::forward<TSaver>(saver));
        }

        void TrackTestErrors(IMetricCalcer& metricCalcer) {
            CB_ENSURE(!TestWasTracked, "Can't track test more than once");
            TestWasTracked = true;
            Owner.TrackTestErrors(metricCalcer);
        }

    private:
        TBoostingProgressTracker& Owner;
        bool LearnWasTracked = false;
        bool TestWasTracked = false;
    };

}
