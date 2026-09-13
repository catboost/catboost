#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/private/libs/options/catboost_options.h>
#include <catboost/private/libs/options/output_file_options.h>

#include <util/generic/ptr.h>

class TFullModel;

namespace NCB {
    class TEvalResult;

    // Host progress management for the Metal tree learner. Metrics, logging and
    // overfitting detection use the same components as the existing trainers.
    class TMetalTrainingProgress {
    public:
        TMetalTrainingProgress(
            const NCatboostOptions::TCatBoostOptions& options,
            const NCatboostOptions::TOutputFilesOptions& outputOptions,
            const TTrainingDataProviders& data,
            float bias,
            NPar::ILocalExecutor* executor,
            const TFullModel* initModel = nullptr,
            const TDataProviders* initModelApplyCompatiblePools = nullptr,
            bool forceCalcEvalMetricOnEveryIteration = false,
            const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor = Nothing(),
            ui32 approxDimension = 1,
            TConstArrayRef<ui32> baselineColumns = {});
        ~TMetalTrainingProgress();

        void StartIteration();
        // The optional GPU cursor is row-major: [object][approximation dimension].
        bool OnIteration(
            ui32 iteration,
            const TFullModel& singleTreeModel,
            TConstArrayRef<float> learnCursor = {});
        bool ReplayIteration(
            ui32 iteration,
            const TFullModel& singleTreeModel,
            const TMetricsAndTimeLeftHistory& restoredHistory);
        const TMetricsAndTimeLeftHistory& GetHistory() const;
        void RestoreTimeHistory(const TVector<TTimeInfo>& timeHistory);
        void Finish(TFullModel* model, const TVector<TEvalResult*>& evalResult);

    private:
        class TImpl;
        THolder<TImpl> Impl;
    };
}
