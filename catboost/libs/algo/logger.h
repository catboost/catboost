#pragma once

#include "learn_context.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/tensorboard_logger/tensorboard_logger.h>

struct TLogger {
    THolder<TOFStream> LearnErrLog, TestErrLog;
    THolder<TTensorBoardLogger> LearnTensorBoardLogger, TestTensorBoardLogger;
};

enum class EPhase {
    Learn,
    Test
};

THolder<TLogger> CreateLogger(const TVector<THolder<IMetric>>& errors, TLearnContext& ctx, const bool hasTest);
void Log(int iteration, const TVector<double>& errorsHistory, const TVector<THolder<IMetric>>& errors, TLogger* logger, const EPhase phase);
