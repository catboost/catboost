#pragma once

#include "learn_context.h"
#include "metric.h"

#include <catboost/libs/tensorboard_logger/tensorboard_logger.h>

struct TLogger {
    THolder<TOFStream> LearnErrLog, TestErrLog;
    THolder<TTensorBoardLogger> LearnTensorBoardLogger, TestTensorBoardLogger;
};

enum class EPhase {
    Learn,
    Test
};

THolder<TLogger> CreateLogger(const yvector<THolder<IMetric>>& errors, TLearnContext& ctx, const bool hasTest);
void Log(int iteration, const yvector<double>& errorsHistory, const yvector<THolder<IMetric>>& errors, TLogger* logger, const EPhase phase);
