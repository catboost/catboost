#pragma once

#include "enums.h"

#include <util/generic/fwd.h>


TConstArrayRef<ELossFunction> GetAllObjectives();

bool IsSingleDimensionalCompatibleError(ELossFunction lossFunction);

bool IsMultiDimensionalCompatibleError(ELossFunction lossFunction);

bool IsForCrossEntropyOptimization(ELossFunction lossFunction);

bool IsForOrderOptimization(ELossFunction lossFunction);

bool IsForAbsoluteValueOptimization(ELossFunction lossFunction);

bool IsOnlyForCrossEntropyOptimization(ELossFunction lossFunction);

bool IsClassificationOnlyMetric(ELossFunction lossFunction);

bool IsBinaryClassCompatibleMetric(ELossFunction lossFunction);
bool IsMultiClassCompatibleMetric(ELossFunction lossFunction);

// some metrics are both binclass and multiclass (e.g. HingeLoss)
bool IsBinaryClassOnlyMetric(ELossFunction lossFunction);
bool IsMultiClassOnlyMetric(ELossFunction lossFunction);

bool IsClassificationObjective(ELossFunction lossFunction);

bool IsClassificationObjective(TStringBuf lossDescription);

bool IsRegressionObjective(ELossFunction lossFunction);

bool IsRegressionObjective(TStringBuf lossDescription);

bool IsRegressionMetric(ELossFunction lossFunction);

bool IsGroupwiseMetric(ELossFunction lossFunction);

bool IsPairwiseMetric(ELossFunction lossFunction);

bool UsesPairsForCalculation(ELossFunction lossFunction);

bool IsPlainMode(EBoostingType boostingType);

bool IsPlainOnlyModeLoss(ELossFunction lossFunction);

bool IsPairwiseScoring(ELossFunction lossFunction);

bool IsGpuPlainDocParallelOnlyMode(ELossFunction lossFunction);

bool ShouldGenerateYetiRankPairs(ELossFunction lossFunction);

bool IsPairLogit(ELossFunction lossFunction);

bool IsSecondOrderScoreFunction(EScoreFunction scoreFunction);

bool AreZeroWeightsAfterBootstrap(EBootstrapType type);

bool ShouldSkipCalcOnTrainByDefault(ELossFunction lossFunction);

bool IsUserDefined(ELossFunction lossFunction);

bool ShouldSkipFstrGrowPolicy(EGrowPolicy growPolicy);
