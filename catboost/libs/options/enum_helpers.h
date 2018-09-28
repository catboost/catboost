#pragma once

#include "enums.h"
#include <util/generic/string.h>

bool IsSingleDimensionalError(ELossFunction lossFunction);

bool IsMultiDimensionalError(ELossFunction lossFunction);

bool IsForCrossEntropyOptimization(ELossFunction lossFunction);

bool IsForOrderOptimization(ELossFunction lossFunction);

bool IsForAbsoluteValueOptimization(ELossFunction lossFunction);

bool IsOnlyForCrossEntropyOptimization(ELossFunction lossFunction);

bool IsBinaryClassError(ELossFunction lossFunction);

bool IsClassificationLoss(ELossFunction lossFunction);

bool IsClassificationLoss(const TString& lossDescription);

bool IsRegressionLoss(ELossFunction lossFunction);

bool IsRegressionLoss(const TString& lossDescription);

bool IsMultiClassError(ELossFunction lossFunction);

bool IsQuerywiseError(ELossFunction lossFunction);

bool IsPairwiseError(ELossFunction lossFunction);

bool IsPlainMode(EBoostingType boostingType);

bool IsPlainOnlyModeLoss(ELossFunction lossFunction);

bool IsPairwiseScoring(ELossFunction lossFunction);

bool IsGpuPlainDocParallelOnlyMode(ELossFunction lossFunction);

bool ShouldGenerateYetiRankPairs(ELossFunction lossFunction);

bool IsPairLogit(ELossFunction lossFunction);

bool IsSecondOrderScoreFunction(EScoreFunction scoreFunction);

bool AreZeroWeightsAfterBootstrap(EBootstrapType type);

bool ShouldSkipCalcOnTrainByDefault(ELossFunction lossFunction);

