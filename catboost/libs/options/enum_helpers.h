#pragma once

#include "enums.h"
#include <util/generic/string.h>

bool IsOnlyForCrossEntropyOptimization(ELossFunction lossFunction);

bool IsBinaryClassError(ELossFunction lossFunction);

bool IsClassificationLoss(ELossFunction lossFunction);

bool IsClassificationLoss(const TString& lossDescription);

bool IsMultiClassError(ELossFunction lossFunction);

bool IsQuerywiseError(ELossFunction lossFunction);

bool IsPairwiseError(ELossFunction lossFunction);

bool IsPlainMode(EBoostingType boostingType);

bool IsPlainOnlyModeLoss(ELossFunction lossFunction);

bool IsPairwiseScoring(ELossFunction lossFunction);

bool IsGpuDocParallelOnlyMode(ELossFunction lossFunction);

bool IsItNecessaryToGeneratePairs(ELossFunction lossFunction);

bool IsSecondOrderScoreFunction(EScoreFunction scoreFunction);

bool AreZeroWeightsAfterBootstrap(EBootstrapType type);

