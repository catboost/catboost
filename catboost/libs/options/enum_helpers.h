#pragma once

#include "enums.h"
#include <util/generic/string.h>

bool IsClassificationLoss(ELossFunction lossFunction);

bool IsClassificationLoss(const TString& lossDescription);

bool IsMultiClassError(ELossFunction lossFunction);

bool IsQuerywiseError(ELossFunction lossFunction);

bool IsPairwiseError(ELossFunction lossFunction);

bool IsPlainMode(EBoostingType boostingType);

bool IsSecondOrderScoreFunction(EScoreFunction scoreFunction);

bool AreZeroWeightsAfterBootstrap(EBootstrapType type);

