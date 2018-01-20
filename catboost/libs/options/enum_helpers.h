#pragma once

#include "enums.h"
#include <util/generic/string.h>

bool IsSupportedOnGpu(ELossFunction lossFunction);

bool IsClassificationLoss(ELossFunction lossFunction);

bool IsClassificationLoss(const TString& lossFunction);

bool IsMultiClassError(ELossFunction lossFunction);

bool IsPairwiseError(ELossFunction lossFunction);

bool IsQuerywiseError(ELossFunction lossFunction);
