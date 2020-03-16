#pragma once

#include "enums.h"

#include <util/generic/fwd.h>

TConstArrayRef<ELossFunction> GetAllObjectives();

// metric type (regression, multi-regression, classification(bin, multi), ranking(pair, group))
bool IsRegressionMetric(ELossFunction lossFunction);
bool IsMultiRegressionMetric(ELossFunction loss);

bool IsClassificationMetric(ELossFunction lossFunction);
bool IsBinaryClassCompatibleMetric(ELossFunction lossFunction);
bool IsMultiClassCompatibleMetric(ELossFunction lossFunction);
bool IsMultiClassCompatibleMetric(TStringBuf lossDescription);
bool IsClassificationOnlyMetric(ELossFunction lossFunction);
bool IsBinaryClassOnlyMetric(ELossFunction lossFunction);
bool IsMultiClassOnlyMetric(ELossFunction lossFunction);

bool IsRankingMetric(ELossFunction lossFunction);
bool IsGroupwiseMetric(ELossFunction lossFunction);
bool IsGroupwiseMetric(TStringBuf metricName);
bool IsPairwiseMetric(ELossFunction lossFunction);
bool IsPairwiseMetric(TStringBuf lossFunction);
ERankingType GetRankingType(ELossFunction loss);

// objective type
bool IsClassificationObjective(ELossFunction lossFunction);
bool IsClassificationObjective(TStringBuf lossDescription);
bool IsRegressionObjective(ELossFunction lossFunction);
bool IsRegressionObjective(TStringBuf lossDescription);
bool IsMultiRegressionObjective(ELossFunction loss);
bool IsMultiRegressionObjective(TStringBuf loss);

// various
bool UsesPairsForCalculation(ELossFunction lossFunction);

bool IsPlainMode(EBoostingType boostingType);

bool IsPlainOnlyModeLoss(ELossFunction lossFunction);

bool IsPairwiseScoring(ELossFunction lossFunction);

bool IsGpuPlainDocParallelOnlyMode(ELossFunction lossFunction);

bool IsYetiRankLossFunction(ELossFunction lossFunction);

bool IsPairLogit(ELossFunction lossFunction);

bool IsSecondOrderScoreFunction(EScoreFunction scoreFunction);

bool AreZeroWeightsAfterBootstrap(EBootstrapType type);

bool ShouldSkipCalcOnTrainByDefault(ELossFunction lossFunction);

bool IsUserDefined(ELossFunction lossFunction);

bool IsEmbeddingFeatureEstimator(EFeatureCalcerType estimatorType);

bool IsBuildingFullBinaryTree(EGrowPolicy growPolicy);

bool IsPlainOnlyModeScoreFunction(EScoreFunction scoreFunction);

bool ShouldBinarizeLabel(ELossFunction lossFunction);

bool IsCvStratifiedObjective(TStringBuf lossDescription);

EFstrType AdjustFeatureImportanceType(EFstrType type, ELossFunction lossFunction);
EFstrType AdjustFeatureImportanceType(EFstrType type, TStringBuf lossDescription);

bool IsInternalFeatureImportanceType(EFstrType type);
