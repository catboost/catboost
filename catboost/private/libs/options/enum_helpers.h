#pragma once

#include "enums.h"

#include <util/generic/fwd.h>

TConstArrayRef<ELossFunction> GetAllObjectives();

// metric type (regression, multi-regression, classification(bin, multi, multilabel), ranking(pair, group))
bool IsRegressionMetric(ELossFunction lossFunction);
bool IsMultiRegressionMetric(ELossFunction loss);
bool IsSurvivalRegressionMetric(ELossFunction loss);
bool IsMultiLabelMetric(ELossFunction loss);
bool IsMultiLabelOnlyMetric(ELossFunction loss);
bool IsMultiTargetMetric(ELossFunction loss);
bool IsMultiTargetOnlyMetric(ELossFunction loss);

bool IsClassificationMetric(ELossFunction lossFunction);
bool IsBinaryClassCompatibleMetric(ELossFunction lossFunction);
bool IsMultiClassCompatibleMetric(ELossFunction lossFunction);
bool IsMultiClassCompatibleMetric(TStringBuf lossDescription);
bool IsClassificationOnlyMetric(ELossFunction lossFunction);
bool IsBinaryClassOnlyMetric(ELossFunction lossFunction);
bool IsMultiClassOnlyMetric(ELossFunction lossFunction);

bool IsRankingMetric(ELossFunction lossFunction);
bool IsRankingMetric(TStringBuf metricName);
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
bool IsMultiTargetObjective(ELossFunction loss);
bool IsMultiTargetObjective(TStringBuf loss);
bool IsSurvivalRegressionObjective(ELossFunction loss);
bool IsSurvivalRegressionObjective(TStringBuf loss);
bool IsMultiLabelObjective(ELossFunction lossFunction);
bool IsMultiLabelObjective(TStringBuf lossFunction);

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
bool IsUserDefined(TStringBuf metricName);

bool HasGpuImplementation(ELossFunction loss);
bool HasGpuImplementation(TStringBuf metricName);

bool IsEmbeddingFeatureEstimator(EFeatureCalcerType estimatorType);
bool IsClassificationOnlyEstimator(EFeatureCalcerType estimatorType);

bool IsBuildingFullBinaryTree(EGrowPolicy growPolicy);

bool IsPlainOnlyModeScoreFunction(EScoreFunction scoreFunction);

bool ShouldBinarizeLabel(ELossFunction lossFunction);

bool IsCvStratifiedObjective(TStringBuf lossDescription);

EFstrType AdjustFeatureImportanceType(EFstrType type, ELossFunction lossFunction);
EFstrType AdjustFeatureImportanceType(EFstrType type, TStringBuf lossDescription);

bool IsInternalFeatureImportanceType(EFstrType type);

bool IsUncertaintyPredictionType(EPredictionType type);

EEstimatedSourceFeatureType FeatureTypeToEstimatedSourceFeatureType(EFeatureType featureType);

EFeatureType EstimatedSourceFeatureTypeToFeatureType(EEstimatedSourceFeatureType featureType);
