#include "enum_helpers.h"
#include "loss_description.h"

#include <util/generic/array_ref.h>
#include <util/generic/is_in.h>
#include <util/generic/strbuf.h>
#include <util/generic/vector.h>
#include <util/string/cast.h>

TConstArrayRef<ELossFunction> GetAllObjectives() {
    static TVector<ELossFunction> allObjectives = {
        ELossFunction::Logloss, ELossFunction::CrossEntropy, ELossFunction::RMSE, ELossFunction::MAE,
        ELossFunction::Quantile, ELossFunction::LogLinQuantile, ELossFunction::Expectile,
        ELossFunction::MAPE, ELossFunction::Poisson, ELossFunction::MultiClass,
        ELossFunction::MultiClassOneVsAll, ELossFunction::PairLogit, ELossFunction::PairLogitPairwise,
        ELossFunction::YetiRank, ELossFunction::YetiRankPairwise, ELossFunction::QueryRMSE,
        ELossFunction::QuerySoftMax, ELossFunction::QueryCrossEntropy, ELossFunction::Lq,
        ELossFunction::Huber, ELossFunction::StochasticFilter, ELossFunction::UserPerObjMetric,
        ELossFunction::UserQuerywiseMetric
    };
    return allObjectives;
}

bool IsSingleDimensionalCompatibleError(ELossFunction lossFunction) {
    return (lossFunction != ELossFunction::MultiClass &&
            lossFunction != ELossFunction::MultiClassOneVsAll);
}

bool IsMultiDimensionalCompatibleError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll ||
            lossFunction == ELossFunction::Precision ||
            lossFunction == ELossFunction::Recall ||
            lossFunction == ELossFunction::F1 ||
            lossFunction == ELossFunction::TotalF1 ||
            lossFunction == ELossFunction::MCC ||
            lossFunction == ELossFunction::Accuracy ||
            lossFunction == ELossFunction::AUC ||
            lossFunction == ELossFunction::HingeLoss ||
            lossFunction == ELossFunction::HammingLoss ||
            lossFunction == ELossFunction::ZeroOneLoss ||
            lossFunction == ELossFunction::Kappa ||
            lossFunction == ELossFunction::WKappa);
}

bool IsForCrossEntropyOptimization(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::Logloss ||  // binary classification metrics
            lossFunction == ELossFunction::CrossEntropy ||
            lossFunction == ELossFunction::Precision ||
            lossFunction == ELossFunction::Recall ||
            lossFunction == ELossFunction::F1 ||
            lossFunction == ELossFunction::MCC ||
            lossFunction == ELossFunction::BalancedAccuracy ||
            lossFunction == ELossFunction::BalancedErrorRate ||
            lossFunction == ELossFunction::Accuracy ||
            lossFunction == ELossFunction::CtrFactor ||
            lossFunction == ELossFunction::BrierScore ||
            lossFunction == ELossFunction::HingeLoss ||
            lossFunction == ELossFunction::HammingLoss ||
            lossFunction == ELossFunction::ZeroOneLoss ||
            lossFunction == ELossFunction::Kappa ||
            lossFunction == ELossFunction::WKappa ||
            lossFunction == ELossFunction::LogLikelihoodOfPrediction ||
            lossFunction == ELossFunction::MultiClass ||  // multiclassification metrics
            lossFunction == ELossFunction::MultiClassOneVsAll ||
            lossFunction == ELossFunction::TotalF1 ||
            lossFunction == ELossFunction::PairLogit ||  // ranking metrics
            lossFunction == ELossFunction::PairLogitPairwise ||
            lossFunction == ELossFunction::PairAccuracy ||
            lossFunction == ELossFunction::QuerySoftMax ||
            lossFunction == ELossFunction::QueryCrossEntropy);
}

bool IsGroupwiseOrderMetric(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::YetiRank ||
            lossFunction == ELossFunction::PrecisionAt ||
            lossFunction == ELossFunction::RecallAt ||
            lossFunction == ELossFunction::MAP ||
            lossFunction == ELossFunction::YetiRankPairwise ||
            lossFunction == ELossFunction::PFound ||
            lossFunction == ELossFunction::NDCG ||
            lossFunction == ELossFunction::AverageGain ||
            lossFunction == ELossFunction::QueryAverage ||
            lossFunction == ELossFunction::StochasticFilter);
}

bool IsForOrderOptimization(ELossFunction lossFunction) {
    return lossFunction == ELossFunction::AUC || IsGroupwiseOrderMetric(lossFunction);
}

bool IsForAbsoluteValueOptimization(ELossFunction lossFunction) {
    return  !IsForOrderOptimization(lossFunction) && !IsForCrossEntropyOptimization(lossFunction);
}

bool IsOnlyForCrossEntropyOptimization(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::BalancedErrorRate ||
            lossFunction == ELossFunction::BalancedAccuracy ||
            lossFunction == ELossFunction::Kappa ||
            lossFunction == ELossFunction::WKappa ||
            lossFunction == ELossFunction::LogLikelihoodOfPrediction ||
            lossFunction == ELossFunction::HammingLoss ||
            lossFunction == ELossFunction::ZeroOneLoss ||
            lossFunction == ELossFunction::Logloss ||
            lossFunction == ELossFunction::CrossEntropy ||
            lossFunction == ELossFunction::Accuracy ||
            lossFunction == ELossFunction::Precision ||
            lossFunction == ELossFunction::Recall ||
            lossFunction == ELossFunction::F1 ||
            lossFunction == ELossFunction::TotalF1 ||
            lossFunction == ELossFunction::MCC ||
            lossFunction == ELossFunction::CtrFactor);
}

bool IsClassificationOnlyMetric(ELossFunction lossFunction) {
    return (
        IsOnlyForCrossEntropyOptimization(lossFunction) ||
        lossFunction == ELossFunction::BrierScore ||
        lossFunction == ELossFunction::HingeLoss ||
        lossFunction == ELossFunction::MultiClass ||
        lossFunction == ELossFunction::MultiClassOneVsAll
    );
}

bool IsBinaryClassCompatibleMetric(ELossFunction lossFunction) {
    if (IsClassificationOnlyMetric(lossFunction)) {
        return !IsMultiClassOnlyMetric(lossFunction);
    } else {
       return (lossFunction == ELossFunction::AUC) ||
           (IsGroupwiseMetric(lossFunction) && (lossFunction != ELossFunction::QueryRMSE));
    }
}

bool IsMultiClassCompatibleMetric(ELossFunction lossFunction) {
    return IsMultiDimensionalCompatibleError(lossFunction);
}

bool IsBinaryClassOnlyMetric(ELossFunction lossFunction) {
    return IsClassificationOnlyMetric(lossFunction) && !IsMultiDimensionalCompatibleError(lossFunction);
}

bool IsMultiClassOnlyMetric(ELossFunction lossFunction) {
    return (IsMultiDimensionalCompatibleError(lossFunction) &&
        !IsSingleDimensionalCompatibleError(lossFunction));
}


bool IsClassificationObjective(ELossFunction lossFunction) {
    return IsClassificationOnlyMetric(lossFunction) && IsIn(GetAllObjectives(), lossFunction);
}

bool IsClassificationObjective(const TStringBuf lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsClassificationObjective(lossType);
}

bool IsCvStratifiedObjective(const TStringBuf lossDescription) {
    ELossFunction lossFunction = ParseLossType(lossDescription);
    return (
        lossFunction == ELossFunction::Logloss ||
        lossFunction == ELossFunction::MultiClass ||
        lossFunction == ELossFunction::MultiClassOneVsAll
    );
}

bool IsRegressionObjective(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::MAE ||
            lossFunction == ELossFunction::MAPE ||
            lossFunction == ELossFunction::Poisson ||
            lossFunction == ELossFunction::Quantile ||
            lossFunction == ELossFunction::Expectile ||
            lossFunction == ELossFunction::RMSE ||
            lossFunction == ELossFunction::LogLinQuantile ||
            lossFunction == ELossFunction::Lq ||
            lossFunction == ELossFunction::Huber
    );
}

bool IsRegressionObjective(const TStringBuf lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsRegressionObjective(lossType);
}

bool IsRegressionMetric(ELossFunction lossFunction) {
    return (IsRegressionObjective(lossFunction) ||
        lossFunction == ELossFunction::SMAPE ||
        lossFunction == ELossFunction::R2 ||
        lossFunction == ELossFunction::MSLE ||
        lossFunction == ELossFunction::MedianAbsoluteError
    );
}

bool IsGroupwiseMetric(TStringBuf metricName) {
    ELossFunction lossFunction = ParseLossType(metricName);
    return IsGroupwiseMetric(lossFunction);
}

bool IsGroupwiseMetric(ELossFunction lossFunction) {
    return (
        lossFunction == ELossFunction::QueryRMSE ||
        lossFunction == ELossFunction::QuerySoftMax  ||
        lossFunction == ELossFunction::QueryCrossEntropy ||
        IsGroupwiseOrderMetric(lossFunction) ||
        IsPairwiseMetric(lossFunction)
    );
}

bool IsPairwiseMetric(ELossFunction lossFunction) {
    return (
        lossFunction == ELossFunction::PairLogit ||
        lossFunction == ELossFunction::PairLogitPairwise ||
        lossFunction == ELossFunction::PairAccuracy
    );
}

bool UsesPairsForCalculation(ELossFunction lossFunction) {
    return (
        lossFunction == ELossFunction::PairLogit ||
        lossFunction == ELossFunction::YetiRank ||
        lossFunction == ELossFunction::YetiRankPairwise ||
        lossFunction == ELossFunction::PairLogitPairwise
    );
}

bool IsPlainMode(EBoostingType boostingType) {
    return (boostingType == EBoostingType::Plain);
}

bool IsPairwiseScoring(ELossFunction lossFunction) {
    return (
        lossFunction == ELossFunction::YetiRankPairwise ||
        lossFunction == ELossFunction::PairLogitPairwise ||
        lossFunction == ELossFunction::QueryCrossEntropy
    );
}

bool IsGpuPlainDocParallelOnlyMode(ELossFunction lossFunction) {
    return (
            lossFunction == ELossFunction::YetiRankPairwise ||
            lossFunction == ELossFunction::PairLogitPairwise ||
            lossFunction == ELossFunction::QueryCrossEntropy ||
            lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll
    );
}

bool IsPlainOnlyModeLoss(ELossFunction lossFunction) {
    return (
            lossFunction == ELossFunction::YetiRankPairwise ||
            lossFunction == ELossFunction::PairLogitPairwise ||
            lossFunction == ELossFunction::QueryCrossEntropy
    );
}

bool IsYetiRankLossFunction(ELossFunction lossFunction) {
    return (
        lossFunction == ELossFunction::YetiRank ||
        lossFunction == ELossFunction::YetiRankPairwise
    );
}

bool IsPairLogit(ELossFunction lossFunction) {
    return (
            lossFunction == ELossFunction::PairLogit ||
            lossFunction == ELossFunction::PairLogitPairwise
    );
}

bool IsSecondOrderScoreFunction(EScoreFunction function) {
    switch (function) {
        case EScoreFunction::NewtonL2:
        case EScoreFunction::NewtonCosine: {
            return true;
        }
        case EScoreFunction::Cosine:
        case EScoreFunction::SolarL2:
        case EScoreFunction::LOOL2:
        case EScoreFunction::L2: {
            return false;
        }
        default: {
            ythrow TCatBoostException() << "Unknown score function " << function;
        }
    }
    Y_UNREACHABLE();
}

bool AreZeroWeightsAfterBootstrap(EBootstrapType type) {
    switch (type) {
        case EBootstrapType::Bernoulli:
        case EBootstrapType::Poisson:
            return true;
        default:
            return false;
    }
}

bool ShouldSkipCalcOnTrainByDefault(ELossFunction lossFunction) {
    switch (lossFunction) {
        case ELossFunction::PFound:
        case ELossFunction::YetiRankPairwise:
        case ELossFunction::YetiRank:
        case ELossFunction::NDCG:
        case ELossFunction::AUC:
        case ELossFunction::MedianAbsoluteError:
            return true;
        default:
            return false;
    }
}

bool IsUserDefined(ELossFunction lossFunction) {
    switch (lossFunction) {
        case ELossFunction::PythonUserDefinedPerObject:
        case ELossFunction::UserPerObjMetric:
        case ELossFunction::UserQuerywiseMetric:
            return true;
        default:
            return false;
    }
}

bool IsEmbeddingFeatureEstimator(EFeatureEstimatorType estimatorType) {
    return estimatorType == EFeatureEstimatorType::CosDistanceWithClassCenter ||
            estimatorType == EFeatureEstimatorType::GaussianHomoscedasticModel ||
            estimatorType == EFeatureEstimatorType::GaussianHeteroscedasticModel;
}

bool ShouldSkipFstrGrowPolicy(EGrowPolicy growPolicy) {
    return (
        growPolicy == EGrowPolicy::Depthwise ||
        growPolicy == EGrowPolicy::Lossguide
    );
}

bool IsPlainOnlyModeScoreFunction(EScoreFunction scoreFunction) {
    return (
        scoreFunction != EScoreFunction::Cosine &&
        scoreFunction != EScoreFunction::NewtonCosine
    );
}

bool ShouldBinarizeLabel(ELossFunction lossFunction) {
    return lossFunction == ELossFunction::Logloss;
}
