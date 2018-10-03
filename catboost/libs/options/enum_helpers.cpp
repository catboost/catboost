#include "enum_helpers.h"
#include "loss_description.h"

#include <util/string/cast.h>


bool IsSingleDimensionalError(ELossFunction lossFunction) {
    return (lossFunction != ELossFunction::MultiClass &&
            lossFunction != ELossFunction::MultiClassOneVsAll);
}

bool IsMultiDimensionalError(ELossFunction lossFunction) {
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

bool IsForOrderOptimization(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::AUC ||  // classification metric
            lossFunction == ELossFunction::YetiRank ||  // ranking metrics
            lossFunction == ELossFunction::PrecisionAt ||
            lossFunction == ELossFunction::RecallAt ||
            lossFunction == ELossFunction::MAP ||
            lossFunction == ELossFunction::YetiRankPairwise ||
            lossFunction == ELossFunction::PFound ||
            lossFunction == ELossFunction::NDCG ||
            lossFunction == ELossFunction::AverageGain ||
            lossFunction == ELossFunction::QueryAverage);
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
            lossFunction == ELossFunction::AUC ||
            lossFunction == ELossFunction::Accuracy ||
            lossFunction == ELossFunction::Precision ||
            lossFunction == ELossFunction::Recall ||
            lossFunction == ELossFunction::F1 ||
            lossFunction == ELossFunction::TotalF1 ||
            lossFunction == ELossFunction::MCC ||
            lossFunction == ELossFunction::CtrFactor);
}

bool IsBinaryClassError(ELossFunction lossFunction) {
    return (IsOnlyForCrossEntropyOptimization(lossFunction) ||
            lossFunction == ELossFunction::BrierScore ||
            lossFunction == ELossFunction::HingeLoss);
}

bool IsMultiClassError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll ||
            lossFunction == ELossFunction::HingeLoss ||
            lossFunction == ELossFunction::Kappa ||
            lossFunction == ELossFunction::WKappa ||
            lossFunction == ELossFunction::HammingLoss ||
            lossFunction == ELossFunction::ZeroOneLoss);
}

bool IsClassificationLoss(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::Logloss ||
            lossFunction == ELossFunction::CrossEntropy ||
            lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll);
}

bool IsClassificationLoss(const TString& lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsClassificationLoss(lossType);
}

bool IsRegressionLoss(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::MAE ||
            lossFunction == ELossFunction::MAPE ||
            lossFunction == ELossFunction::Poisson ||
            lossFunction == ELossFunction::Quantile ||
            lossFunction == ELossFunction::RMSE ||
            lossFunction == ELossFunction::LogLinQuantile ||
            lossFunction == ELossFunction::SMAPE);
}

bool IsRegressionLoss(const TString& lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsRegressionLoss(lossType);
}

bool IsQuerywiseError(ELossFunction lossFunction) {
    return (
        lossFunction == ELossFunction::QueryRMSE ||
        lossFunction == ELossFunction::QuerySoftMax ||
        lossFunction == ELossFunction::PairLogit ||
        lossFunction == ELossFunction::YetiRank ||
        lossFunction == ELossFunction::YetiRankPairwise ||
        lossFunction == ELossFunction::PairLogitPairwise
    );
}

bool IsPairwiseError(ELossFunction lossFunction) {
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

bool ShouldGenerateYetiRankPairs(ELossFunction lossFunction) {
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
        case EScoreFunction::NewtonCorrelation: {
            return true;
        }
        case EScoreFunction::Correlation:
        case EScoreFunction::SolarL2:
        case EScoreFunction::LOOL2:
        case EScoreFunction::L2: {
            return false;
        }
        default: {
            ythrow TCatboostException() << "Unknown score function " << function;
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
