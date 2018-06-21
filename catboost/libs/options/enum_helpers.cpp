#include "enum_helpers.h"
#include "loss_description.h"

#include <util/string/cast.h>

bool IsOnlyForCrossEntropyOptimization(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::BalancedErrorRate ||
            lossFunction == ELossFunction::BalancedAccuracy ||
            lossFunction == ELossFunction::Kappa ||
            lossFunction == ELossFunction::WKappa ||
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
    return IsBinaryClassError(lossFunction) || IsMultiClassError(lossFunction);
}

bool IsClassificationLoss(const TString& lossDescription) {
    ELossFunction lossType = ParseLossType(lossDescription);
    return IsClassificationLoss(lossType);
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
        lossFunction == ELossFunction::PairLogitPairwise
    );
}

bool IsGpuDocParallelOnlyMode(ELossFunction lossFunction) {
    return (
            lossFunction == ELossFunction::YetiRankPairwise ||
            lossFunction == ELossFunction::PairLogitPairwise ||
            lossFunction == ELossFunction::QueryCrossEntropy
    );
}

bool IsPlainOnlyModeLoss(ELossFunction lossFunction) {
    return (
            lossFunction == ELossFunction::YetiRankPairwise ||
            lossFunction == ELossFunction::PairLogitPairwise ||
            lossFunction == ELossFunction::QueryCrossEntropy
    );
}

bool IsItNecessaryToGeneratePairs(ELossFunction lossFunction) {
    return (
        lossFunction == ELossFunction::YetiRank ||
        lossFunction == ELossFunction::YetiRankPairwise
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
            return true;
        default:
            return false;
    }
}
