#include "enum_helpers.h"
#include "loss_description.h"

#include <util/string/cast.h>

bool IsSupportedOnGpu(ELossFunction lossFunction) {
    switch (lossFunction) {
        case ELossFunction::RMSE:
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
            return true;
        default:
            return false;
    }
}

bool IsClassificationLoss(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::Logloss ||
            lossFunction == ELossFunction::CrossEntropy ||
            lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll ||
            lossFunction == ELossFunction::AUC ||
            lossFunction == ELossFunction::Accuracy ||
            lossFunction == ELossFunction::Precision ||
            lossFunction == ELossFunction::Recall ||
            lossFunction == ELossFunction::F1 ||
            lossFunction == ELossFunction::TotalF1 ||
            lossFunction == ELossFunction::MCC ||
            lossFunction == ELossFunction::CtrFactor);
}

bool IsClassificationLoss(const TString& lossFunction) {
    ELossFunction lossType = ParseLossType(lossFunction);
    return IsClassificationLoss(lossType);
}

bool IsMultiClassError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::MultiClass ||
            lossFunction == ELossFunction::MultiClassOneVsAll);
}

bool IsPairwiseError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::PairLogit);
}

bool IsQuerywiseError(ELossFunction lossFunction) {
    return (lossFunction == ELossFunction::QueryRMSE ||
            lossFunction == ELossFunction::QuerySoftMax);
}

bool IsPlainMode(EBoostingType boostingType) {
    return (boostingType == EBoostingType::Plain);
}
