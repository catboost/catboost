#include "target_classifier.h"

#include <catboost/libs/helpers/vector_helpers.h>

#include <library/cpp/grid_creator/binarization.h>

#include <util/generic/algorithm.h>


static TVector<float> GetMultiClassBorders(int cnt) {
    TVector<float> borders(cnt);
    for (int i = 0; i < cnt; ++i) {
        borders[i] = 0.5 + i;
    }
    return borders;
}

static TVector<float> SelectBorders(
    TConstArrayRef<float> target,
    int targetBorderCount,
    EBorderSelectionType targetBorderType,
    bool allowConstLabel,
    bool targetValuesMayContainNans = false) {

    TVector<float> learnTarget(target.begin(), target.end());

    THashSet<float> borderSet = BestSplit(learnTarget, targetBorderCount, targetBorderType, targetValuesMayContainNans);
    TVector<float> borders(borderSet.begin(), borderSet.end());
    CB_ENSURE((borders.ysize() > 0) || allowConstLabel, "0 target borders");
    if (borders.empty()) {
        borders.push_back(target.front());
    }

    Sort(borders.begin(), borders.end());

    return borders;
}

TTargetClassifier BuildTargetClassifier(
    TConstArrayRef<float> target,
    ELossFunction loss,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    int targetBorderCount,
    EBorderSelectionType targetBorderType,
    bool allowConstLabel,
    ui32 targetId = 0) {

    if (targetBorderCount == 0) {
        return TTargetClassifier();
    }

    CB_ENSURE(!target.empty(), "train target should not be empty");

    TMinMax<float> targetBounds = CalcMinMax<float>(target);
    CB_ENSURE(
        (targetBounds.Min != targetBounds.Max) || allowConstLabel,
        "target in train should not be constant");

    switch (loss) {
        case ELossFunction::Cox:
        case ELossFunction::RMSE:
        case ELossFunction::MultiRMSE:
        case ELossFunction::SurvivalAft:
        case ELossFunction::RMSEWithUncertainty:
        case ELossFunction::Quantile:
        case ELossFunction::MultiQuantile:
        case ELossFunction::Expectile:
        case ELossFunction::Lq:
        case ELossFunction::LogLinQuantile:
        case ELossFunction::Poisson:
        case ELossFunction::MAE:
        case ELossFunction::MAPE:
        case ELossFunction::PairLogit:
        case ELossFunction::PairLogitPairwise:
        case ELossFunction::QueryRMSE:
        case ELossFunction::GroupQuantile:
        case ELossFunction::QuerySoftMax:
        case ELossFunction::YetiRank:
        case ELossFunction::YetiRankPairwise:
        case ELossFunction::StochasticFilter:
        case ELossFunction::LambdaMart:
        case ELossFunction::StochasticRank:
        case ELossFunction::Logloss:
        case ELossFunction::CrossEntropy:
        case ELossFunction::MultiLogloss:
        case ELossFunction::MultiCrossEntropy:
        case ELossFunction::Huber:
        case ELossFunction::UserPerObjMetric:
        case ELossFunction::UserQuerywiseMetric:
        case ELossFunction::Tweedie:
        case ELossFunction::Focal:
        case ELossFunction::LogCosh:
            return TTargetClassifier(
                SelectBorders(target, targetBorderCount, targetBorderType, allowConstLabel),
                targetId);

        case ELossFunction::MultiRMSEWithMissingValues:
            return TTargetClassifier(
                SelectBorders(target, targetBorderCount, targetBorderType, allowConstLabel, true),
                targetId);

        case ELossFunction::MultiClass:
        case ELossFunction::MultiClassOneVsAll:
            return TTargetClassifier(GetMultiClassBorders(targetBorderCount), targetId);

        case ELossFunction::PythonUserDefinedMultiTarget:
        case ELossFunction::PythonUserDefinedPerObject: {
            Y_ASSERT(objectiveDescriptor.Defined());
            return TTargetClassifier(
                SelectBorders(target, targetBorderCount, targetBorderType, allowConstLabel),
                targetId);
        }

        default:
            CB_ENSURE(false, "provided error function is not supported");
    }
}
