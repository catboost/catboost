#pragma once


#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>

#include <catboost/libs/model/target_classifier.h>
#include <catboost/private/libs/options/enums.h>

#include <library/cpp/grid_creator/binarization.h>

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>


TTargetClassifier BuildTargetClassifier(
    TConstArrayRef<float> target,
    ELossFunction loss,
    const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
    int targetBorderCount,
    EBorderSelectionType targetBorderType,
    bool allowConstLabel,
    ui32 targetId);
