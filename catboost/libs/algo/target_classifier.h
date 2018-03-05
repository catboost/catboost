#pragma once

#include <catboost/libs/options/enums.h>
#include <catboost/libs/model/target_classifier.h>
#include <catboost/libs/metrics/metric.h>

#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <library/grid_creator/binarization.h>

TTargetClassifier BuildTargetClassifier(const TVector<float>& target,
                                        ELossFunction loss,
                                        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
                                        int targetBorderCount,
                                        EBorderSelectionType targetBorderType);
