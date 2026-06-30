#pragma once

#include "projection.h"
#include "target_classifier.h"

#include <catboost/private/libs/algo_helpers/custom_objective_descriptor.h>
#include <catboost/libs/data/util.h>

#include <util/generic/array_ref.h>
#include <util/generic/map.h>


namespace NCatboostOptions {
    class TCatFeatureParams;
}

namespace NCB {
    class TFeaturesLayout;
}


/* TODO(kirillovs, noxoomo): remove dirty hack. targetClassifier id = 0 is fake classifier for counter
 * cause almost all catboost code needs target classes
 */
struct TCtrInfo {
    ECtrType Type;
    ui32 BorderCount = 0;
    ui32 TargetClassifierIdx = -1;
    TVector<float> Priors;

public:
    Y_SAVELOAD_DEFINE(Type, BorderCount, TargetClassifierIdx, Priors);
};

inline int GetTargetBorderCount(const TCtrInfo& ctrInfo, ui32 targetClassesCount) {
    if (ctrInfo.Type == ECtrType::BinarizedTargetMeanValue || ctrInfo.Type == ECtrType::Counter) {
        return 1;
    }
    Y_ASSERT(targetClassesCount > 0);

    return ctrInfo.Type == ECtrType::Buckets ? targetClassesCount : targetClassesCount - 1;
}

class TCtrHelper {
public:
    void InitCtrHelper(
        const NCatboostOptions::TCatFeatureParams& catFeatureParams,
        const NCB::TFeaturesLayout& layout,
        NCB::TMaybeData<TConstArrayRef<TConstArrayRef<float>>> targets,
        ELossFunction loss,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        bool allowConstLabel);

    const TVector<TCtrInfo>& GetCtrInfo(const TProjection& projection) const {
        if (projection.IsSingleCatFeature()) {
            const int featureId = projection.CatFeatures[0];
            if (PerFeatureCtrs.contains(featureId)) {
                return PerFeatureCtrs.at(featureId);
            } else {
                return SimpleCtrs;
            }
        }
        return TreeCtrs;
    }

    const TVector<TTargetClassifier>& GetTargetClassifiers() const {
        return TargetClassifiers;
    }

    Y_SAVELOAD_DEFINE(TargetClassifiers, SimpleCtrs, PerFeatureCtrs, TreeCtrs);

private:
    TVector<TTargetClassifier> TargetClassifiers;

    TVector<TCtrInfo> SimpleCtrs;
    TMap<int, TVector<TCtrInfo>> PerFeatureCtrs;
    TVector<TCtrInfo> TreeCtrs;
};
