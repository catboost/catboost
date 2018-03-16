#pragma once

#include "features_layout.h"
#include "target_classifier.h"
#include "projection.h"
#include <catboost/libs/options/cat_feature_options.h>
#include <catboost/libs/metrics/metric.h>

//TODO(kirillovs, noxoomo): remove dirty hack. targetClassifier id = 0 is fake classifier for counter cause almost all catboost code needs target classes
struct TCtrInfo {
    ECtrType Type;
    ui32 BorderCount = 0;
    ui32 TargetClassifierIdx = -1;
    TVector<float> Priors;

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
    void InitCtrHelper(const NCatboostOptions::TCatFeatureParams& catFeatureParams,
                       const TFeaturesLayout& layout,
                       const TVector<float>& target,
                       ELossFunction loss,
                       const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor);

    const TVector<TCtrInfo>& GetCtrInfo(const TProjection& projection) const {
        if (projection.IsSingleCatFeature()) {
            const int featureId = projection.CatFeatures[0];
            if (PerFeatureCtrs.has(featureId)) {
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

    Y_SAVELOAD_DEFINE(TargetClassifiers, SimpleCtrs, PerFeatureCtrs, TreeCtrs)

private:
    TVector<TTargetClassifier> TargetClassifiers;

    TVector<TCtrInfo> SimpleCtrs;
    TMap<int, TVector<TCtrInfo>> PerFeatureCtrs;
    TVector<TCtrInfo> TreeCtrs;
};
