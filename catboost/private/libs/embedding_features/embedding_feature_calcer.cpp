#include "embedding_feature_calcer.h"

namespace NCB {
    void TEmbeddingFeatureCalcer::TrimFeatures(TConstArrayRef<ui32> featureIndices) {
        const ui32 featureCount = FeatureCount();
        CB_ENSURE(
            featureIndices.size() <= featureCount && featureIndices.back() < featureCount,
            "Specified trim feature indices is greater than number of features that calcer produce"
        );
        ActiveFeatureIndices = TVector<ui32>(featureIndices.begin(), featureIndices.end());
    }

    TConstArrayRef<ui32> TEmbeddingFeatureCalcer::GetActiveFeatureIndices() const {
        return MakeConstArrayRef(ActiveFeatureIndices);
    }
}
