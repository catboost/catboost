#include "features_layout_helpers.h"

NCB::TFeaturesLayout MakeFeaturesLayout(const TFullModel& model)
{
    return NCB::TFeaturesLayout(
        model.ModelTrees->GetFloatFeatures(),
        model.ModelTrees->GetCatFeatures(),
        model.ModelTrees->GetTextFeatures(),
        model.ModelTrees->GetEmbeddingFeatures()
    );
}
