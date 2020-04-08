#include "serialization_helper.h"

template <class TContainer>
inline TString Print(TContainer&& data) {
    TStringBuilder builder;
    for (auto& val : data) {
        builder << val << " ";
    }
    return builder;
}

NCatboostCuda::TCtr NCatboostCuda::MigrateCtr(TBinarizedFeaturesManager& featuresManager,
                                              const TModelFeaturesMap& map,
                                              const TCtr& oldCtr) {
    TCtr newCtr = oldCtr;
    TVector<TBinarySplit> binarySplits = newCtr.FeatureTensor.GetSplits();
    for (auto& split : binarySplits) {
        split.FeatureId = UpdateFeatureId(featuresManager, map, split.FeatureId);
    }
    TVector<ui32> catFeatures = newCtr.FeatureTensor.GetCatFeatures();
    for (auto& catFeature : catFeatures) {
        catFeature = UpdateFeatureId(featuresManager, map, catFeature);
    }
    newCtr.FeatureTensor = TFeatureTensor();
    newCtr.FeatureTensor.AddBinarySplit(binarySplits);
    newCtr.FeatureTensor.AddCatFeature(catFeatures);
    return newCtr;
}


template <class TFeatInfo>
inline void ValidateBorders(const TFeatInfo& featureInfo, const NCatboostCuda::TBinarizedFeaturesManager& manager, ui32 id) {
    CB_ENSURE(featureInfo.Borders == manager.GetBorders(id),
              "Error: progress borders should be consistent: featureId=" << featureInfo.Feature << " borders "
                                                                         << Print(featureInfo.Borders) << " vs " << Print(manager.GetBorders(id)));

}



ui32 NCatboostCuda::UpdateFeatureId(TBinarizedFeaturesManager& featuresManager,
                                    const TModelFeaturesMap& map,
                                    const ui32 featureId) {
    if (map.Ctrs.contains(featureId)) {
        const auto& info = map.Ctrs.at(featureId);
        TCtr remapedCtr = MigrateCtr(featuresManager, map, info.Ctr);

        if (featuresManager.IsKnown(remapedCtr)) {
            ui32 remappedId = featuresManager.GetId(remapedCtr);

            CB_ENSURE(info.Borders == featuresManager.GetBorders(remappedId),
                      " tensor : " << remapedCtr.FeatureTensor << "  (ctr type "
                                   << remapedCtr.Configuration.Type << "). Error: progress borders should be consistent: " << remappedId << " / " << featureId << " " << Print(info.Borders) << " vs " << Print(featuresManager.GetBorders(remappedId)));
            return remappedId;
        } else {
            return featuresManager.AddCtr(remapedCtr,
                                          TVector<float>(info.Borders));
        }
    } else if (map.FloatFeatures.contains(featureId)) {
        auto& floatInfo = map.FloatFeatures.at(featureId);
        ValidateBorders(floatInfo, featuresManager, floatInfo.Feature);
        return floatInfo.Feature;
    } else if (map.CatFeaturesMap.contains(featureId)) {
        const ui32 dataProviderId = map.CatFeaturesMap.at(featureId);
        return featuresManager.GetFeatureManagerIdForCatFeature(dataProviderId);
    } else if (map.CalculatedFeaturesMap.contains(featureId)) {
        const auto& featureInfo = map.CalculatedFeaturesMap.at(featureId);
        auto featureManagerId = featuresManager.GetId(featureInfo.Feature);
        ValidateBorders(featureInfo, featuresManager, featureManagerId);
        return featureManagerId;
    } else {
        ythrow yexception() << "Error: can't remap featureId #" << featureId;
    }
}
