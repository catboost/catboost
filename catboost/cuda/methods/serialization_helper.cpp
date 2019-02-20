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
        const ui32 featureManagerId = featuresManager.GetFeatureManagerIdForFloatFeature(floatInfo.DataProviderId);
        CB_ENSURE(floatInfo.Borders == featuresManager.GetBorders(featureManagerId),
                  "Error: progress borders should be consistent: featureId=" << featureId << " borders "
                  << Print(floatInfo.Borders) << " vs " << Print(featuresManager.GetBorders(featureManagerId)));
        return featureManagerId;
    } else if (map.CatFeaturesMap.contains(featureId)) {
        const ui32 dataProviderId = map.CatFeaturesMap.at(featureId);
        return featuresManager.GetFeatureManagerIdForCatFeature(dataProviderId);
    } else {
        ythrow yexception() << "Error: can't remap featureId #" << featureId;
    }
}
