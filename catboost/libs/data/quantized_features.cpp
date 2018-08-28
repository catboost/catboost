#include "quantized_features.h"

void TAllFeatures::Swap(TAllFeatures& other) {
    FloatHistograms.swap(other.FloatHistograms);
    CatFeaturesRemapped.swap(other.CatFeaturesRemapped);
    OneHotValues.swap(other.OneHotValues);
    IsOneHot.swap(other.IsOneHot);
}


size_t TAllFeatures::GetDocCount() const {
    for (const auto& floatHistogram : FloatHistograms) {
        if (!floatHistogram.empty())
            return floatHistogram.size();
    }
    for (const auto& catFeatures : CatFeaturesRemapped) {
        if (!catFeatures.empty())
            return catFeatures.size();
    }
    return 0;
}
