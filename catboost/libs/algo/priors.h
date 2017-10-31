#pragma once

#include "features_layout.h"
#include <catboost/libs/model/projection.h>

#include <library/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/generic/set.h>
#include <util/generic/map.h>

class TPriors {
public:
    void Init(const yvector<float>& commonPriors,
              const yvector<std::pair<int, yvector<float>>>& perFeaturePriors,
              const TFeaturesLayout& layout) {
        DefaultPriors = commonPriors;

        for (const auto& featurePriors : perFeaturePriors) {
            int feature = featurePriors.first;
            CB_ENSURE(layout.IsCorrectFeatureIdx(feature), "Feature " + ToString(feature) + " in per-feature-priors does not exist");
            CB_ENSURE(layout.GetFeatureType(feature) == EFeatureType::Categorical, "Feature " + ToString(feature) + " in per-feature-priors is not categorical");
            int featureIdx = layout.GetInternalFeatureIdx(feature);
            PerFeaturePriors[featureIdx] = featurePriors.second;
        }
    }

    const yvector<float>& GetPriors(const TProjection& proj) const {
        if (!proj.IsSingleCatFeature() || !PerFeaturePriors.has(proj.CatFeatures[0])) {
            return DefaultPriors;
        }
        return PerFeaturePriors.at(proj.CatFeatures[0]);
    }

    void Swap(TPriors& other) {
        DefaultPriors.swap(other.DefaultPriors);
        PerFeaturePriors.swap(other.PerFeaturePriors);
    }

    Y_SAVELOAD_DEFINE(DefaultPriors, PerFeaturePriors)

private:
    yvector<float> DefaultPriors;
    ymap<int, yvector<float>> PerFeaturePriors;
};
