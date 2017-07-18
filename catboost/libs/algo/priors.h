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
        ymap<int, yvector<float>> additionalPriors;
        for (const auto& featureToPriors : perFeaturePriors) {
            int feature = featureToPriors.first;
            CB_ENSURE(layout.IsCorrectFeatureIdx(feature), "Feature " + ToString(feature) + " in per-feature-priors does not exist");
            CB_ENSURE(layout.GetFeatureType(feature) == EFeatureType::Categorical,
                     "Feature " + ToString(feature) + " in per-feature-priors is not categorical");
            additionalPriors[layout.GetInternalFeatureIdx(feature)] = featureToPriors.second;
        }

        Priors.push_back(commonPriors);

        int maxCatFeature = 0;
        if (!additionalPriors.empty()) {
            maxCatFeature = additionalPriors.rbegin()->first;
        }

        PerFeaturePriorVec.resize(maxCatFeature, 0);
        for (int i = 0; i < maxCatFeature; ++i) {
            if (additionalPriors.has(i)) {
                PerFeaturePriorVec[i] = Priors.ysize();
                Priors.emplace_back(additionalPriors[i]);
            }
        }
    }

    const yvector<float>& GetPriors(const TProjection& proj) const {
        if (!proj.IsSingleCatFeature() || proj.CatFeatures[0] >= PerFeaturePriorVec.ysize()) {
            return Priors[0];
        }
        return Priors[PerFeaturePriorVec[proj.CatFeatures[0]]];
    }

    void Swap(TPriors& other) {
        Priors.swap(other.Priors);
        PerFeaturePriorVec.swap(other.PerFeaturePriorVec);
    }

    Y_SAVELOAD_DEFINE(Priors, PerFeaturePriorVec)

private:
    yvector<yvector<float>> Priors;
    yvector<int> PerFeaturePriorVec;
};

