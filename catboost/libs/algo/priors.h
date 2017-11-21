#pragma once

#include "features_layout.h"
#include "projection.h"

#include <library/binsaver/bin_saver.h>

#include <util/generic/vector.h>
#include <util/generic/set.h>
#include <util/generic/map.h>

class TPriors {
public:
    void Init(const TVector<float>& commonPriors,
              const TVector<float>& counterPriors,
              const TVector<std::pair<int, TVector<float>>>& perFeaturePriors,
              const TVector<TCtrDescription>& ctrDescriptions,
              const TFeaturesLayout& layout) {
        DefaultPriors.resize((int)ECtrType::CtrTypesCount, commonPriors);
        DefaultPriors[(int)ECtrType::Counter] = counterPriors;

        PerFeaturePriors.resize((int)ECtrType::CtrTypesCount);
        CtrDescriptions = ctrDescriptions;

        for (const auto& featurePriors : perFeaturePriors) {
            int feature = featurePriors.first;
            CB_ENSURE(layout.IsCorrectFeatureIdx(feature), "Feature " + ToString(feature) + " in per-feature-priors does not exist");
            CB_ENSURE(layout.GetFeatureType(feature) == EFeatureType::Categorical, "Feature " + ToString(feature) + " in per-feature-priors is not categorical");
            int featureIdx = layout.GetInternalFeatureIdx(feature);
            for (auto& perFeaturePriors : PerFeaturePriors) {
                perFeaturePriors[featureIdx] = featurePriors.second;
            }
        }
    }

    const TVector<float>& GetPriors(const TProjection& proj, int ctrIdx) const {
        auto ctrType = CtrDescriptions[ctrIdx].CtrType;
        if (!proj.IsSingleCatFeature() || !PerFeaturePriors[(int)ctrType].has(proj.CatFeatures[0])) {
            return DefaultPriors[(int)ctrType];
        }
        return PerFeaturePriors[(int)ctrType].at(proj.CatFeatures[0]);
    }

    void Swap(TPriors& other) {
        DefaultPriors.swap(other.DefaultPriors);
        PerFeaturePriors.swap(other.PerFeaturePriors);
    }

    Y_SAVELOAD_DEFINE(DefaultPriors, PerFeaturePriors)

private:
    TVector<TVector<float>> DefaultPriors;
    TVector<ymap<int, TVector<float>>> PerFeaturePriors;

    TVector<TCtrDescription> CtrDescriptions;
};
