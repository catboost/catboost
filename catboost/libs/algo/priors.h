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
              const yvector<std::pair<int, yvector<float>>>& perCtrPriors,
              const yvector<std::pair<std::pair<int, int>, yvector<float>>>& perFeatureCtrPriors,
              int ctrCount,
              const TFeaturesLayout& layout) {
        Priors.push_back(commonPriors);
        int featureCount = layout.GetCatFeatureCount();
        PerFeatureCtrPriorVec.resize(featureCount, yvector<int>(ctrCount, 0));

        for (const auto& ctrPriors : perCtrPriors) {
            int ctrIdx = ctrPriors.first;
            CB_ENSURE(0 <= ctrIdx && ctrIdx < ctrCount, "Ctr " + ToString(ctrIdx) + " in per-ctr-priors does not exist");
            for (int featureIdx = 0; featureIdx < featureCount; ++featureIdx) {
                PerFeatureCtrPriorVec[featureIdx][ctrIdx] = Priors.ysize();
            }
            Priors.push_back(ctrPriors.second);
        }

        for (const auto& featurePriors : perFeaturePriors) {
            int feature = featurePriors.first;
            CB_ENSURE(layout.IsCorrectFeatureIdx(feature), "Feature " + ToString(feature) + " in per-feature-priors does not exist");
            CB_ENSURE(layout.GetFeatureType(feature) == EFeatureType::Categorical, "Feature " + ToString(feature) + " in per-feature-priors is not categorical");
            int featureIdx = layout.GetInternalFeatureIdx(feature);
            for (int ctrIdx = 0; ctrIdx < ctrCount; ++ctrIdx) {
                PerFeatureCtrPriorVec[featureIdx][ctrIdx] = Priors.ysize();
            }
            Priors.push_back(featurePriors.second);
        }

        for (const auto& featureCtrPriors : perFeatureCtrPriors) {
            int feature = featureCtrPriors.first.first;
            int ctrIdx = featureCtrPriors.first.second;
            CB_ENSURE(layout.IsCorrectFeatureIdx(feature), "Feature " + ToString(feature) + " in per-feature-ctr-priors does not exist");
            CB_ENSURE(layout.GetFeatureType(feature) == EFeatureType::Categorical, "Feature " + ToString(feature) + " in per-feature-ctr-priors is not categorical");
            CB_ENSURE(0 <= ctrIdx && ctrIdx < ctrCount, "Ctr " + ToString(ctrIdx) + " in per-feature-ctr-priors does not exist");
            int featureIdx = layout.GetInternalFeatureIdx(feature);
            PerFeatureCtrPriorVec[featureIdx][ctrIdx] = Priors.ysize();
            Priors.push_back(featureCtrPriors.second);
        }
    }

    const yvector<float>& GetPriors(const TProjection& proj, int ctrIdx) const {
        return Priors[PerFeatureCtrPriorVec[proj.IsSingleCatFeature() ? proj.CatFeatures[0] : 0][ctrIdx]];
    }

    void Swap(TPriors& other) {
        Priors.swap(other.Priors);
        PerFeatureCtrPriorVec.swap(other.PerFeatureCtrPriorVec);
    }

    Y_SAVELOAD_DEFINE(Priors, PerFeatureCtrPriorVec)

private:
    yvector<yvector<float>> Priors;
    yvector<yvector<int>> PerFeatureCtrPriorVec;
};
