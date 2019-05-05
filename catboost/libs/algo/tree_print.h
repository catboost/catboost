#pragma once

#include <catboost/libs/options/enums.h>

#include <util/generic/string.h>


struct TFeatureCombination;
struct TProjection;
struct TSplitCandidate;
struct TSplit;

namespace NCB {
    class TFeaturesLayout;
}


TString BuildFeatureDescription(
    const NCB::TFeaturesLayout& featuresLayout,
    const int internalFeatureIdx,
    EFeatureType type);
TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TProjection& proj);
TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TFeatureCombination& proj);
TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TSplitCandidate& feature);
TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TSplit& feature);
