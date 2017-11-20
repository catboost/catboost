#pragma once

#include "features_layout.h"
#include "projection.h"
#include "split.h"


TString BuildFeatureDescription(const TFeaturesLayout& featuresLayout, const int internalFeatureIdx, EFeatureType type);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TProjection& proj);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TFeatureCombination& proj);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TSplitCandidate& feature);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TSplit& feature);
