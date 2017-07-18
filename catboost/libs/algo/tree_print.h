#pragma once

#include "calc_fstr.h"
#include "features_layout.h"
#include <catboost/libs/model/tensor_struct.h>
#include <catboost/libs/model/projection.h>

TString BuildFeatureDescription(const TFeaturesLayout& featuresLayout, const int internalFeatureIdx, EFeatureType type);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TSplit& split);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TTensorStructure3& ts);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TProjection& proj);
TString BuildDescription(const TFeaturesLayout& featuresLayout, const TFeature& feature);
