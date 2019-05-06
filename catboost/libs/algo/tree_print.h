#pragma once

#include "projection.h"
#include "split.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/data_new/features_layout.h>
#include <catboost/libs/model/model.h>


TString BuildFeatureDescription(const TMaybe<NCB::TFeaturesLayout>& featuresLayout, const int internalFeatureIdx, EFeatureType type);
TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TProjection& proj);
TString BuildDescription(const TMaybe<NCB::TFeaturesLayout>& featuresLayout, const TFeatureCombination& proj);
TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TSplitCandidate& feature);
TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TSplit& feature);
TString BuildDescription(const TMaybe<NCB::TFeaturesLayout>& layout, const TModelSplit& feature);

TVector<TString> GetTreeSplitsDescriptions(const TFullModel& model, int tree_idx, NCB::TDataProvider pool);
TVector<TString> GetTreeLeafValuesDescriptions(const TFullModel& model, int tree_idx, int leaves_num);