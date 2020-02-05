#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/model/model.h>
#include <catboost/private/libs/options/enums.h>

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
TString BuildDescription(const NCB::TFeaturesLayout& layout, const TModelSplit& feature);

TVector<TString> GetTreeSplitsDescriptions(const TFullModel& model, size_t treeIdx, const NCB::TDataProviderPtr pool);
TVector<TString> GetTreeLeafValuesDescriptions(const TFullModel& model, size_t treeIdx);
TConstArrayRef<TNonSymmetricTreeStepNode> GetTreeStepNodes(const TFullModel& model, size_t treeIdx);
TVector<ui32> GetTreeNodeToLeaf(const TFullModel& model, size_t treeIdx);
