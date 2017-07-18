#include "tree_print.h"
#include <catboost/libs/data/pool.h>

#include <util/string/builder.h>

TString BuildFeatureDescription(const TFeaturesLayout& featuresLayout, const int internalFeatureIdx, EFeatureType type) {
    TString externalFeatureDescription = featuresLayout.GetExternalFeatureDescription(internalFeatureIdx, type);
    if (externalFeatureDescription.empty()) {
        int featureIdx = featuresLayout.GetFeature(internalFeatureIdx, type);
        switch (type) {
            case EFeatureType::Float:
                return "f" + ToString<int>(featureIdx);
                break;
            case EFeatureType::Categorical:
                return "c" + ToString<int>(featureIdx);
                break;
            default:
                Y_ASSERT(false);
        }
    }
    return externalFeatureDescription;
}

TString BuildDescription(const TFeaturesLayout& featuresLayout, const TSplit& split) {
    TStringBuilder result;
    TFeature feature;
    int splitIdxOrValue;
    split.BuildTFeatureFormat(&feature, &splitIdxOrValue);

    if (feature.Type == ESplitType::OnlineCtr || feature.Type == ESplitType::FloatFeature) {
        result << "(" << BuildDescription(featuresLayout, feature) << ", split" << splitIdxOrValue << ")";
    } else {
        result << "(" << BuildDescription(featuresLayout, feature) << ", value = " << splitIdxOrValue << ")";
    }

    return TString(result);
}

TString BuildDescription(const TFeaturesLayout& featuresLayout, const TProjection& proj) {
    TStringBuilder result;
    result << "{";
    int fc = 0;
    for (const int featureIdx : proj.CatFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(featuresLayout, featureIdx, EFeatureType::Categorical);
        result << featureDescription;
    }

    for (const TBinFeature& feature : proj.BinFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(featuresLayout, feature.FloatFeature, EFeatureType::Float);
        result << featureDescription << " b" << feature.SplitIdx;
    }

    for (const TOneHotFeature& feature : proj.OneHotFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(featuresLayout, feature.CatFeatureIdx, EFeatureType::Categorical);
        result << featureDescription << " val = " << feature.Value;
    }
    result << "}";
    return TString(result);
}

TString BuildDescription(const TFeaturesLayout& featuresLayout, const TTensorStructure3& ts) {
    TStringBuilder result;
    for (const auto& split : ts.SelectedSplits) {
        result << BuildDescription(featuresLayout, split);
    }
    return TString(result);
}

TString BuildDescription(const TFeaturesLayout& layout, const TFeature& feature) {
    TStringBuilder result;
    if (feature.Type == ESplitType::OnlineCtr) {
        result << BuildDescription(layout, feature.Ctr.Projection);
        result << " pr" << (int)feature.Ctr.PriorIdx;
        result << " tb" << (int)feature.Ctr.TargetBorderIdx;
        result << " type" << (int)feature.Ctr.CtrTypeIdx;
    } else if (feature.Type == ESplitType::FloatFeature) {
        result << BuildFeatureDescription(layout, feature.FeatureIdx, EFeatureType::Float);
    } else {
        Y_ASSERT(feature.Type == ESplitType::OneHotFeature);
        result << BuildFeatureDescription(layout, feature.FeatureIdx, EFeatureType::Categorical);
    }
    return TString(result);
}
