#include "tree_print.h"

#include "projection.h"
#include "split.h"

#include <catboost/libs/data_new/features_layout.h>

#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/system/yassert.h>


TString BuildFeatureDescription(
    const NCB::TFeaturesLayout& featuresLayout,
    const int internalFeatureIdx,
    EFeatureType type) {

    TString externalFeatureDescription = featuresLayout.GetExternalFeatureDescription(
        internalFeatureIdx,
        type);
    if (externalFeatureDescription.empty()) {
        // just return index
        return ToString<int>(featuresLayout.GetExternalFeatureIdx(internalFeatureIdx, type));
    }
    return externalFeatureDescription;
}

TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TProjection& proj) {
    TStringBuilder result;
    result << "{";
    int fc = 0;
    for (const int featureIdx : proj.CatFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(
            featuresLayout,
            featureIdx,
            EFeatureType::Categorical);
        result << featureDescription;
    }

    for (const TBinFeature& feature : proj.BinFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(
            featuresLayout,
            feature.FloatFeature,
            EFeatureType::Float);
        result << featureDescription << " b" << feature.SplitIdx;
    }

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(
            featuresLayout,
            feature.CatFeatureIdx,
            EFeatureType::Categorical);
        result << featureDescription << " val = " << feature.Value;
    }
    result << "}";
    return result;
}

TString BuildDescription(const NCB::TFeaturesLayout& featuresLayout, const TFeatureCombination& proj) {
    TStringBuilder result;
    result << "{";
    int fc = 0;
    for (const int featureIdx : proj.CatFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(
            featuresLayout,
            featureIdx,
            EFeatureType::Categorical);
        result << featureDescription;
    }

    for (const auto& feature : proj.BinFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(
            featuresLayout,
            feature.FloatFeature,
            EFeatureType::Float);
        result << featureDescription << " border=" << feature.Split;
    }

    for (const TOneHotSplit& feature : proj.OneHotFeatures) {
        if (fc++ > 0) {
            result << ", ";
        }
        TString featureDescription = BuildFeatureDescription(
            featuresLayout,
            feature.CatFeatureIdx,
            EFeatureType::Categorical);
        result << featureDescription << " val = " << feature.Value;
    }
    result << "}";
    return result;
}

TString BuildDescription(const NCB::TFeaturesLayout& layout, const TSplitCandidate& feature) {
    TStringBuilder result;
    if (feature.Type == ESplitType::OnlineCtr) {
        result << BuildDescription(layout, feature.Ctr.Projection);
        result << " pr" << (int)feature.Ctr.PriorIdx;
        result << " tb" << (int)feature.Ctr.TargetBorderIdx;
        result << " type" << (int)feature.Ctr.CtrIdx;
    } else if (feature.Type == ESplitType::FloatFeature) {
        result << BuildFeatureDescription(layout, feature.FeatureIdx, EFeatureType::Float);
    } else {
        Y_ASSERT(feature.Type == ESplitType::OneHotFeature);
        result << BuildFeatureDescription(layout, feature.FeatureIdx, EFeatureType::Categorical);
    }
    return result;
}

TString BuildDescription(const NCB::TFeaturesLayout& layout, const TSplit& feature) {
    TStringBuilder result;
    result << BuildDescription(layout, static_cast<const TSplitCandidate&>(feature));

    if (feature.Type == ESplitType::OnlineCtr) {
        result << ", border=" << feature.BinBorder;
    } else if (feature.Type == ESplitType::FloatFeature) {
        result << ", bin=" << feature.BinBorder;
    } else {
        Y_ASSERT(feature.Type == ESplitType::OneHotFeature);
        result << ", value=" << feature.BinBorder;
    }
    return result;
}
