#include "tree_print.h"

#include "projection.h"
#include "split.h"

#include <catboost/libs/data_new/features_layout.h>
#include <catboost/libs/data_new/objects.h>

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

TString BuildDescription(const NCB::TFeaturesLayout& layout, const TModelSplit& feature) {
    TStringBuilder result;
    if (feature.Type == ESplitType::OnlineCtr) {
        result << BuildDescription(layout, feature.OnlineCtr.Ctr.Base.Projection);
        result << " pr_num" << (int)feature.OnlineCtr.Ctr.PriorNum;
        result << " tb" << (int)feature.OnlineCtr.Ctr.TargetBorderIdx;
        result << " type" << (int)feature.OnlineCtr.Ctr.Base.CtrType;
    } else if (feature.Type == ESplitType::FloatFeature) {
        result << BuildFeatureDescription(layout, feature.FloatFeature.FloatFeature, EFeatureType::Float);
    } else {
        Y_ASSERT(feature.Type == ESplitType::OneHotFeature);
        result << BuildFeatureDescription(layout, feature.OneHotFeature.CatFeatureIdx, EFeatureType::Categorical);
    }

    if (feature.Type == ESplitType::OnlineCtr) {
        result << ", border=" << feature.OnlineCtr.Border;
    } else if (feature.Type == ESplitType::FloatFeature) {
        result << ", bin=" << feature.FloatFeature.Split;
    } else {
        Y_ASSERT(feature.Type == ESplitType::OneHotFeature);
        result << ", value=";
    }
    return result;
}

TVector<TString> GetTreeSplitsDescriptions(const TFullModel& model, int tree_idx, const NCB::TDataProvider& pool) {
    //TODO: support non symmetric trees
    CB_ENSURE(model.IsOblivious(), "Is not supported for non symmetric trees");

    THashMap<ui32, TString> cat_features_hash = MergeCatFeaturesHashToString(pool.ObjectsData.Get()[0]);

    TVector<TString> splits;

    int tree_num = model.ObliviousTrees.TreeStartOffsets.size();
    int tree_split_end;
    TVector<TModelSplit> bin_features = model.ObliviousTrees.GetBinFeatures();

    if (tree_idx + 1 < tree_num) {
        tree_split_end = model.ObliviousTrees.TreeStartOffsets[tree_idx + 1];
    } else {
        tree_split_end = model.ObliviousTrees.TreeSplits.size();
    }

    NCB::TFeaturesLayout featuresLayout = *(pool.MetaInfo.FeaturesLayout.Get());

    for (int split_idx = model.ObliviousTrees.TreeStartOffsets[tree_idx]; split_idx < tree_split_end; ++split_idx) {
        TModelSplit bin_feature = bin_features[model.ObliviousTrees.TreeSplits[split_idx]];
        TString feature_description = BuildDescription(featuresLayout, bin_feature);

        if (bin_feature.Type == ESplitType::OneHotFeature) {
            feature_description += cat_features_hash[(ui32)bin_feature.OneHotFeature.Value];
        }

        splits.push_back(feature_description);
    }

    return splits;
}

TVector<TString> GetTreeLeafValuesDescriptions(const TFullModel& model, int tree_idx, int leaves_num) {
    //TODO: support non symmetric trees
    CB_ENSURE(model.IsOblivious(), "Is not supported for non symmetric trees");

    int leaf_offset = 0;
    TVector<double> leaf_values;

    for (int idx = 0; idx < tree_idx; ++idx) {
        leaf_offset += (1uLL <<  model.ObliviousTrees.TreeSizes[idx]) * model.ObliviousTrees.ApproxDimension;
    }
    int tree_leaf_count = (1uLL <<  model.ObliviousTrees.TreeSizes[tree_idx]) * model.ObliviousTrees.ApproxDimension;

    for (int idx = 0; idx < tree_leaf_count; ++idx) {
        leaf_values.push_back(model.ObliviousTrees.LeafValues[leaf_offset + idx]);
    }

    std::reverse(leaf_values.begin(), leaf_values.end());

    TVector<TString> leaf_descriptions;
    int values_per_list = leaf_values.size() / leaves_num;

    for (int leaf_idx = 0; leaf_idx < leaves_num; ++leaf_idx) {
        TStringBuilder description;
        for (int internal_idx = 0; internal_idx < values_per_list; ++internal_idx) {
            double value = leaf_values[leaf_idx + internal_idx * leaves_num];
            description << "val = " << FloatToString(value, EFloatToStringMode::PREC_POINT_DIGITS, 3) << "\n";
        }
        leaf_descriptions.push_back(description);
    }

    return leaf_descriptions;
}
