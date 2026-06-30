#include "tree_print.h"

#include "projection.h"
#include "split.h"

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/model_dataset_compatibility.h>
#include <catboost/libs/data/objects.h>

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
    } else if (feature.Type == ESplitType::EstimatedFeature) {
        result << "estimated_" << (feature.IsOnlineEstimatedFeature ? "online" : "offline")
            << "_feature " << feature.FeatureIdx;
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
    } else if ((feature.Type == ESplitType::FloatFeature) || (feature.Type == ESplitType::EstimatedFeature)) {
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
        const auto ctrType = feature.OnlineCtr.Ctr.Base.CtrType;
        result << " counter_type=" << ctrType;
        result << " prior_numerator=" << feature.OnlineCtr.Ctr.PriorNum;
        if (ctrType == ECtrType::Buckets) {
            result << " target_bucket=" << feature.OnlineCtr.Ctr.TargetBorderIdx;
        }
    } else if (feature.Type == ESplitType::FloatFeature) {
        result << BuildFeatureDescription(layout, feature.FloatFeature.FloatFeature, EFeatureType::Float);
    } else if (feature.Type == ESplitType::EstimatedFeature) {
        const TModelEstimatedFeature& split = feature.EstimatedFeature.ModelEstimatedFeature;
        result << " src_feature_id=" << split.SourceFeatureId;
        result << " calcer_id=" << split.CalcerId;
        result << " local_id=" << split.LocalId;
    } else {
        Y_ASSERT(feature.Type == ESplitType::OneHotFeature);
        result << BuildFeatureDescription(layout, feature.OneHotFeature.CatFeatureIdx, EFeatureType::Categorical);
    }

    if (feature.Type == ESplitType::OnlineCtr) {
        result << ", border=" << feature.OnlineCtr.Border;
    } else if (feature.Type == ESplitType::FloatFeature) {
        result << ", bin=" << feature.FloatFeature.Split;
    } else if (feature.Type == ESplitType::EstimatedFeature) {
        result << ", bin=" << feature.EstimatedFeature.Split;
    } else {
        Y_ASSERT(feature.Type == ESplitType::OneHotFeature);
        result << ", value=";
    }
    return result;
}

// utility function for python_package/catboost/core.py plot_tree function
TVector<TString> GetTreeSplitsDescriptions(const TFullModel& model, size_t treeIdx, const NCB::TDataProviderPtr pool) {
    CB_ENSURE(treeIdx < model.GetTreeCount(),
        "Requested tree splits description for tree " << treeIdx << ", but model has " << model.GetTreeCount());

    if (pool) {
        CheckModelAndDatasetCompatibility(model, *pool->ObjectsData.Get());
    }

    TVector<TString> splits;

    const auto binFeatures = model.ModelTrees->GetBinFeatures();

    size_t treeSplitEnd = (treeIdx + 1 < model.GetTreeCount())
        ? model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIdx + 1]
        : model.ModelTrees->GetModelTreeData()->GetTreeSplits().size();

    THashMap<ui32, TString> catFeaturesHash;
    NCB::TFeaturesLayout featuresLayout;
    if (pool) {
        catFeaturesHash = MergeCatFeaturesHashToString(pool.Get()->ObjectsData.Get()[0]);
        featuresLayout = *(pool.Get()->MetaInfo.FeaturesLayout.Get());
    } else {
        TVector<ui32> catFeaturesExternalIndexes;
        for (const auto& feature: model.ModelTrees->GetCatFeatures()) {
            catFeaturesExternalIndexes.push_back(feature.Position.FlatIndex);
        }
        featuresLayout = NCB::TFeaturesLayout(model.GetNumFloatFeatures() + model.GetNumCatFeatures(), catFeaturesExternalIndexes, {}, {}, {}, false);
    }

    for (size_t splitIdx = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIdx]; splitIdx < treeSplitEnd; ++splitIdx) {
        TModelSplit binFeature = binFeatures[model.ModelTrees->GetModelTreeData()->GetTreeSplits()[splitIdx]];
        TString featureDescription = BuildDescription(featuresLayout, binFeature);

        if (binFeature.Type == ESplitType::OneHotFeature) {
            CB_ENSURE(pool,
                "Please pass training dataset to plot_tree function, "
                "training dataset is required if categorical features are present in the model.");
            featureDescription += catFeaturesHash[(ui32)binFeature.OneHotFeature.Value];
        }

        splits.push_back(featureDescription);
    }

    return splits;
}

TVector<TString> GetTreeLeafValuesDescriptions(const TFullModel& model, size_t treeIdx) {
    CB_ENSURE(treeIdx < model.GetTreeCount(),
        "Requested tree leaf values for tree " << treeIdx << ", but model has " << model.GetTreeCount());
    auto applyData = model.ModelTrees->GetApplyData();
    size_t leafOffset = applyData->TreeFirstLeafOffsets[treeIdx];
    size_t nextTreeLeafOffset = (treeIdx + 1 < model.GetTreeCount())
        ? applyData->TreeFirstLeafOffsets[treeIdx + 1]
        : model.ModelTrees->GetModelTreeData()->GetLeafValues().size();

    TVector<double> leafValues(
        model.ModelTrees->GetModelTreeData()->GetLeafValues().begin() + leafOffset,
        model.ModelTrees->GetModelTreeData()->GetLeafValues().begin() + nextTreeLeafOffset);

    TVector<TString> leafDescriptions;

    size_t leavesNum = (nextTreeLeafOffset - leafOffset) / model.GetDimensionsCount();
    for (size_t leafIdx = 0; leafIdx < leavesNum; ++leafIdx) {
        TStringBuilder description;
        for (size_t dim = 0; dim < model.GetDimensionsCount(); ++dim) {
            double value = leafValues[leafIdx * model.GetDimensionsCount() + dim];
            description << "val = " << FloatToString(value, EFloatToStringMode::PREC_POINT_DIGITS, 3) << "\n";
        }
        leafDescriptions.push_back(description);
    }

    return leafDescriptions;
}

TConstArrayRef<TNonSymmetricTreeStepNode> GetTreeStepNodes(const TFullModel& model, size_t treeIdx) {
    CB_ENSURE(treeIdx < model.GetTreeCount(),
        "Requested tree step nodes for tree " << treeIdx << ", but model has " << model.GetTreeCount());
    Y_ASSERT(!model.IsOblivious());
    const size_t offset = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
    const auto start = model.ModelTrees->GetModelTreeData()->GetNonSymmetricStepNodes().begin() + offset;
    const auto end = start + model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx];
    return TConstArrayRef<TNonSymmetricTreeStepNode>(start, end);
}

TVector<ui32> GetTreeNodeToLeaf(const TFullModel& model, size_t treeIdx) {
    CB_ENSURE(treeIdx < model.GetTreeCount(),
        "Requested tree leaves for nodes for tree " << treeIdx << ", but model has " << model.GetTreeCount());
    Y_ASSERT(!model.IsOblivious());
    const size_t offset = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
    const auto start = model.ModelTrees->GetModelTreeData()->GetNonSymmetricNodeIdToLeafId().begin() + offset;
    const auto end = start + model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx];
    auto applyData = model.ModelTrees->GetApplyData();
    const size_t firstLeafOffset = applyData->TreeFirstLeafOffsets[treeIdx];
    const size_t dimensionsCount = model.GetDimensionsCount();
    TVector<ui32> nodeToLeaf(start, end);
    for (auto& value : nodeToLeaf) {
        value = (value - firstLeafOffset) / dimensionsCount;
    }
    return nodeToLeaf;
}
