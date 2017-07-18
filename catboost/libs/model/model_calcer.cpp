#include "model_calcer.h"

#include <set>

namespace NCatBoost {

void TFullModelCalcer::BuildBinTrees() {
    std::set<TSplit> allSplits;
    for (const auto& treeStruct : Model.TreeStruct) {
        allSplits.insert(treeStruct.SelectedSplits.begin(), treeStruct.SelectedSplits.end());
    }
    yvector<TSplit> usedSplits;
    usedSplits.assign(allSplits.begin(), allSplits.end());
    UsedBinaryFeaturesCount = usedSplits.size();
    yhash<TSplit, int, TSplitHash> binFeatureIndexes;
    for (int i = 0; i < usedSplits.ysize(); ++i) {
        Y_ENSURE (usedSplits[i].Type == ESplitType::FloatFeature, "cat features unimplemented for now");
        binFeatureIndexes[usedSplits[i]] = i;
    }
    BinaryTrees.reserve(Model.TreeStruct.size());
    for (const auto& treeStruct : Model.TreeStruct) {
        yvector<int> binFeaturesTree;
        binFeaturesTree.reserve(treeStruct.SelectedSplits.size());
        for (const auto& split: treeStruct.SelectedSplits) {
            binFeaturesTree.push_back(binFeatureIndexes[split]);
        }
        BinaryTrees.emplace_back(std::move(binFeaturesTree));
    }
    for (const auto& split : usedSplits) {
        if (split.Type == ESplitType::FloatFeature) {
            if (UsedFloatFeatures.empty() || UsedFloatFeatures.back().FeatureIndex != split.BinFeature.FloatFeature) {
                UsedFloatFeatures.emplace_back();
                UsedFloatFeatures.back().FeatureIndex = split.BinFeature.FloatFeature;
            }
            UsedFloatFeatures.back().Borders.push_back(
                Model.Borders[split.BinFeature.FloatFeature][split.BinFeature.SplitIdx]);
        } else {
            if (UsedCtrFeatures.empty() || UsedCtrFeatures.back().Ctr != split.OnlineCtr.Ctr) {
                UsedCtrFeatures.emplace_back();
                UsedCtrFeatures.back().Ctr = split.OnlineCtr.Ctr;
            }
            UsedCtrFeatures.back().Borders.push_back(split.OnlineCtr.Border);
        }
    }
}

void TFullModelCalcer::Save(TOutputStream* out) {
    Model.Save(out);
}

void TFullModelCalcer::Load(TInputStream* in) {
    Model.Load(in);
}
}
