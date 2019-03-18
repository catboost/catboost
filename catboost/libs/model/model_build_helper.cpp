#include "model_build_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>

TCommonModelBuilderHelper::TCommonModelBuilderHelper(
    const TVector<TFloatFeature> &allFloatFeatures,
    const TVector<TCatFeature> &allCategoricalFeatures,
    int approxDimension
)
    : ApproxDimension(approxDimension)
    , FloatFeatures(allFloatFeatures)
    , CatFeatures(allCategoricalFeatures)
{
    if (!FloatFeatures.empty()) {
        CB_ENSURE(IsSorted(FloatFeatures.begin(), FloatFeatures.end(),
                           [](const TFloatFeature& f1, const TFloatFeature& f2) {
                               return f1.FeatureId < f2.FeatureId && f1.FlatFeatureIndex < f2.FlatFeatureIndex;
                           }),
                  "Float features should be sorted"
        );
        FloatFeaturesInternalIndexesMap.resize((size_t)FloatFeatures.back().FeatureIndex + 1, Max<size_t>());
        for (auto i : xrange(FloatFeatures.size())) {
            FloatFeaturesInternalIndexesMap.at((size_t)FloatFeatures[i].FeatureIndex) = i;
        }
    }
    if (!CatFeatures.empty()) {
        CB_ENSURE(IsSorted(CatFeatures.begin(), CatFeatures.end(),
                           [](const TCatFeature& f1, const TCatFeature& f2) {
                               return f1.FeatureId < f2.FeatureId && f1.FlatFeatureIndex < f2.FlatFeatureIndex;
                           }),
                  "Cat features should be sorted"
        );
        CatFeaturesInternalIndexesMap.resize((size_t)CatFeatures.back().FeatureIndex + 1, Max<size_t>());
        for (auto i : xrange(CatFeatures.size())) {
            CatFeaturesInternalIndexesMap.at((size_t)CatFeatures[i].FeatureIndex) = i;
        }
    }
}

void TCommonModelBuilderHelper::ProcessSplitsSet(const TSet<TModelSplit>& modelSplitSet, TObliviousTrees* tree) {
    tree->ApproxDimension = ApproxDimension;
    tree->CatFeatures = CatFeatures;
    tree->FloatFeatures = FloatFeatures;
    for (auto& feature : tree->FloatFeatures) {
        feature.Borders.clear();
    }
    for (auto& feature : tree->CatFeatures) {
        feature.UsedInModel = false;
    }
    THashSet<int> usedCatFeatureIndexes;
    for (const auto& split : modelSplitSet) {
        if (split.Type == ESplitType::FloatFeature) {
            const size_t internalFloatIndex = FloatFeaturesInternalIndexesMap.at((size_t)split.FloatFeature.FloatFeature);
            tree->FloatFeatures.at(internalFloatIndex).Borders.push_back(split.FloatFeature.Split);
        } else if (split.Type == ESplitType::OneHotFeature) {
            usedCatFeatureIndexes.insert(split.OneHotFeature.CatFeatureIdx);
            if (tree->OneHotFeatures.empty() || tree->OneHotFeatures.back().CatFeatureIndex != split.OneHotFeature.CatFeatureIdx) {
                auto& ref = tree->OneHotFeatures.emplace_back();
                ref.CatFeatureIndex = split.OneHotFeature.CatFeatureIdx;
            }
            tree->OneHotFeatures.back().Values.push_back(split.OneHotFeature.Value);
        } else {
            const auto& projection = split.OnlineCtr.Ctr.Base.Projection;
            usedCatFeatureIndexes.insert(projection.CatFeatures.begin(), projection.CatFeatures.end());
            if (tree->CtrFeatures.empty() || tree->CtrFeatures.back().Ctr != split.OnlineCtr.Ctr) {
                tree->CtrFeatures.emplace_back();
                tree->CtrFeatures.back().Ctr = split.OnlineCtr.Ctr;
            }
            tree->CtrFeatures.back().Borders.push_back(split.OnlineCtr.Border);
        }
    }
    for (auto usedCatFeatureIdx : usedCatFeatureIndexes) {
        tree->CatFeatures[CatFeaturesInternalIndexesMap.at(usedCatFeatureIdx)].UsedInModel = true;
    }
    for (const auto& split : modelSplitSet) {
        const int binFeatureIdx = BinFeatureIndexes.ysize();
        Y_ASSERT(!BinFeatureIndexes.contains(split));
        BinFeatureIndexes[split] = binFeatureIdx;
    }
    Y_ASSERT(modelSplitSet.size() == BinFeatureIndexes.size());
}


TObliviousTreeBuilder::TObliviousTreeBuilder(const TVector<TFloatFeature>& allFloatFeatures, const TVector<TCatFeature>& allCategoricalFeatures, int approxDimension)
    : TCommonModelBuilderHelper(allFloatFeatures, allCategoricalFeatures, approxDimension)
{
}

void TObliviousTreeBuilder::AddTree(const TVector<TModelSplit>& modelSplits,
                                    const TVector<TVector<double>>& treeLeafValues,
                                    TConstArrayRef<double> treeLeafWeights
) {
    CB_ENSURE(ApproxDimension == treeLeafValues.ysize());
    auto leafCount = treeLeafValues.at(0).size();

    TVector<double> leafValues(ApproxDimension * leafCount);

    for (size_t dimension = 0; dimension < treeLeafValues.size(); ++dimension) {
        CB_ENSURE(treeLeafValues[dimension].size() == (1u << modelSplits.size()));
        for (size_t leafId = 0; leafId < leafCount; ++leafId) {
            leafValues[leafId * ApproxDimension + dimension] = treeLeafValues[dimension][leafId];
        }
    }
    AddTree(modelSplits, leafValues, treeLeafWeights);
}

void TObliviousTreeBuilder::AddTree(const TVector<TModelSplit>& modelSplits,
                                    TConstArrayRef<double> treeLeafValues,
                                    TConstArrayRef<double> treeLeafWeights
) {
    CB_ENSURE((1u << modelSplits.size()) * ApproxDimension == treeLeafValues.size());
    LeafValues.insert(LeafValues.end(), treeLeafValues.begin(), treeLeafValues.end());
    if (!treeLeafWeights.empty()) {
        LeafWeights.push_back(TVector<double>(treeLeafWeights.begin(), treeLeafWeights.end()));
    }
    Trees.emplace_back(modelSplits);
}

TObliviousTrees TObliviousTreeBuilder::Build() {
    TSet<TModelSplit> modelSplitSet;
    for (const auto& tree : Trees) {
        for (const auto& split : tree) {
            modelSplitSet.insert(split);
            if (split.Type == ESplitType::OnlineCtr) {
                auto& proj = split.OnlineCtr.Ctr.Base.Projection;
                for (const auto& binF : proj.BinFeatures) {
                    modelSplitSet.insert(TModelSplit(binF));
                }
                for (const auto& oheFeature : proj.OneHotFeatures) {
                    modelSplitSet.insert(TModelSplit(oheFeature));
                }
            }
        }
    }
    // filling binary tree splits
    TObliviousTrees result;
    ProcessSplitsSet(modelSplitSet, &result);
    result.LeafValues = LeafValues;
    result.LeafWeights = LeafWeights;
    for (const auto& treeStruct : Trees) {
        for (const auto& split : treeStruct) {
            result.TreeSplits.push_back(BinFeatureIndexes.at(split));
        }
        if (result.TreeStartOffsets.empty()) {
            result.TreeStartOffsets.push_back(0);
        } else {
            result.TreeStartOffsets.push_back(result.TreeStartOffsets.back() + result.TreeSizes.back());
        }
        result.TreeSizes.push_back(treeStruct.ysize());
    }
    result.UpdateMetadata();
    return result;
}

void TNonSymmetricTreeModelBuilder::AddTree(THolder<TNonSymmetricTreeNode> head) {
    auto prevSize = FlatSplitsVector.size();
    TreeStartOffsets.push_back(prevSize);
    AddTreeNode(*head);
    TreeSizes.push_back(FlatSplitsVector.size() - prevSize);
}

TObliviousTrees TNonSymmetricTreeModelBuilder::Build() {
    TObliviousTrees result;
    ProcessSplitsSet(ModelSplitSet, &result);
    Y_ASSERT(FlatSplitsVector.size() == FlatNodeValueIndexes.size());
    Y_ASSERT(FlatNodeValueIndexes.size() == FlatNonSymmetricStepNodes.size());
    Y_ASSERT(LeafWeights.empty() || LeafWeights.size() == FlatValueVector.size() / ApproxDimension);
    result.NonSymmetricStepNodes = std::move(FlatNonSymmetricStepNodes);
    result.NonSymmetricNodeIdToLeafId = std::move(FlatNodeValueIndexes);
    result.LeafValues = std::move(FlatValueVector);
    for (const auto& split : FlatSplitsVector) {
        if (split) {
            result.TreeSplits.push_back(BinFeatureIndexes.at(*split));
        } else {
            result.TreeSplits.push_back(0);
        }
    }
    result.TreeSizes = std::move(TreeSizes);
    result.TreeStartOffsets = std::move(TreeStartOffsets);
    result.LeafWeights.emplace_back(std::move(LeafWeights));
    result.UpdateMetadata();
    return result;
}

TNonSymmetricTreeModelBuilder::TNonSymmetricTreeModelBuilder(
    const TVector<TFloatFeature>& allFloatFeatures,
    const TVector<TCatFeature>& allCategoricalFeatures,
    int approxDimension
)
    : TCommonModelBuilderHelper(allFloatFeatures, allCategoricalFeatures, approxDimension)
{}

ui32 TNonSymmetricTreeModelBuilder::AddTreeNode(const TNonSymmetricTreeNode& node) {
    const ui32 nodeId = FlatNonSymmetricStepNodes.size();
    node.Validate();
    if (node.IsSplitNode()) {
        ModelSplitSet.insert(*node.SplitCondition);
        FlatSplitsVector.emplace_back(*node.SplitCondition);
        auto stepNodeId = FlatNonSymmetricStepNodes.size();
        FlatNonSymmetricStepNodes.emplace_back();
        if (node.Left->IsSplitNode() == node.Right->IsSplitNode()) {
            FlatNodeValueIndexes.emplace_back(Max<ui32>());
            FlatNonSymmetricStepNodes[stepNodeId] = TNonSymmetricTreeStepNode{
                static_cast<ui16>(AddTreeNode(*node.Left) - nodeId),
                static_cast<ui16>(AddTreeNode(*node.Right) - nodeId)
            };
        } else {
            if (node.Right->IsSplitNode()) {
                InsertNodeValue(*node.Left);
                FlatNonSymmetricStepNodes[stepNodeId] = TNonSymmetricTreeStepNode{
                    0,
                    static_cast<ui16>(AddTreeNode(*node.Right) - nodeId)
                };
            } else {
                InsertNodeValue(*node.Right);
                FlatNonSymmetricStepNodes[stepNodeId] = TNonSymmetricTreeStepNode{
                    static_cast<ui16>(AddTreeNode(*node.Left) - nodeId),
                    0
                };
            }
        }
    } else {
        FlatSplitsVector.emplace_back();
        FlatNonSymmetricStepNodes.emplace_back(TNonSymmetricTreeStepNode{0, 0});
        InsertNodeValue(node);
    }
    return nodeId;
}

void TNonSymmetricTreeModelBuilder::InsertNodeValue(const TNonSymmetricTreeNode& node) {
    FlatNodeValueIndexes.emplace_back(FlatValueVector.size());
    if (HoldsAlternative<double>(node.Value)) {
        CB_ENSURE(ApproxDimension == 1, "got single value for multidimensional model");
        FlatValueVector.emplace_back(Get<double>(node.Value));
    } else {
        const auto& valueVector = Get<TVector<double>>(node.Value);
        CB_ENSURE(ApproxDimension == static_cast<int>(valueVector.size())
            , "Different model approx dimension and value dimensions");
        for (const auto& value : valueVector) {
            FlatValueVector.emplace_back(value);
        }
    }
    if (node.NodeWeight) {
        LeafWeights.push_back(*node.NodeWeight);
    }
}
