#include "model_build_helper.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/algorithm.h>
#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/xrange.h>
#include <util/generic/ylimits.h>

TCommonModelBuilderHelper::TCommonModelBuilderHelper(
    const TVector<TFloatFeature>& allFloatFeatures,
    const TVector<TCatFeature>& allCategoricalFeatures,
    const TVector<TTextFeature>& allTextFeatures,
    const TVector<TEmbeddingFeature>& allEmbeddingFeatures,
    int approxDimension
)
    : ApproxDimension(approxDimension)
    , FloatFeatures(allFloatFeatures)
    , CatFeatures(allCategoricalFeatures)
    , TextFeatures(allTextFeatures)
    , EmbeddingFeatures(allEmbeddingFeatures)
{
    if (!FloatFeatures.empty()) {
        CB_ENSURE(IsSorted(FloatFeatures.begin(), FloatFeatures.end(),
                           [](const TFloatFeature& f1, const TFloatFeature& f2) {
                               return f1.Position.FlatIndex < f2.Position.FlatIndex;
                           }),
                  "Float features should be sorted"
        );
        FloatFeaturesInternalIndexesMap.resize((size_t)FloatFeatures.back().Position.Index + 1, Max<size_t>());
        for (auto i : xrange(FloatFeatures.size())) {
            FloatFeaturesInternalIndexesMap.at((size_t)FloatFeatures[i].Position.Index) = i;
        }
    }
    if (!CatFeatures.empty()) {
        CB_ENSURE(IsSorted(CatFeatures.begin(), CatFeatures.end(),
                           [](const TCatFeature& f1, const TCatFeature& f2) {
                               return f1.Position.FlatIndex < f2.Position.FlatIndex;
                           }),
                  "Cat features should be sorted"
        );
        CatFeaturesInternalIndexesMap.resize((size_t)CatFeatures.back().Position.Index + 1, Max<size_t>());
        for (auto i : xrange(CatFeatures.size())) {
            CatFeaturesInternalIndexesMap.at((size_t)CatFeatures[i].Position.Index) = i;
        }
    }
    if (!TextFeatures.empty()) {
        CB_ENSURE(
            IsSorted(
                TextFeatures.begin(),
                TextFeatures.end(),
                [](const TTextFeature& f1, const TTextFeature& f2) {
                    return f1.Position.FlatIndex < f2.Position.FlatIndex;
                }
            ),
            "Text features should be sorted"
        );
        TextFeaturesInternalIndexesMap.resize((size_t)TextFeatures.back().Position.Index + 1, Max<size_t>());
        for (auto i : xrange(TextFeatures.size())) {
            TextFeaturesInternalIndexesMap.at((size_t)TextFeatures[i].Position.Index) = i;
        }
    }
    if (!EmbeddingFeatures.empty()) {
        CB_ENSURE(
            IsSorted(
                EmbeddingFeatures.begin(),
                EmbeddingFeatures.end(),
                [](const TEmbeddingFeature& f1, const TEmbeddingFeature& f2) {
                    return f1.Position.FlatIndex < f2.Position.FlatIndex;
                }
            ),
            "Embedding features should be sorted"
        );
        EmbeddingFeaturesInternalIndexesMap.resize((size_t)EmbeddingFeatures.back().Position.Index + 1, Max<size_t>());
        for (auto i : xrange(EmbeddingFeatures.size())) {
            EmbeddingFeaturesInternalIndexesMap.at((size_t)EmbeddingFeatures[i].Position.Index) = i;
        }
    }
}

template <class T>
static void MakeFeaturesUnused(TArrayRef<T> features) {
    for (auto& feature : features) {
        feature.SetUsedInModel(false);
    }
}

template <class T>
static void MarkUsedFeatures(
    const THashSet<int>& usedFeatureIds,
    TConstArrayRef<size_t> InternalIndexesMap,
    TArrayRef<T> features
) {
    for (int usedFeatureIdx : usedFeatureIds) {
        features[InternalIndexesMap.at(usedFeatureIdx)].SetUsedInModel(true);
    }
}

void TCommonModelBuilderHelper::ProcessSplitsSet(const TSet<TModelSplit>& modelSplitSet, TModelTrees* trees) {
    trees->SetApproxDimension(ApproxDimension);
    for (auto& feature : FloatFeatures) {
        feature.Borders.clear();
    }
    trees->SetFloatFeatures(std::move(FloatFeatures));

    MakeFeaturesUnused(MakeArrayRef(CatFeatures.begin(), CatFeatures.end()));
    MakeFeaturesUnused(MakeArrayRef(TextFeatures.begin(), TextFeatures.end()));
    MakeFeaturesUnused(MakeArrayRef(EmbeddingFeatures.begin(), EmbeddingFeatures.end()));

    trees->SetCatFeatures(std::move(CatFeatures));
    trees->SetTextFeatures(std::move(TextFeatures));
    trees->SetEmbeddingFeatures(std::move(EmbeddingFeatures));

    trees->ProcessSplitsSet(modelSplitSet,
                            FloatFeaturesInternalIndexesMap,
                            CatFeaturesInternalIndexesMap,
                            TextFeaturesInternalIndexesMap,
                            EmbeddingFeaturesInternalIndexesMap);
    for (const auto& split : modelSplitSet) {
        const int binFeatureIdx = BinFeatureIndexes.ysize();
        Y_ASSERT(!BinFeatureIndexes.contains(split));
        BinFeatureIndexes[split] = binFeatureIdx;
    }
    Y_ASSERT(modelSplitSet.size() == BinFeatureIndexes.size());
}


TObliviousTreeBuilder::TObliviousTreeBuilder(
    const TVector<TFloatFeature>& allFloatFeatures,
    const TVector<TCatFeature>& allCategoricalFeatures,
    const TVector<TTextFeature>& allTextFeatures,
    const TVector<TEmbeddingFeature>& allEmbeddingFeatures,
    int approxDimension)
    : TCommonModelBuilderHelper(
        allFloatFeatures,
        allCategoricalFeatures,
        allTextFeatures,
        allEmbeddingFeatures,
        approxDimension)
{
}

void TObliviousTreeBuilder::AddTree(const TVector<TModelSplit>& modelSplits,
                                    const TVector<TVector<double>>& treeLeafValues,
                                    TConstArrayRef<double> treeLeafWeights
) {
    CB_ENSURE(ApproxDimension == treeLeafValues.ysize());
    auto leafCount = treeLeafValues.at(0).size();

    TVector<double> leafValues;
    leafValues.yresize(ApproxDimension * leafCount);

    for (size_t dimension = 0; dimension < treeLeafValues.size(); ++dimension) {
        CB_ENSURE(treeLeafValues[dimension].size() == (1ull << modelSplits.size()));
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
    CB_ENSURE((1ull << modelSplits.size()) * ApproxDimension == treeLeafValues.size());
    LeafValues.insert(LeafValues.end(), treeLeafValues.begin(), treeLeafValues.end());
    if (!treeLeafWeights.empty()) {
        CB_ENSURE((1ull << modelSplits.size()) == treeLeafWeights.size());
        LeafWeights.insert(LeafWeights.end(), treeLeafWeights.begin(), treeLeafWeights.end());
    }
    Trees.emplace_back(modelSplits);
}

void TObliviousTreeBuilder::Build(TModelTrees* result) {
    *result = TModelTrees{};
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
    ProcessSplitsSet(modelSplitSet, result);
    result->SetLeafValues(std::move(LeafValues));
    result->SetLeafWeights(std::move(LeafWeights));
    for (const auto& treeStruct : Trees) {
        for (const auto& split : treeStruct) {
            result->AddTreeSplit(BinFeatureIndexes.at(split));
        }
        result->AddTreeSize(treeStruct.ysize());
    }
    result->UpdateRuntimeData();
}

void TNonSymmetricTreeModelBuilder::AddTree(THolder<TNonSymmetricTreeNode> head) {
    auto prevSize = FlatSplitsVector.size();
    TreeStartOffsets.push_back(prevSize);
    AddTreeNode(*head);
    TreeSizes.push_back(FlatSplitsVector.size() - prevSize);
}

void TNonSymmetricTreeModelBuilder::Build(TModelTrees* result) {
    *result = TModelTrees{};
    ProcessSplitsSet(ModelSplitSet, result);
    Y_ASSERT(FlatSplitsVector.size() == FlatNodeValueIndexes.size());
    Y_ASSERT(FlatNodeValueIndexes.size() == FlatNonSymmetricStepNodes.size());
    Y_ASSERT(LeafWeights.empty() || LeafWeights.size() == FlatValueVector.size() / ApproxDimension);
    result->SetNonSymmetricStepNodes(std::move(FlatNonSymmetricStepNodes));
    result->SetNonSymmetricNodeIdToLeafId(std::move(FlatNodeValueIndexes));
    result->SetLeafValues(std::move(FlatValueVector));
    for (const auto& split : FlatSplitsVector) {
        if (split) {
            result->AddTreeSplit(BinFeatureIndexes.at(*split));
        } else {
            result->AddTreeSplit(0);
        }
    }
    result->SetTreeSizes(std::move(TreeSizes));
    result->SetTreeStartOffsets(std::move(TreeStartOffsets));
    result->SetLeafWeights(std::move(LeafWeights));
    result->UpdateRuntimeData();
}

TNonSymmetricTreeModelBuilder::TNonSymmetricTreeModelBuilder(
    const TVector<TFloatFeature>& allFloatFeatures,
    const TVector<TCatFeature>& allCategoricalFeatures,
    const TVector<TTextFeature>& allTextFeatures,
    const TVector<TEmbeddingFeature>& allEmbeddingFeatures,
    int approxDimension
)
    : TCommonModelBuilderHelper(allFloatFeatures,
                                allCategoricalFeatures,
                                allTextFeatures,
                                allEmbeddingFeatures,
                                approxDimension)
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
            const auto leftId = AddTreeNode(*node.Left);
            const auto rightId = AddTreeNode(*node.Right);
            FlatNonSymmetricStepNodes[stepNodeId] = TNonSymmetricTreeStepNode{
                static_cast<ui16>(leftId - nodeId),
                static_cast<ui16>(rightId - nodeId)
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
    if (std::holds_alternative<double>(node.Value)) {
        CB_ENSURE(ApproxDimension == 1, "got single value for multidimensional model");
        FlatValueVector.emplace_back(std::get<double>(node.Value));
    } else {
        const auto& valueVector = std::get<TVector<double>>(node.Value);
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
