#pragma once

#include "model.h"

#include <util/generic/array_ref.h>
#include <util/generic/set.h>

class TCommonModelBuilderHelper {
public:
    TCommonModelBuilderHelper(
        const TVector<TFloatFeature>& allFloatFeatures,
        const TVector<TCatFeature>& allCategoricalFeatures,
        const TVector<TTextFeature>& allTextFeatures,
        const TVector<TEmbeddingFeature>& allEmbeddingFeatures,
        int approxDimension);
    void ProcessSplitsSet(const TSet<TModelSplit>& modelSplitSet, TModelTrees* tree);
public:
    int ApproxDimension = 1;
    TVector<TFloatFeature> FloatFeatures;
    TVector<size_t> FloatFeaturesInternalIndexesMap;
    TVector<TCatFeature> CatFeatures;
    TVector<size_t> CatFeaturesInternalIndexesMap;
    TVector<TTextFeature> TextFeatures;
    TVector<size_t> TextFeaturesInternalIndexesMap;
    TVector<TEmbeddingFeature> EmbeddingFeatures;
    TVector<size_t> EmbeddingFeaturesInternalIndexesMap;
    THashMap<TModelSplit, int> BinFeatureIndexes;
};

class TObliviousTreeBuilder : private TCommonModelBuilderHelper {
public:
    TObliviousTreeBuilder(
        const TVector<TFloatFeature>& allFloatFeatures,
        const TVector<TCatFeature>& allCategoricalFeatures,
        const TVector<TTextFeature>& allTextFeatures,
        const TVector<TEmbeddingFeature>& allEmbeddingFeatures,
        int approxDimension
    );
    void AddTree(
        const TVector<TModelSplit>& modelSplits,
        const TVector<TVector<double>>& treeLeafValues,
        TConstArrayRef<double> treeLeafWeights);
    void AddTree(
        const TVector<TModelSplit>& modelSplits,
        TConstArrayRef<double> treeLeafValues,
        TConstArrayRef<double> treeLeafWeights);
    void AddTree(
        const TVector<TModelSplit>& modelSplits,
        const TVector<TVector<double>>& treeLeafValues) {

        AddTree(modelSplits, treeLeafValues, TVector<double>());
    }
    void Build(TModelTrees* result);
private:
    TVector<TVector<TModelSplit>> Trees;
    TVector<double> LeafValues;
    TVector<double> LeafWeights;
};

class TNonSymmetricTreeNode {
public:
    struct TEmptyValue {};
    TMaybe<TModelSplit> SplitCondition;
    THolder<TNonSymmetricTreeNode> Left;
    THolder<TNonSymmetricTreeNode> Right;
    std::variant<TEmptyValue, double, TVector<double>> Value;
    TMaybe<double> NodeWeight;

    bool IsSplitNode() const {
        return (bool)SplitCondition;
    }

    void Validate() const {
        if (IsSplitNode()) {
            CB_ENSURE(Left && Right, "Split node should contain both left and right nodes");
            CB_ENSURE(std::holds_alternative<TEmptyValue>(Value), "Split node must hold empty value");
        } else {
            CB_ENSURE(!Left && !Right, "Value node should have no child leafs");
            CB_ENSURE(!std::holds_alternative<TEmptyValue>(Value), "Split node must hold empty value");
        }
    }
};

class TNonSymmetricTreeModelBuilder : private TCommonModelBuilderHelper {
public:
    TNonSymmetricTreeModelBuilder(
        const TVector<TFloatFeature>& allFloatFeatures,
        const TVector<TCatFeature>& allCategoricalFeatures,
        const TVector<TTextFeature>& allTextFeatures,
        const TVector<TEmbeddingFeature>& allEmbeddingFeatures,
        int approxDimension
    );
    void AddTree(THolder<TNonSymmetricTreeNode> head);
    void Build(TModelTrees* result);
private:
    ui32 AddTreeNode(const TNonSymmetricTreeNode& node);
    void InsertNodeValue(const TNonSymmetricTreeNode& node);
private:
    TSet<TModelSplit> ModelSplitSet;
    TVector<int> TreeSizes;
    TVector<int> TreeStartOffsets;
    TVector<TMaybe<TModelSplit>> FlatSplitsVector;
    TVector<double> FlatValueVector;
    TVector<double> LeafWeights;
    TVector<ui32> FlatNodeValueIndexes;
    TVector<TNonSymmetricTreeStepNode> FlatNonSymmetricStepNodes;
};
