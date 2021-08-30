#pragma once

#include <catboost/libs/model/model.h>
#include <catboost/private/libs/algo/tree_print.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>
#include <util/system/types.h>

struct TMxTree {
    struct TValsInLeaf {
        TVector<double> Vals;
    };

    TVector<int> SrcFeatures;
    TVector<TValsInLeaf> Leaves;
};

struct TNodeInfo {
    TVector<double> Values; // [dimension]
    double Count = 0;
};

struct TTriangleNodes {
    int ParentIdx;
    int LeftChildIdx;
    int RightChildIdx;

    int SplitfeatureIdx;
};

struct TFeature {
    ESplitType Type;
    int FeatureIdx;
    TModelCtr Ctr;
    TModelEstimatedFeature EstimatedFeature;
    EFeatureCalcerType FeatureCalcerType;
    static constexpr size_t FloatFeatureBaseHash = 12321;
    static constexpr size_t CtrBaseHash = 89321;
    static constexpr size_t OneHotFeatureBaseHash = 517931;
    static constexpr size_t EstimatedFeatureBaseHash = 2123719;

public:
    TFeature() = default;
    TFeature(const TFloatFeature& feature) : Type(ESplitType::FloatFeature), FeatureIdx(feature.Position.Index) {}
    TFeature(const TOneHotFeature& feature) : Type(ESplitType::OneHotFeature), FeatureIdx(feature.CatFeatureIndex) {}
    TFeature(const TCtrFeature& feature) : Type(ESplitType::OnlineCtr), Ctr(feature.Ctr) {}
    TFeature(const TEstimatedFeature& feature, EFeatureCalcerType featureCalcerType)
        : Type(ESplitType::EstimatedFeature)
        , EstimatedFeature(feature.ModelEstimatedFeature)
        , FeatureCalcerType(featureCalcerType)
    {}

    bool operator==(const TFeature& other) const {
        if (Type != other.Type) {
            return false;
        }
        if (Type == ESplitType::OnlineCtr) {
            return Ctr == other.Ctr;
        } else if (Type == ESplitType::EstimatedFeature) {
            return EstimatedFeature == other.EstimatedFeature && FeatureCalcerType == other.FeatureCalcerType;
        }
        else {
            return FeatureIdx == other.FeatureIdx;
        }
    }

    bool operator!=(const TFeature& other) const {
        return !(*this == other);
    }

    size_t GetHash() const {
        switch(Type) {
            case ESplitType::FloatFeature:
                return MultiHash(FloatFeatureBaseHash, FeatureIdx);
            case ESplitType::OneHotFeature:
                return MultiHash(OneHotFeatureBaseHash, FeatureIdx);
            case ESplitType::OnlineCtr:
                return MultiHash(CtrBaseHash, Ctr.GetHash());
            case ESplitType::EstimatedFeature:
                return MultiHash(EstimatedFeatureBaseHash, EstimatedFeature.SourceFeatureId, EstimatedFeature.LocalId, EstimatedFeature.SourceFeatureType, FeatureCalcerType);
            default:
                CB_ENSURE(false, "Unsupported split type " << Type);
        }
    }
    TString BuildDescription(const NCB::TFeaturesLayout& layout) const;
};

struct TFeatureHash {
    size_t operator()(const TFeature& f) const {
        return f.GetHash();
    }
};

TFeature GetFeature(const TFullModel& model, const TModelSplit& split);

int GetMaxSrcFeature(const TVector<TMxTree>& trees);

void ConvertToPercents(TVector<double>& res);

TVector<double> CalcFeaturesInfo(
    TVector<TVector<ui64>> trueDocsPerFeature,
    const ui64 docCount,
    bool symmetric);

TVector<double> CalculateEffectToInfoRate(const TVector<double>& effect, const TVector<double>& info);

struct TFeaturePairInteractionInfo {
    double Score;
    int Feature1, Feature2;

public:
    TFeaturePairInteractionInfo()
        : Score(0)
        , Feature1(-1)
        , Feature2(-1)
    {
    }
    TFeaturePairInteractionInfo(double score, int feature1, int feature2)
        : Score(score)
        , Feature1(feature1)
        , Feature2(feature2)
    {
    }

    bool operator<(const TFeaturePairInteractionInfo& other) const {
        return Score < other.Score;
    }
};

const int EXISTING_PAIRS_COUNT = -1;

TVector<TFeaturePairInteractionInfo> CalcMostInteractingFeatures(const TVector<TMxTree>& trees,
                                                                 int topPairsCount = EXISTING_PAIRS_COUNT);


TVector<TFeaturePairInteractionInfo> CalcMostInteractingFeatures(const TFullModel& model,
                                                                 const THashMap<TFeature, int, TFeatureHash>& featureToIdx,
                                                                 int topPairsCount = EXISTING_PAIRS_COUNT);


/*
 * Don't use this function for converted oblivious model,
 * results would be mismatch with results of CalcEffect if tree has oblivious sub trees
 */
template <class T>
TVector<double> CalcEffectForNonObliviousModel(
    const TFullModel& model,
    const THashMap<TFeature, int, TFeatureHash>& featureToIdx,
    TConstArrayRef<T> weightedDocCountInLeaf) {

    CB_ENSURE_INTERNAL(!model.IsOblivious(), "CalcEffectForNonObliviousModel function got oblivious model");

    const auto& binFeatures = model.ModelTrees->GetBinFeatures();
    const auto leafValues = model.ModelTrees->GetModelTreeData()->GetLeafValues();
    const int approxDimension = model.ModelTrees->GetDimensionsCount();
    const int featureCount = featureToIdx.size();
    TVector<double> res(featureCount, 0);

    for (size_t treeIdx = 0; treeIdx < model.GetTreeCount(); ++treeIdx) {
        TVector<TTriangleNodes> nodesStack;

        const int treeIdxsStart = model.ModelTrees->GetModelTreeData()->GetTreeStartOffsets()[treeIdx];
        const int treeIdxsEnd = treeIdxsStart + model.ModelTrees->GetModelTreeData()->GetTreeSizes()[treeIdx];

        THashMap<int, TNodeInfo> nodeIdxToInfo;

        for (int nodeIdx = treeIdxsStart; nodeIdx < treeIdxsEnd; ++nodeIdx) {
            const auto& node = model.ModelTrees->GetModelTreeData()->GetNonSymmetricStepNodes()[nodeIdx];

            if (node.LeftSubtreeDiff == 0 || node.RightSubtreeDiff == 0) { // node is terminal
                const int leafValueIndex = model.ModelTrees->GetModelTreeData()->GetNonSymmetricNodeIdToLeafId()[nodeIdx];
                TVector<double> values(
                    leafValues.begin() + leafValueIndex,
                    leafValues.begin() + leafValueIndex + approxDimension);
                nodeIdxToInfo[nodeIdx] = TNodeInfo{
                    values,
                    weightedDocCountInLeaf[leafValueIndex / approxDimension]
                };
                if (node.LeftSubtreeDiff == 0 && node.RightSubtreeDiff == 0) {
                    continue;
                }
            }
            const int split = model.ModelTrees->GetModelTreeData()->GetTreeSplits()[nodeIdx];
            const auto& feature = GetFeature(model, binFeatures[split]);
            const int featureIdx = featureToIdx.at(feature);
            nodesStack.push_back(
                TTriangleNodes {
                    nodeIdx,
                    nodeIdx + node.LeftSubtreeDiff,
                    nodeIdx + node.RightSubtreeDiff,
                    featureIdx
                }
            );
        }
        while (!nodesStack.empty()) {
            const TTriangleNodes nodes = nodesStack.back();
            nodesStack.pop_back();

            const int parentNodeIdx = nodes.ParentIdx;
            const int leftNodeIdx = nodes.LeftChildIdx;
            const int rightNodeIdx = nodes.RightChildIdx;
            const int featureIdx = nodes.SplitfeatureIdx;

            TNodeInfo leftInfo = nodeIdxToInfo[leftNodeIdx];
            nodeIdxToInfo.erase(leftNodeIdx);
            TNodeInfo rightInfo = nodeIdxToInfo[rightNodeIdx];
            nodeIdxToInfo.erase(rightNodeIdx);

            const double count1 = leftInfo.Count;
            const double count2 = rightInfo.Count;
            const double sumCount = count1 + count2;

            TVector<double> parentAvrg;
            for (int dimension = 0; dimension < approxDimension; dimension++) {
                const double val1 = count1 ? leftInfo.Values[dimension] : 0;
                const double val2 = count2 ? rightInfo.Values[dimension] : 0;

                const double avrg = (val1 * count1 + val2 * count2) / (sumCount ? sumCount : 1);
                const double dif = Sqr(val1 - avrg) * count1 + Sqr(val2 - avrg) * count2;

                res[featureIdx] += dif;
                parentAvrg.push_back(avrg);
            }
            nodeIdxToInfo[parentNodeIdx] = TNodeInfo{parentAvrg, sumCount};
        }
    }
    ConvertToPercents(res);
    return res;
}

template <typename T>
TVector<double> CalcEffect(
    const TVector<TMxTree>& trees,
    const TVector<TConstArrayRef<T>>& weightedDocCountInLeaf) {

    int featureCount = GetMaxSrcFeature(trees) + 1;
    TVector<double> res(featureCount, 0);

    for (int treeIdx = 0; treeIdx < trees.ysize(); treeIdx++) {
        const auto& tree = trees[treeIdx];
        for (int feature = 0; feature < tree.SrcFeatures.ysize(); feature++) {
            int srcIdx = tree.SrcFeatures[feature];
            for (int leafIdx = 0; leafIdx < tree.Leaves.ysize(); ++leafIdx) {
                int inverted = leafIdx ^ (1 << feature);
                if (inverted < leafIdx) {
                    continue;
                }

                double count1 = weightedDocCountInLeaf[treeIdx][leafIdx];
                double count2 = weightedDocCountInLeaf[treeIdx][inverted];
                if (count1 == 0 || count2 == 0) {
                    continue;
                }

                for (int valInLeafIdx = 0; valInLeafIdx < tree.Leaves[leafIdx].Vals.ysize(); ++valInLeafIdx) {
                    double val1 = tree.Leaves[leafIdx].Vals[valInLeafIdx];
                    double val2 = tree.Leaves[inverted].Vals[valInLeafIdx];

                    double avrg = (val1 * count1 + val2 * count2) / (count1 + count2);
                    double dif = Sqr(val1 - avrg) * count1 + Sqr(val2 - avrg) * count2;

                    res[srcIdx] += dif;
                }
            }
        }
    }
    ConvertToPercents(res);
    return res;
}

template <typename T>
TVector<double> CalcEffect(
    const TVector<TMxTree>& trees,
    const TVector<TVector<T>>& weightedDocCountInLeaf) {

    TVector<TConstArrayRef<T>> weightInLeafArrRef;
    for (const auto& treeWeights: weightedDocCountInLeaf) {
        weightInLeafArrRef.push_back(TConstArrayRef<T>(
            treeWeights.begin(),
            treeWeights.size()));
    }
    return CalcEffect(trees, weightInLeafArrRef);
}
