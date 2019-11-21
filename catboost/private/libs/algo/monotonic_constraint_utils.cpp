#include "monotonic_constraint_utils.h"


TVector<ui32> BuildLinearOrderOnLeafsOfMonotonicSubtree(
    const TVector<int>& treeMonotonicConstraints,
    ui32 monotonicSubtreeIndex
) {
    ui32 currDepthBitMask = 1u;
    ui32 leastLeafIndex = 0u;
    TVector<ui32> monotonicSplitBitMasks;
    for (int constraint : treeMonotonicConstraints) {
        if (constraint == 0) {
            if (monotonicSubtreeIndex & 1u) {
                leastLeafIndex |= currDepthBitMask;
            }
            monotonicSubtreeIndex = monotonicSubtreeIndex >> 1;
        } else {
            monotonicSplitBitMasks.push_back(currDepthBitMask);
            if (constraint == -1) {
                leastLeafIndex |= currDepthBitMask;
            }
        }
        currDepthBitMask = currDepthBitMask << 1;
    }
    Y_ASSERT(monotonicSubtreeIndex == 0u);
    const ui32 monotonicSplitCount = monotonicSplitBitMasks.size();
    TVector<ui32> leafOrder(1u << monotonicSplitCount, leastLeafIndex);
    for (ui32 leafRank = 0u; leafRank < leafOrder.size(); ++leafRank) {
        for (ui32 monotonicDepth = 0u; monotonicDepth < monotonicSplitCount; ++monotonicDepth) {
            if ((leafRank >> (monotonicSplitCount - 1 - monotonicDepth)) & 1u) {
                leafOrder[leafRank] ^= monotonicSplitBitMasks[monotonicDepth];
            }
        }
    }
    return leafOrder;
}

TVector<TVector<ui32>> BuildMonotonicLinearOrdersOnLeafs(const TVector<int>& treeMonotonicConstraints) {
    int nonMonotonicFeatureCount = 0;
    for (ui32 i = 0; i < treeMonotonicConstraints.size(); ++i) {
        if (treeMonotonicConstraints[i] == 0) {
            nonMonotonicFeatureCount += 1;
        }
    }
    TVector<TVector<ui32>> result;
    const ui32 subTreeCount = (1u << nonMonotonicFeatureCount);
    result.reserve(subTreeCount);
    for (ui32 subTreeIndex=0; subTreeIndex < subTreeCount; ++subTreeIndex) {
        result.push_back(BuildLinearOrderOnLeafsOfMonotonicSubtree(treeMonotonicConstraints, subTreeIndex));
    }
    return result;
}

namespace {
    class TIsotonicLevelSet {
    /* Level set of a function f is a set of the form {x : f(x) = v} for some value v.
     * This class describes a level set of the function that represents the solution of
     * one-dimensional isotonic regression.
     */
    public:
        TIsotonicLevelSet(int begin, double weight, double value)
            : Begin_(begin)
            , End_(begin + 1)
            , Weight(weight)
            , SumWeightedValue(weight * value) {
        }

        void MergeLeft(const TIsotonicLevelSet& leftSet) {
            Y_ASSERT(leftSet.End_ == Begin_);
            Begin_ = leftSet.Begin_;
            Weight = leftSet.Weight + Weight;
            SumWeightedValue = leftSet.SumWeightedValue + SumWeightedValue;
        }

        double Average() const {
            return SumWeightedValue / Weight;
        }

        int Begin() const {
            return Begin_;
        }

        int End() const {
            return End_;
        }
    private:
        int Begin_ = 0;
        int End_ = 0;
        double Weight = 0.0;
        double SumWeightedValue = 0.0;
    };
}

void CalcOneDimensionalIsotonicRegression(
    const TVector<double>& values,
    const TVector<double>& weights,
    const TVector<ui32>& indexOrder,
    TVector<double>* solution
) {
    const int size = indexOrder.size();
    TVector<TIsotonicLevelSet> levelSets;
    for (int pointRank = 0; pointRank < size; ++pointRank) {
        auto pointIndex = indexOrder[pointRank];
        TIsotonicLevelSet newLevelSet(pointRank, weights[pointIndex], values[pointIndex]);
        while (!levelSets.empty() && levelSets.back().Average() >= newLevelSet.Average()) {
            newLevelSet.MergeLeft(levelSets.back());
            levelSets.pop_back();
        }
        levelSets.push_back(newLevelSet);
    }
    for (const auto& levelSet : levelSets) {
        const double levelSetValue = levelSet.Average();
        for (int pointRank = levelSet.Begin(); pointRank < levelSet.End(); ++pointRank) {
            (*solution)[indexOrder[pointRank]] = levelSetValue;
        }
    }
}


TVector<int> GetTreeMonotoneConstraints(const TSplitTree& tree, const TMap<ui32, int>& monotoneConstraints) {
    TVector<int> treeMonotoneConstraints(tree.GetDepth(), 0);
    if (!monotoneConstraints.empty()) {
        for (int splitIndex = 0; splitIndex < tree.GetDepth(); ++splitIndex) {
            if (tree.Splits[splitIndex].Type == ESplitType::FloatFeature) {
                int splitFeatureId = tree.Splits[splitIndex].FeatureIdx;
                Y_ASSERT(splitFeatureId != -1);
                if (monotoneConstraints.contains(splitFeatureId)) {
                    treeMonotoneConstraints[splitIndex] = monotoneConstraints.at(splitFeatureId);
                }
            }
        }
    }
    return treeMonotoneConstraints;
}

bool CheckMonotonicity(const TVector<ui32>& indexOrder, const TVector<double>& values) {
    for (ui32 i = 0; i + 1 < indexOrder.size(); ++i) {
        if (values[indexOrder[i]] > values[indexOrder[i + 1]]) {
            return false;
        }
    }
    return true;
}
