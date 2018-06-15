#pragma once

#include <util/generic/vector.h>
#include <util/generic/ymath.h>

struct TMxTree {
    struct TValsInLeaf {
        TVector<double> Vals;
    };

    TVector<int> SrcFeatures;
    TVector<TValsInLeaf> Leaves;
};


int GetMaxSrcFeature(const TVector<TMxTree>& trees);

void ConvertToPercents(TVector<double>& res);

template<class T>
TVector<double> CalcEffect(const TVector<TMxTree>& trees,
                           const TVector<TVector<T>>& weightedDocCountInLeaf)
{
    TVector<double> res;
    int featureCount = GetMaxSrcFeature(trees) + 1;
    res.resize(featureCount);

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


TVector<double> CalcFeaturesInfo(TVector<TVector<ui64>> trueDocsPerFeature,
                                 const ui64 docCount,
                                 bool symmetric);

TVector<double> CalculateEffectToInfoRate(const TVector<double>& effect,
                                          const TVector<double>& info);

struct TFeaturePairInteractionInfo {
    double Score;
    int Feature1, Feature2;

    TFeaturePairInteractionInfo()
        : Score(0)
        , Feature1(-1)
        , Feature2(-1)
    {
    }
    TFeaturePairInteractionInfo(double score, int f1, int f2)
        : Score(score)
        , Feature1(f1)
        , Feature2(f2)
    {
    }

    bool operator<(const TFeaturePairInteractionInfo& other) const {
        return Score < other.Score;
    }
};

const int EXISTING_PAIRS_COUNT = -1;

TVector<TFeaturePairInteractionInfo> CalcMostInteractingFeatures(const TVector<TMxTree>& trees,
                                                                 int topPairsCount = EXISTING_PAIRS_COUNT);
