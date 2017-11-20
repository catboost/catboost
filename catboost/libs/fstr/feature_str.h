#pragma once

#include <util/generic/vector.h>

struct TMxTree {
    struct TValsInLeaf {
        TVector<double> Vals;
    };

    TVector<int> SrcFeatures;
    TVector<TValsInLeaf> Leafs;
};

TVector<double> CalcEffect(const TVector<TMxTree>& trees,
                           const TVector<TVector<ui64>>& docCountInLeaf);

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
