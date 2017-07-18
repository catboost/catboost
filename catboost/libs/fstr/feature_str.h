#pragma once

#include <util/generic/vector.h>

struct TMxTree {
    struct TValsInLeaf {
        yvector<double> Vals;
    };

    yvector<int> SrcFeatures;
    yvector<TValsInLeaf> Leafs;
};

yvector<double> CalcEffect(const yvector<TMxTree>& trees,
                           const yvector<yvector<ui64>>& docCountInLeaf);

yvector<double> CalcFeaturesInfo(yvector<yvector<ui64>> trueDocsPerFeature,
                                 const ui64 docCount,
                                 bool symmetric);

yvector<double> CalculateEffectToInfoRate(const yvector<double>& effect,
                                          const yvector<double>& info);

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

yvector<TFeaturePairInteractionInfo> CalcMostInteractingFeatures(const yvector<TMxTree>& trees,
                                                                 int topPairsCount = EXISTING_PAIRS_COUNT);
