#pragma once

#include "pair.h"

#include <util/random/fast.h>
#include <util/generic/vector.h>
#include <util/generic/utility.h>
#include <util/ysaveload.h>
#include <util/generic/hash.h>

struct TDocInfo {
    float Target = 0;
    float Weight = 1;
    yvector<float> Factors;
    yvector<double> Baseline;
    TString Id;

    void Swap(TDocInfo& other) {
        DoSwap(Target, other.Target);
        DoSwap(Weight, other.Weight);
        Factors.swap(other.Factors);
        Baseline.swap(other.Baseline);
        DoSwap(Id, other.Id);
    }
};

struct TPool {
    yvector<TDocInfo> Docs;
    yvector<int> CatFeatures;
    yvector<TString> FeatureId;
    yhash<int, TString> CatFeaturesHashToString;
    yvector<TPair> Pairs;
};
