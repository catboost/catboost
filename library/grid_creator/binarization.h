#pragma once

#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/generic/hash_set.h>

enum class EBorderSelectionType {
    Median = 1,
    GreedyLogSum = 2,
    UniformAndQuantiles = 3,
    MinEntropy = 4,
    MaxLogSum = 5,
    Uniform = 6
};

yhash_set<float> BestSplit(
    yvector<float>& featureVals,
    int bordersCount,
    EBorderSelectionType type,
    bool nanValuesIsInfty=false);

namespace NSplitSelection {
class IBinarizer {
public:
    // featureValues vector might be changed!
    virtual yhash_set<float> BestSplit(yvector<float>& featureValues,
                                       int bordersCount,
                                       bool isSorted=false) const = 0;

    virtual ~IBinarizer() {}
};

class TMedianInBinBinarizer : public IBinarizer {
public:
    yhash_set<float> BestSplit(yvector<float>& featureValues,
                               int bordersCount,
                               bool isSorted) const override;
};

class TMedianPlusUniformBinarizer : public IBinarizer {
public:
    yhash_set<float> BestSplit(yvector<float>& featureValues,
                               int bordersCount,
                               bool isSorted) const override;
};

class TMinEntropyBinarizer : public IBinarizer {
public:
    yhash_set<float> BestSplit(yvector<float>& featureValues,
                               int bordersCount,
                               bool isSorted) const override;
};

class TMaxSumLogBinarizer : public IBinarizer {
public:
    yhash_set<float> BestSplit(yvector<float>& featureValues,
                               int bordersCount,
                               bool isSorted) const override;
};

// Works in O(binCount * log(n)) + O(nlogn) for sorting.
class TMedianBinarizer : public IBinarizer {
public:
    yhash_set<float> BestSplit(yvector<float>& featureValues,
                               int bordersCount,
                               bool isSorted) const override;
};

class TUniformBinarizer : public IBinarizer {
public:
    yhash_set<float> BestSplit(yvector<float>& featureValues,
                               int bordersCount,
                               bool isSorted) const override;
};

}  // namespace NSplitSelection
