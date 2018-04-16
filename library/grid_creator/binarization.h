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

THashSet<float> BestSplit(
    TVector<float>& featureVals,
    int bordersCount,
    EBorderSelectionType type,
    bool nanValuesIsInfty = false);

size_t CalcMemoryForFindBestSplit(int bordersCount,
                                  size_t docsCount,
                                  EBorderSelectionType type);

namespace NSplitSelection {
    class IBinarizer {
    public:
        // featureValues vector might be changed!
        virtual THashSet<float> BestSplit(TVector<float>& featureValues,
                                          int bordersCount,
                                          bool isSorted = false) const = 0;

        virtual ~IBinarizer() {
        }
    };

    class TMedianInBinBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                  int bordersCount,
                                  bool isSorted) const override;
    };

    class TMedianPlusUniformBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                  int bordersCount,
                                  bool isSorted) const override;
    };

    class TMinEntropyBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                  int bordersCount,
                                  bool isSorted) const override;
    };

    class TMaxSumLogBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                  int bordersCount,
                                  bool isSorted) const override;
    };

    // Works in O(binCount * log(n)) + O(nlogn) for sorting.
    class TMedianBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                  int bordersCount,
                                  bool isSorted) const override;
    };

    class TUniformBinarizer: public IBinarizer {
    public:
        THashSet<float> BestSplit(TVector<float>& featureValues,
                                  int bordersCount,
                                  bool isSorted) const override;
    };

}
