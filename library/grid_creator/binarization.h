#pragma once

#include <util/generic/fwd.h>

enum class EBorderSelectionType {
    Median = 1,
    GreedyLogSum = 2,
    UniformAndQuantiles = 3,
    MinEntropy = 4,
    MaxLogSum = 5,
    Uniform = 6,
    GreedyMinEntropy = 7,
};

THashSet<float> BestSplit(
    TVector<float>& features,
    int maxBordersCount,
    EBorderSelectionType type,
    bool filterNans = false,
    bool featuresAreSorted = false);

THashSet<float> BestWeightedSplit(
    const TVector<float>& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EBorderSelectionType type,
    bool filterNans = false,
    bool featuresAreSorted = false);

size_t CalcMemoryForFindBestSplit(
    int maxBordersCount,
    size_t docsCount,
    EBorderSelectionType type);

namespace NSplitSelection {
    class IBinarizer {
    public:
        // featureValues vector might be changed!
        virtual THashSet<float> BestSplit(
            TVector<float>& features,
            int maxBordersCount,
            bool featuresAreSorted = false) const = 0;

        virtual ~IBinarizer() = default;
    };

    THolder<IBinarizer> MakeBinarizer(EBorderSelectionType borderSelectionType);
}

// The rest is for unit tests only
enum class EPenaltyType {
    MinEntropy,
    MaxSumLog,
    W2
};

enum class EOptimizationType {
    Exact,
    Greedy,
};


template <EPenaltyType type>
double Penalty(double weight);

template <EPenaltyType penaltyType>
THashSet<float> BestWeightedSplit(
    const TVector<float>& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EOptimizationType optimizationType,
    bool filterNans,
    bool featuresAreSorted);

std::pair<TVector<float>, TVector<float>> GroupAndSortWeighedValues(
    const TVector<float>& featureValues,
    const TVector<float>& weights,
    bool filterNans,
    bool isSorted);

std::pair<TVector<float>, TVector<float>> GroupAndSortValues(
    const TVector<float>& featureValues,
    bool filterNans,
    bool isSorted);
