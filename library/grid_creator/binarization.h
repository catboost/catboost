#pragma once

#include <util/generic/fwd.h>

enum class EBorderSelectionType {
    Median = 1,
    GreedyLogSum = 2,
    UniformAndQuantiles = 3,
    MinEntropy = 4,
    MaxLogSum = 5,
    Uniform = 6
};

THashSet<float> BestSplit(
    TVector<float>& features,
    int maxBordersCount,
    EBorderSelectionType type,
    bool nanValueIsInfty = false,
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
