#pragma once

#include <util/generic/fwd.h>
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
    TVector<float>& features,
    int bordersCount,
    EBorderSelectionType type,
    bool nanValueIsInfty = false,
    bool featuresAreSorted = false);

size_t CalcMemoryForFindBestSplit(
    int bordersCount,
    size_t docsCount,
    EBorderSelectionType type);

namespace NSplitSelection {
    class IBinarizer {
    public:
        // featureValues vector might be changed!
        virtual THashSet<float> BestSplit(
            TVector<float>& features,
            int bordersCount,
            bool featuresAreSorted = false) const = 0;

        virtual ~IBinarizer() = default;
    };

    THolder<IBinarizer> MakeBinarizer(EBorderSelectionType borderSelectionType);
}
