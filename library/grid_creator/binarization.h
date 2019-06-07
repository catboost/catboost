#pragma once

#include <util/generic/fwd.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>


// TODO(akhropov): move all these to NSplitSelection namespace as well

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
    TVector<float>&& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EBorderSelectionType type,
    bool filterNans = false,
    bool featuresAreSorted = false);


namespace NSplitSelection {

    // used for sparse data
    struct TDefaultValue {
        float Value;
        ui64 Count;

    public:
        TDefaultValue(float value, ui64 count)
            : Value(value)
            , Count(count)
        {
            Y_ENSURE(Count >= 1, "It is required that default value count is non-0");
        }
    };

    struct TFeatureValues {
        // Values could contain values that are equal to DefaultValue, but don't add it explicitly
        TVector<float> Values;
        bool ValuesSorted;
        TMaybe<TDefaultValue> DefaultValue;

    public:
        explicit TFeatureValues(
            TVector<float>&& values,
            bool valuesSorted = false,
            TMaybe<TDefaultValue> defaultValue = Nothing())
            : Values(std::move(values))
            , ValuesSorted(valuesSorted)
            , DefaultValue(std::move(defaultValue))
        {}
    };

    /* Note: memory for features.Values argument is not included, only memory that could be allocated
     *   inside BestSplit
     */
    size_t CalcMemoryForFindBestSplit(
        int maxBordersCount,
        size_t nonDefaultObjectCount,
        const TMaybe<TDefaultValue>& defaultValue,
        EBorderSelectionType type);

    THashSet<float> BestSplit(
        TFeatureValues&& features,
        bool featureValuesMayContainNans,
        int maxBordersCount,
        EBorderSelectionType type);


    class IBinarizer {
    public:
        virtual ~IBinarizer() = default;

        virtual THashSet<float> BestSplit(TFeatureValues&& features, int maxBordersCount) const = 0;
    };

    THolder<IBinarizer> MakeBinarizer(EBorderSelectionType borderSelectionType);


    // The rest is for unit tests only
    namespace NImpl {

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
            TVector<float>&& featureValues,
            const TVector<float>& weights,
            int maxBordersCount,
            EOptimizationType optimizationType,
            bool filterNans,
            bool featuresAreSorted);

        std::pair<TVector<float>, TVector<float>> GroupAndSortWeighedValues(
            TVector<float>&& featureValues,
            TVector<float>&& weights,
            bool filterNans,
            bool isSorted);

        std::pair<TVector<float>, TVector<float>> GroupAndSortValues(
            TFeatureValues&& features,
            bool filterNans,
            bool cumulativeWeights = false);

    }
}
