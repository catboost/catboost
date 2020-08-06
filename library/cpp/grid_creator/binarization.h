#pragma once

#include <util/generic/fwd.h>
#include <util/generic/hash_set.h>
#include <util/generic/maybe.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/generic/ymath.h>


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
    bool featuresAreSorted = false,
    const TMaybe<TVector<float>>& initialBorders = Nothing());

THashSet<float> BestWeightedSplit(
    TVector<float>&& featureValues,
    const TVector<float>& weights,
    int maxBordersCount,
    EBorderSelectionType type,
    bool filterNans = false,
    bool featuresAreSorted = false);


// used for sparse data

template <class T>
struct TDefaultValue {
    T Value;
    ui64 Count;

public:
    TDefaultValue(T value, ui64 count)
        : Value(value)
        , Count(count)
    {
        Y_ENSURE(Count >= 1, "It is required that default value count is non-0");
    }
};


namespace NSplitSelection {

    struct TFeatureValues {
        // Values could contain values that are equal to DefaultValue, but don't add it explicitly
        TVector<float> Values;
        bool ValuesSorted;
        TMaybe<TDefaultValue<float>> DefaultValue;

    public:
        explicit TFeatureValues(
            TVector<float>&& values,
            bool valuesSorted = false,
            TMaybe<TDefaultValue<float>> defaultValue = Nothing())
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
        const TMaybe<TDefaultValue<float>>& defaultValue,
        EBorderSelectionType type);


    struct TDefaultQuantizedBin {
        ui32 Idx; // if for splits: bin borders are [Border[Idx - 1], Borders[Idx])
        float Fraction;

    public:
        bool operator==(const TDefaultQuantizedBin & rhs) const {
            constexpr float EPS = 1e-6f;
            return (Idx == rhs.Idx) && (Abs(Fraction - rhs.Fraction) < EPS);
        }
    };

    struct TQuantization {
        TVector<float> Borders;
        TMaybe<TDefaultQuantizedBin> DefaultQuantizedBin;

    public:
        TQuantization() = default;

        explicit TQuantization(
            TVector<float>&& borders,
            TMaybe<TDefaultQuantizedBin> defaultQuantizedBin = Nothing())
            : Borders(std::move(borders))
            , DefaultQuantizedBin(defaultQuantizedBin)
        {}

        bool operator==(const TQuantization& rhs) const {
            return (Borders == rhs.Borders) && (DefaultQuantizedBin == rhs.DefaultQuantizedBin);
        }
    };

    TQuantization BestSplit(
        TFeatureValues&& features,
        bool featureValuesMayContainNans,
        int maxBordersCount,
        EBorderSelectionType type,
        // if defined - calculate DefaultQuantizedBin
        TMaybe<float> quantizedDefaultBinFraction = Nothing(),
        const TMaybe<TVector<float>>& initialBorders = Nothing());


    class IBinarizer {
    public:
        virtual ~IBinarizer() = default;

        virtual TQuantization BestSplit(
            TFeatureValues&& features,
            int maxBordersCount,
            TMaybe<float> quantizedDefaultBinFraction = Nothing(),
            const TMaybe<TVector<float>>& initialBorders = Nothing()) const = 0;
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
