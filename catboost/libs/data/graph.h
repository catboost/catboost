#pragma once

#include <catboost/private/libs/options/enums.h>

#include <util/generic/vector.h>
#include <util/generic/ymath.h>

#include <util/system/types.h>

#include <limits>
#include <string>

struct TPair;

namespace NCB {
    enum class EFloatGraphFeatureType {
        Mean,
        Min,
        Max
    };

    constexpr ui32 kFloatAggregationFeaturesCount = 3;

    struct TPairInGroup;
    using TGroupedPairsInfo = TVector<TPairInGroup>;
    using TFlatPairsInfo = TVector<TPair>;
    using TRawPairsData = std::variant<TFlatPairsInfo, TGroupedPairsInfo>;

    TVector<TVector<ui32>> ConvertGraphToAdjMatrix(const TRawPairsData& graph, ui32 objectCount);

    struct TFloatAggregation {
        float Mean = 0;
        float Min = std::numeric_limits<float>::max();
        float Max = std::numeric_limits<float>::lowest();
    };

    template <typename TFloatFeatureAccessor>
    TFloatAggregation CalcAggregationFeatures(
        const TVector<TVector<ui32>>& matrix,
        TFloatFeatureAccessor floatAccessor,
        ui32 objectIdx
    ) {
        TFloatAggregation agg;
        if (!matrix[objectIdx].empty()) {
            for (auto neighbour: matrix[objectIdx]) {
                const auto& value = floatAccessor(neighbour);
                agg.Mean += value;
                agg.Max = Max(agg.Max, value);
                agg.Min = Min(agg.Min, value);
            }
            agg.Mean /= matrix[objectIdx].size();
        }
        return agg;
    }

    TVector<NCB::EFloatGraphFeatureType> GetAggregationTypeNames(EFeatureType featureType);
}
