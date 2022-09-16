#include "statistics_data_structures.h"

#include <catboost/libs/helpers/json_helpers.h>

using namespace NCB;


TFloatFeatureStatistics::TFloatFeatureStatistics()
    : MinValue(std::numeric_limits<double>::max())
    , MaxValue(std::numeric_limits<double>::min())
    , Sum(0.)
    , ObjectCount(0)
{
}

void TFloatFeatureStatistics::Update(float feature) {
    with_lock(Mutex) {
        MinValue = Min<float>(MinValue, feature);
        MaxValue = Max<float>(MaxValue, feature);
        Sum += feature;
        ObjectCount += 1;
    }
}

NJson::TJsonValue TFloatFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    result.InsertValue("MinValue", MinValue);
    result.InsertValue("MaxValue", MaxValue);
    result.InsertValue("Sum", Sum);
    result.InsertValue("ObjectCount", ObjectCount);
    if (ObjectCount > 0) {
        result.InsertValue("Mean", Sum / double(ObjectCount));
    } else {
        result.InsertValue("Mean", std::nan(""));
    }

    return result;
}

void TFloatFeatureStatistics::Update(const TFloatFeatureStatistics& update) {
    with_lock(Mutex) {
        MinValue = Min<float>(MinValue, update.MinValue);
        MaxValue = Max<float>(MaxValue, update.MaxValue);
        Sum += update.Sum;
        ObjectCount += update.ObjectCount;
    }
}

void TTargetsStatistics::Init(const TDataMetaInfo& metaInfo) {
    TargetType = metaInfo.TargetType;
    TargetCount = metaInfo.TargetCount;
    switch (TargetType) {
        case ERawTargetType::Float:
            FloatTargetStatistics.resize(TargetCount);
            break;
        case ERawTargetType::Integer:
        case ERawTargetType::String:
            StringTargetStatistics.resize(TargetCount);
            for (ui32 i = 0; i < TargetCount; ++i) {
                StringTargetStatistics[i].TargetType = TargetType;
            }
            break;
        default:
            Y_ASSERT(false);
    }
}

void TTargetsStatistics::Update(ui32 flatTargetIdx, TStringBuf value) {
    StringTargetStatistics[flatTargetIdx].Update(value);
}

void TTargetsStatistics::Update(ui32 flatTargetIdx, float value) {
    FloatTargetStatistics[flatTargetIdx].Update(value);
}

NJson::TJsonValue TTargetsStatistics::ToJson() const {
    NJson::TJsonValue result;
    InsertEnumType("TargetType", TargetType, &result);
    result.InsertValue("TargetCount", TargetCount);

    switch (TargetType) {
        case ERawTargetType::Float:
            result.InsertValue("TargetStatistics", AggregateStatistics(FloatTargetStatistics));
            break;
        case ERawTargetType::Integer:
        case ERawTargetType::String:
            result.InsertValue("TargetStatistics", AggregateStatistics(StringTargetStatistics));
            break;
        default:
            Y_ASSERT(false);
    }
    return result;
}

void TStringTargetStatistic::Update(TStringBuf feature) {
    CB_ENSURE(TargetType == ERawTargetType::String, "bad target type");
    with_lock(Mutex) {
        StringTargets[feature]++;
    }
}

void TStringTargetStatistic::Update(ui32 feature) {
    with_lock(Mutex) {
        IntegerTargets[feature]++;
    }
}

void TStringTargetStatistic::Update(const TStringTargetStatistic& update) {
    with_lock(Mutex) {
        CB_ENSURE(TargetType == update.TargetType, "bad target type");
        for (auto const& x : update.IntegerTargets) {
            IntegerTargets[x.first] += x.second;
        }
        for (auto const& x : update.StringTargets) {
            StringTargets[x.first] += x.second;
        }
    }
}

NJson::TJsonValue TStringTargetStatistic::ToJson() const {
    NJson::TJsonValue result;
    NJson::TJsonValue targetsDistribution;
    InsertEnumType("TargetType", TargetType, &result);
    switch (TargetType) {
        case ERawTargetType::String:
            for (auto const& x : StringTargets) {
                NJson::TJsonValue stats;
                stats.InsertValue("Value", x.first);
                stats.InsertValue("Count", x.second);
                targetsDistribution.AppendValue(stats);
            }
            break;
        case ERawTargetType::Integer:
            for (auto const& x : IntegerTargets) {
                NJson::TJsonValue stats;
                stats.InsertValue("Value", x.first);
                stats.InsertValue("Count", x.second);
                targetsDistribution.AppendValue(stats);
            }
            break;
        default:
            Y_ASSERT(false);
    }
    result.InsertValue("TargetsDistributionInSample", targetsDistribution);
    return result;
}
