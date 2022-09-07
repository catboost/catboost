#include "statistics_data_structures.h"

TFloatFeatureStatistics::TFloatFeatureStatistics()
    : MinValue(std::numeric_limits<double>::max())
    , MaxValue(std::numeric_limits<double>::min())
    , Sum(0.)
    , ObjectCount(0)
{
}

void TFloatFeatureStatistics::Update(float feature) {
    TGuard g(Mutex);

    MinValue = Min<float>(MinValue, feature);
    MaxValue = Max<float>(MaxValue, feature);
    Sum += feature;
    ObjectCount += 1;
}

NJson::TJsonValue TFloatFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    result.InsertValue("MinValue", MinValue);
    result.InsertValue("MaxValue", MaxValue);
    result.InsertValue("Sum", Sum);
    if (ObjectCount > 0) {
        result.InsertValue("Mean", Sum / double(ObjectCount));
    } else {
        result.InsertValue("Mean", std::nan(""));
    }

    return result;
}

void TTargetsStatistics::Init(const TDataMetaInfo& MetaInfo) {
    TargetType = MetaInfo.TargetType;
    TargetCount = MetaInfo.TargetCount;
    switch (TargetType) {
        case ERawTargetType::Float:
            FloatTargetStatistics.resize(TargetCount);
            break;
        case ERawTargetType::Integer:
        case ERawTargetType::String:
            for (ui32 i = 0; i < TargetCount; ++i) {
                StringTargetStatistics.emplace_back(TStringTargetStatistic(TargetType));
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
    TVector<NJson::TJsonValue> targetStatistics;
    // only one is not empty
    for (const auto& statistics : FloatTargetStatistics) {
        targetStatistics.emplace_back(statistics.ToJson());
    }
    for (const auto& statistics : StringTargetStatistics) {
        targetStatistics.emplace_back(statistics.ToJson());
    }
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

TStringTargetStatistic::TStringTargetStatistic(ERawTargetType targetType)
    : TargetType(targetType)
{
}

void TStringTargetStatistic::Update(TStringBuf feature) {
    CB_ENSURE(TargetType == ERawTargetType::String, "bad target type");
    TGuard g(Mutex);
    StringTargets[feature]++;
}

void TStringTargetStatistic::Update(ui32 feature) {
    CB_ENSURE(TargetType == ERawTargetType::Integer, "bad target type");
    TGuard g(Mutex);
    IntegerTargets[feature]++;
}

NJson::TJsonValue TStringTargetStatistic::ToJson() const {
    NJson::TJsonValue result;
    NJson::TJsonValue targetsDistribution;
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
                stats.InsertValue("Value", x.first).InsertValue("Count", x.second);
                targetsDistribution.AppendValue(stats);
            }
            break;
        default:
            Y_ASSERT(false);
    }
    result.InsertValue("TargetsDistributionInSample", targetsDistribution);
    return result;
}
