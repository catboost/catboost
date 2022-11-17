#include "statistics_data_structures.h"

#include <catboost/libs/cat_feature/cat_feature.h>
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
    if (std::isinf(feature) || std::isnan(feature)) {
        return;
    }
    with_lock(Mutex) {
        MinValue = Min<float>(MinValue, feature);
        MaxValue = Max<float>(MaxValue, feature);
        Sum += static_cast<long double>(feature);
        ObjectCount += 1;
    }
}

bool TFloatFeatureStatistics::operator==(const TFloatFeatureStatistics& rhs) const {
    return (
        std::tie(MinValue, Sum, ObjectCount) ==
        std::tie(rhs.MinValue, rhs.Sum, rhs.ObjectCount)
    );
}

NJson::TJsonValue TFloatFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    CB_ENSURE_INTERNAL(!std::isnan(MaxValue), "nan value in MaxValue");
    CB_ENSURE_INTERNAL(!std::isnan(MinValue), "nan value in MinValue");
    if (std::isinf(MinValue)) {
        result.InsertValue("MinValue", ToString(MinValue));
    } else {
        result.InsertValue("MinValue", MinValue);
    }
    if (std::isinf(MaxValue)) {
        result.InsertValue("MaxValue", ToString(MaxValue));
    } else {
        result.InsertValue("MaxValue", MaxValue);
    }
    result.InsertValue("Sum", ToString(Sum));
    result.InsertValue("ObjectCount", ObjectCount);
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

bool TTargetsStatistics::operator==(const TTargetsStatistics& a) const {
    return (
        std::tie(TargetType, TargetCount, FloatTargetStatistics, StringTargetStatistics) ==
        std::tie(a.TargetType, a.TargetCount, a.FloatTargetStatistics, a.StringTargetStatistics)
    );
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
    targetsDistribution.SetType(EJsonValueType::JSON_ARRAY);
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


void TSampleIdStatistics::Update(const TString& value) {
    with_lock(Mutex) {
        ObjectCount++;
        SumLen += value.size();
    }
}

NJson::TJsonValue TSampleIdStatistics::ToJson() const {
    NJson::TJsonValue result;
    result.InsertValue("SumLen", ToString(SumLen));
    result.InsertValue("ObjectCount", ObjectCount);
    if (ObjectCount) {
        double average = static_cast<long double>(SumLen) / static_cast<long double>(ObjectCount);
        result.InsertValue("AverageSampleId", average);
    }
    return result;
}

void TSampleIdStatistics::Update(const TSampleIdStatistics& update) {
    with_lock(Mutex) {
        ObjectCount += update.ObjectCount;
        SumLen += update.SumLen;
    }
}

void TTextFeatureStatistics::Update(TStringBuf value) {
    with_lock(Mutex) {
        if (!Example.Defined()) {
            Example = value;
            return;
        }
        if (*Example != value) {
            IsConst = false;
        }
    }
}

NJson::TJsonValue TTextFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    result.InsertValue("IsConst", ToString(IsConst));
    return result;
}

void TTextFeatureStatistics::Update(const TTextFeatureStatistics& update) {
    with_lock(Mutex) {
        if (!Example.Defined()) {
            Example = update.Example;
            IsConst = update.IsConst;
            return;
        }
        if (IsConst != update.IsConst || *Example != *(update.Example)) {
            IsConst = false;
        }
    }
}

void TCatFeatureStatistics::Update(ui32 value) {
    with_lock(Mutex) {
        ImperfectHashSet.insert(value);
    }
}

void TCatFeatureStatistics::Update(TStringBuf value) {
    Update(CalcCatFeatureHash(value));
}

NJson::TJsonValue TCatFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    result.InsertValue("CatFeatureCount", ImperfectHashSet.size());
    return result;
}

void TCatFeatureStatistics::Update(const TCatFeatureStatistics& update) {
    ImperfectHashSet.insert(update.ImperfectHashSet.begin(), update.ImperfectHashSet.end());
}

void TFeatureStatistics::Init(const TDataMetaInfo& metaInfo) {
    FloatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetFloatFeatureCount());
    CatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetCatFeatureCount());
    TextFeatureStatistics.resize(metaInfo.FeaturesLayout->GetTextFeatureCount());
}

NJson::TJsonValue TFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    result.InsertValue("FloatFeatureStatistics", AggregateStatistics(FloatFeatureStatistics));
    result.InsertValue("CatFeaturesStatistics", AggregateStatistics(CatFeatureStatistics));
    result.InsertValue("TextFeaturesStatistics", AggregateStatistics(TextFeatureStatistics));
    //  ToDo: add statistics for Embedding features
    return result;
}

void TFeatureStatistics::Update(const TFeatureStatistics& update) {
    CB_ENSURE(FloatFeatureStatistics.size() == update.FloatFeatureStatistics.size());
    CB_ENSURE(CatFeatureStatistics.size() == update.CatFeatureStatistics.size());
    CB_ENSURE(TextFeatureStatistics.size() == update.TextFeatureStatistics.size());
    for (ui32 i = 0; i < FloatFeatureStatistics.size(); ++i) {
        FloatFeatureStatistics[i].Update(update.FloatFeatureStatistics[i]);
    }
    for (ui32 i = 0; i < CatFeatureStatistics.size(); ++i) {
        CatFeatureStatistics[i].Update(update.CatFeatureStatistics[i]);
    }
    for (ui32 i = 0; i < TextFeatureStatistics.size(); ++i) {
        TextFeatureStatistics[i].Update(update.TextFeatureStatistics[i]);
    }
}

bool TFeatureStatistics::operator==(const TFeatureStatistics& a) const {
    return (
        std::tie(FloatFeatureStatistics, CatFeatureStatistics, TextFeatureStatistics) ==
        std::tie(a.FloatFeatureStatistics, a.CatFeatureStatistics, a.TextFeatureStatistics)
    );
}

NJson::TJsonValue TGroupwiseStats::ToJson() const {
    NJson::TJsonValue stats;
    stats["GroupsCount"] = GroupsCount;
    stats["GroupsTotalSize"] = GroupsTotalSize;
    if (GroupsCount) {
        stats["GroupsAverageSize"] = GetAverageGroupSize();
    }
    return stats;
}

void TGroupwiseStats::InfoLog() const {
    CATBOOST_INFO_LOG << "GroupsCount: " << GroupsCount << "\nGroupsTotalSize: " << GroupsTotalSize << Endl;
    if (GroupsCount) {
        CATBOOST_INFO_LOG << "GroupsAverageSize: " << GetAverageGroupSize() << Endl;
    }
}

NJson::TJsonValue TDatasetStatistics::ToJson() const {
    NJson::TJsonValue result;

    result.InsertValue("TargetsStatistics", TargetsStatistics.ToJson());
    result.InsertValue("FeatureStatistics", FeatureStatistics.ToJson());
    result.InsertValue("SampleIdStatistics", SampleIdStatistics.ToJson());
    if (GroupwiseStats.Defined()) {
        result.InsertValue("GroupStats", GroupwiseStats->ToJson());
    }

    result.InsertValue("ObjectCount", TargetsStatistics.GetObjectCount());

    return result;
}

void TDatasetStatistics::Update(const TDatasetStatistics& update) {
    FeatureStatistics.Update(update.FeatureStatistics);
    TargetsStatistics.Update(update.TargetsStatistics);
    SampleIdStatistics.Update(update.SampleIdStatistics);
}
