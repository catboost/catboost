#include "statistics_data_structures.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/json_helpers.h>

#include <algorithm>

using namespace NCB;


TFloatFeatureStatistics::TFloatFeatureStatistics()
    : MinValue(std::numeric_limits<double>::max())
    , MaxValue(std::numeric_limits<double>::lowest())
    , CustomMin(Nothing())
    , CustomMax(Nothing())
    , OutOfDomainValuesCount(0)
    , Underflow(0)
    , Overflow(0)
    , Sum(0.)
    , SumSqr(0.)
    , ObjectCount(0)
{
}

double TFloatFeatureStatistics::GetMinBorder() const {
    if (CustomMin.Defined() && !std::isinf(CustomMin.GetRef())) {
        return CustomMin.GetRef();
    } else if (ObjectCount) {
        return MinValue;
    } else {
        return std::numeric_limits<double>::lowest();
    }
}

double TFloatFeatureStatistics::GetMaxBorder() const {
    if (CustomMax.Defined() && !std::isinf(CustomMax.GetRef())) {
        return CustomMax.GetRef();
    } else if (ObjectCount) {
        return MaxValue;
    } else {
        return std::numeric_limits<double>::max();
    }
}

void TFloatFeatureStatistics::Update(float feature) {
    with_lock(Mutex) {
        if (std::isinf(feature) || std::isnan(feature)) {
            OutOfDomainValuesCount++;
            return;
        }
        if (CustomMin.Defined() && feature < CustomMin.GetRef()) {
            OutOfDomainValuesCount++;
            Underflow++;
            return;
        }
        if (CustomMin.Defined() && feature > CustomMax.GetRef()) {
            OutOfDomainValuesCount++;
            Overflow++;
            return;
        }
        MinValue = Min<float>(MinValue, feature);
        MaxValue = Max<float>(MaxValue, feature);
        Sum += static_cast<long double>(feature);
        long double featureCasted = feature;
        SumSqr += featureCasted * featureCasted;
        ObjectCount += 1;
    }
}

bool TFloatFeatureStatistics::operator==(const TFloatFeatureStatistics& rhs) const {
    return (
        std::tie(
            MinValue,
            MaxValue,
            CustomMin,
            CustomMax,
            OutOfDomainValuesCount,
            Underflow,
            Overflow,
            Sum,
            SumSqr,
            ObjectCount
        ) ==
        std::tie(
            rhs.MinValue,
            rhs.MaxValue,
            rhs.CustomMin,
            rhs.CustomMax,
            rhs.OutOfDomainValuesCount,
            rhs.Underflow,
            rhs.Overflow,
            rhs.Sum,
            rhs.SumSqr,
            rhs.ObjectCount
        )
    );
}

void InsertFloatValue(const TString& name, double value, NJson::TJsonValue* result) {
    if (!std::isinf(value)) {
        result->InsertValue(name, value);
    } else {
        result->InsertValue(name, ToString(value));
    }
}

NJson::TJsonValue TFloatFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    CB_ENSURE_INTERNAL(!std::isnan(MaxValue), "nan value in MaxValue");
    CB_ENSURE_INTERNAL(!std::isnan(MinValue), "nan value in MinValue");

    if (ObjectCount > 0) {
        CB_ENSURE(!std::isinf(MinValue));
        CB_ENSURE(!std::isinf(MaxValue));
        result.InsertValue("MinValue", MinValue);
        result.InsertValue("MaxValue",  MaxValue);
        result.InsertValue("Sum", ToString(Sum));
        result.InsertValue("SumSqr", ToString(SumSqr));
    }
    result.InsertValue("ObjectCount", ObjectCount);
    if (CustomMin.Defined()) {
        InsertFloatValue("CustomMin", CustomMin.GetRef(), &result);
    }
    if (CustomMax.Defined()) {
        InsertFloatValue("CustomMax", CustomMax.GetRef(), &result);
    }
    if (OutOfDomainValuesCount > 0) {
        result.InsertValue("OutOfDomainValuesCount", OutOfDomainValuesCount);
    }
    if (Underflow > 0) {
        result.InsertValue("Underflow", Underflow);
    }
    if (Overflow > 0) {
        result.InsertValue("Overflow", Overflow);
    }
    return result;
}

void TFloatFeatureStatistics::Update(const TFloatFeatureStatistics& update) {
    with_lock(Mutex) {
        MinValue = Min<float>(MinValue, update.MinValue);
        MaxValue = Max<float>(MaxValue, update.MaxValue);
        OutOfDomainValuesCount += update.OutOfDomainValuesCount;
        Underflow += update.Underflow;
        Overflow += update.Overflow;
        Sum += update.Sum;
        SumSqr += update.SumSqr;
        ObjectCount += update.ObjectCount;
    }
}

template<typename TStatistic>
void SetCustomBorders(const TFeatureCustomBorders& customBorders, TVector<TStatistic>* statistics) {
    for (const auto& [key, value] : customBorders) {
        CB_ENSURE(key < statistics->size(), "id " << key << " statistics size " << statistics->size());
        statistics->at(key).SetCustomBorders(value);
    }
}

void TTargetsStatistics::Init(const TDataMetaInfo& metaInfo, const TFeatureCustomBorders& customBorders) {
    TargetType = metaInfo.TargetType;
    TargetCount = metaInfo.TargetCount;
    switch (TargetType) {
        case ERawTargetType::Float:
            FloatTargetStatistics.resize(TargetCount);
            break;
        case ERawTargetType::Boolean:
        case ERawTargetType::Integer:
        case ERawTargetType::String:
            DiscreteTargetStatistics.resize(TargetCount);
            for (ui32 i = 0; i < TargetCount; ++i) {
                DiscreteTargetStatistics[i].TargetType = TargetType;
            }
            break;
        default:
            CB_ENSURE(TargetType == ERawTargetType::None, "Unexpected target type " << TargetType << ", expected 'None'");
            CATBOOST_DEBUG_LOG << "No target was found for Target statistics\n";
            break;
    }
    SetCustomBorders(customBorders, &FloatTargetStatistics);
}

void TTargetsStatistics::Update(ui32 flatTargetIdx, TStringBuf value) {
    DiscreteTargetStatistics[flatTargetIdx].Update(value);
}

bool TTargetsStatistics::operator==(const TTargetsStatistics& a) const {
    return (
        std::tie(TargetType, TargetCount, FloatTargetStatistics, DiscreteTargetStatistics) ==
        std::tie(a.TargetType, a.TargetCount, a.FloatTargetStatistics, a.DiscreteTargetStatistics)
    );
}

void TTargetsStatistics::Update(ui32 flatTargetIdx, float value) {
    FloatTargetStatistics[flatTargetIdx].Update(value);
}

void TTargetsStatistics::Update(ui32 flatTargetIdx, ui32 value) {
    DiscreteTargetStatistics[flatTargetIdx].Update(value);
}

NJson::TJsonValue TTargetsStatistics::ToJson() const {
    NJson::TJsonValue result;
    InsertEnumType("TargetType", TargetType, &result);
    result.InsertValue("TargetCount", TargetCount);

    switch (TargetType) {
        case ERawTargetType::Float:
            result.InsertValue("TargetStatistics", AggregateStatistics(FloatTargetStatistics));
            break;
        case ERawTargetType::Boolean:
        case ERawTargetType::Integer:
        case ERawTargetType::String:
            result.InsertValue("TargetStatistics", AggregateStatistics(DiscreteTargetStatistics));
            break;
        default:
            CB_ENSURE(TargetType == ERawTargetType::None, "Unexpected target type " << TargetType << ", expected 'None'");
            CATBOOST_DEBUG_LOG << "No target was found, Target statistics is empty\n";
            break;
    }
    return result;
}

void TDiscreteTargetStatistic::Update(TStringBuf feature) {
    CB_ENSURE(TargetType == ERawTargetType::String, "bad target type");
    with_lock(Mutex) {
        StringTargets[feature]++;
    }
}

void TDiscreteTargetStatistic::Update(ui32 feature) {
    with_lock(Mutex) {
        IntegerTargets[feature]++;
    }
}

void TDiscreteTargetStatistic::Update(const TDiscreteTargetStatistic& update) {
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


template <class T>
void OutputSorted(const THashMap<T, ui64>& targets, NJson::TJsonValue* targetsDistribution) {
    TVector<std::pair<T, ui64>> sorted;
    sorted.reserve(targets.size());
    for (auto const& element : targets) {
        sorted.push_back(element);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    for (auto const& [value, count] : sorted) {
        NJson::TJsonValue stats;
        stats.InsertValue("Value", value);
        stats.InsertValue("Count", count);
        targetsDistribution->AppendValue(stats);
    }
}

NJson::TJsonValue TDiscreteTargetStatistic::ToJson() const {
    NJson::TJsonValue result;
    NJson::TJsonValue targetsDistribution;
    targetsDistribution.SetType(NJson::EJsonValueType::JSON_ARRAY);
    InsertEnumType("TargetType", TargetType, &result);
    switch (TargetType) {
        case ERawTargetType::String:
            OutputSorted(StringTargets, &targetsDistribution);
            break;
        case ERawTargetType::Boolean:
        case ERawTargetType::Integer:
            OutputSorted(IntegerTargets, &targetsDistribution);
            break;
        default:
            CB_ENSURE(false);
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

void TFeatureStatistics::Init(
    const TDataMetaInfo& metaInfo,
    const TFeatureCustomBorders& customBorders
) {
    FloatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetFloatFeatureCount());
    CatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetCatFeatureCount());
    TextFeatureStatistics.resize(metaInfo.FeaturesLayout->GetTextFeatureCount());

    SetCustomBorders(customBorders, &FloatFeatureStatistics);
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


TGroupwiseStats& TGroupwiseStats::operator=(TGroupwiseStats& rhs) {
    GroupsTotalSize = rhs.GroupsTotalSize;
    GroupsTotalSqrSize = rhs.GroupsTotalSqrSize;
    GroupsMaxSize = rhs.GroupsMaxSize;
    GroupsCount = rhs.GroupsCount;
    return *this;
}

TGroupwiseStats& TGroupwiseStats::operator=(TGroupwiseStats&& rhs) {
    GroupsTotalSize = rhs.GroupsTotalSize;
    GroupsTotalSqrSize = rhs.GroupsTotalSqrSize;
    GroupsMaxSize = rhs.GroupsMaxSize;
    GroupsCount = rhs.GroupsCount;
    return *this;
}

void TGroupwiseStats::Update(TGroupId groupId) {
    with_lock(Mutex) {
        ++GroupSizes[groupId];
    }
}

void TGroupwiseStats::Flush() {
    for (const auto& [groupId, value] : GroupSizes) {
        GroupsTotalSize += value;
        GroupsTotalSqrSize += value * value;
        if (value > GroupsMaxSize) {
            GroupsMaxSize = value;
        }
    }
    GroupsCount += GroupSizes.size();
    GroupSizes.clear();
}

NJson::TJsonValue TGroupwiseStats::ToJson() const {
    NJson::TJsonValue stats;
    stats["GroupsCount"] = GroupsCount;
    stats["GroupsTotalSize"] = GroupsTotalSize;
    stats["GroupsMaxSize"] = GroupsMaxSize;
    if (GroupsCount) {
        stats["GroupsAverageSize"] = GetAverageGroupSize();
        stats["GroupAverageSqrSize"] = GetAverageGroupSqrSize();
    }
    return stats;
}

void TGroupwiseStats::InfoLog() const {
    CATBOOST_INFO_LOG << "GroupsCount: " << GroupsCount
        << "\nGroupsTotalSize: " << GroupsTotalSize
        << "\nGroupsMaxSize: " << GroupsMaxSize
        << Endl;
    if (GroupsCount) {
        CATBOOST_INFO_LOG << "GroupsAverageSize: " << GetAverageGroupSize()
            << "\nGroupsAverageSqrSize: " << GetAverageGroupSqrSize() << Endl;
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

    if (TargetHistogram.Defined()) {
        result.InsertValue("TargetHistogram", AggregateStatistics(TargetHistogram.GetRef()));
    }
    result.InsertValue("ObjectCount", ObjectsCount);

    return result;
}

void TDatasetStatistics::Update(const TDatasetStatistics& update) {
    FeatureStatistics.Update(update.FeatureStatistics);
    TargetsStatistics.Update(update.TargetsStatistics);
    SampleIdStatistics.Update(update.SampleIdStatistics);
    ObjectsCount += update.ObjectsCount;
}
