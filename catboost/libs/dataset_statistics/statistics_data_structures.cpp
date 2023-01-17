#include "statistics_data_structures.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/helpers/json_helpers.h>

using namespace NCB;


TFloatFeatureStatistics::TFloatFeatureStatistics()
    : MinValue(std::numeric_limits<double>::max())
    , MaxValue(std::numeric_limits<double>::lowest())
    , CustomMin(std::numeric_limits<double>::lowest())
    , CustomMax(std::numeric_limits<double>::max())
    , OutOfDomainValuesCount(0)
    , Sum(0.)
    , SumSqr(0.)
    , ObjectCount(0)
{
}

void TFloatFeatureStatistics::Update(float feature) {
    with_lock(Mutex) {
        if (std::isinf(feature) || std::isnan(feature)) {
            OutOfDomainValuesCount++;
            return;
        }
        if (feature < CustomMin) {
            OutOfDomainValuesCount++;
            return;
        }
        if (feature > CustomMax) {
            OutOfDomainValuesCount++;
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
        std::tie(MinValue, MaxValue, CustomMin, CustomMax, OutOfDomainValuesCount, Sum, SumSqr, ObjectCount) ==
        std::tie(rhs.MinValue, rhs.MaxValue, rhs.CustomMin, rhs.CustomMax, rhs.OutOfDomainValuesCount, rhs.Sum, SumSqr, rhs.ObjectCount)
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
    result.InsertValue("SumSqr", ToString(SumSqr));
    result.InsertValue("ObjectCount", ObjectCount);
    if (CustomMin != std::numeric_limits<double>::lowest()) {
        result.InsertValue("CustomMin", CustomMin);
    }
    if (CustomMax != std::numeric_limits<double>::max()) {
        result.InsertValue("CustomMax", CustomMax);
    }
    result.InsertValue("OutOfDomainValuesCount", OutOfDomainValuesCount);
    return result;
}

void TFloatFeatureStatistics::Update(const TFloatFeatureStatistics& update) {
    with_lock(Mutex) {
        MinValue = Min<float>(MinValue, update.MinValue);
        MaxValue = Max<float>(MaxValue, update.MaxValue);
        OutOfDomainValuesCount += update.OutOfDomainValuesCount;
        Sum += update.Sum;
        SumSqr += update.SumSqr;
        ObjectCount += update.ObjectCount;
    }
}

void TTargetsStatistics::Init(const TDataMetaInfo& metaInfo, const TVector<TMaybe<std::pair<float, float>>>& customBorders) {
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
            CB_ENSURE(false);
    }

    CB_ENSURE(customBorders.empty() || customBorders.size() == FloatTargetStatistics.size() + StringTargetStatistics.size());
    for (size_t idx = 0; idx < customBorders.size(); ++idx) {
        if (customBorders[idx].Defined()) {
            CB_ENSURE(TargetType == ERawTargetType::Float);
            FloatTargetStatistics[idx].SetCustomBorders(customBorders[idx].GetRef());
        }
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
            CB_ENSURE(false);
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

void TFloatFeaturePairwiseProduct::Init(ui32 featureCount, bool calculatePairwiseStatistics) {
    IsCalculated = calculatePairwiseStatistics;
    if (!calculatePairwiseStatistics) {
        return;
    }
    Y_ASSERT(false);
    PairwiseProduct.resize(featureCount * (featureCount - 1) / 2, 0);
    FeatureCount = featureCount;
    PairwiseProductDocsUsed = 0;
}

void TFloatFeaturePairwiseProduct::Update(TConstArrayRef<float> features) {
    CB_ENSURE(IsCalculated);
    TVector<long double> add(Min(PairwiseProduct.size(), size_t(1000)));
    ui32 resIdx = 0;
    ui32 idx = 0;
    for (ui32 i = 0; i < FeatureCount; ++i) {
        for (ui32 j = i + 1; j < FeatureCount; ++j) {
            add[idx++] = features[i] * features[j];

            if (idx == add.size()) {
                idx = 0;
                with_lock(Mutex) {
                    for (; idx < add.size(); ++idx) {
                        PairwiseProduct[resIdx + idx] += add[idx];
                    }
                }
                resIdx += add.size();
                idx = 0;
            }
        }
    }

    with_lock(Mutex) {
        idx = 0;
        for (; resIdx < PairwiseProduct.size(); ++resIdx) {
            PairwiseProduct[resIdx] += add[idx++];
        }
        PairwiseProductDocsUsed++;
    }
}

void TFloatFeaturePairwiseProduct::Update(const TFloatFeaturePairwiseProduct& update) {
    CB_ENSURE(IsCalculated == update.IsCalculated);
    if (!IsCalculated) {
        return;
    }
    CB_ENSURE(FeatureCount == update.FeatureCount);
    for (ui32 idx = 0; idx < PairwiseProduct.size(); ++idx) {
        PairwiseProduct[idx] += update.PairwiseProduct[idx];
    }
    PairwiseProductDocsUsed += update.PairwiseProductDocsUsed;
}

bool TFloatFeaturePairwiseProduct::operator==(const TFloatFeaturePairwiseProduct& rhs) const {
    return (
        std::tie(PairwiseProduct, PairwiseProductDocsUsed, FeatureCount, IsCalculated) ==
        std::tie(rhs.PairwiseProduct, rhs.PairwiseProductDocsUsed, rhs.FeatureCount, rhs.IsCalculated)
    );
}

NJson::TJsonValue TFloatFeaturePairwiseProduct::ToJson(const TVector<TFloatFeatureStatistics>& featureStats) const {
    if (!IsCalculated) {
        return NJson::TJsonValue("NotCalculated");
    }
    CB_ENSURE(FeatureCount == featureStats.size(), "" << FeatureCount << " != " << featureStats.size());
    TVector<TVector<NJson::TJsonValue>> matrix(FeatureCount, TVector<NJson::TJsonValue>(FeatureCount));
    ui32 idx = 0;
    for (ui32 i = 0; i < FeatureCount; ++i) {
        for (ui32 j = i + 1; j < FeatureCount; ++j) {
            matrix[i][j] = ToString(PairwiseProduct[idx]);
            matrix[j][i] = ToString(PairwiseProduct[idx]);
            idx++;
        }
        matrix[i][i] = ToString(featureStats[i].SumSqr);
    }
    TVector<NJson::TJsonValue> vectorOfJsons;
    for (ui32 i = 0; i < FeatureCount; ++i) {
        vectorOfJsons.emplace_back(VectorToJson(matrix[i]));
    }
    return VectorToJson(vectorOfJsons);
}

void TFeatureStatistics::Init(
    const TDataMetaInfo& metaInfo,
    const TVector<TMaybe<std::pair<float, float>>>& customBorders,
    bool calculatePairwiseStatistics
) {
    FloatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetFloatFeatureCount());
    FloatFeaturePairwiseProduct.Init(metaInfo.FeaturesLayout->GetFloatFeatureCount(), calculatePairwiseStatistics);
    CatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetCatFeatureCount());
    TextFeatureStatistics.resize(metaInfo.FeaturesLayout->GetTextFeatureCount());

    CB_ENSURE(customBorders.empty() || customBorders.size() == FloatFeatureStatistics.size(),
              "Not equal" << customBorders.size() << "!="  << FloatFeatureStatistics.size());
    for (size_t idx = 0; idx < customBorders.size(); ++idx) {
        if (customBorders[idx].Defined()) {
            FloatFeatureStatistics[idx].SetCustomBorders(customBorders[idx].GetRef());
        }
    }
}

NJson::TJsonValue TFeatureStatistics::ToJson() const {
    NJson::TJsonValue result;
    result.InsertValue("FloatFeatureStatistics", AggregateStatistics(FloatFeatureStatistics));
    if (FloatFeaturePairwiseProduct.IsCalculated) {
        result.InsertValue("FloatFeaturePairwiseProductSum", FloatFeaturePairwiseProduct.ToJson(FloatFeatureStatistics));
        result.InsertValue("PairwiseProductDocsUsed", FloatFeaturePairwiseProduct.PairwiseProductDocsUsed);
    }
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
    FloatFeaturePairwiseProduct.Update(update.FloatFeaturePairwiseProduct);
}

bool TFeatureStatistics::operator==(const TFeatureStatistics& a) const {
    return (
        std::tie(FloatFeatureStatistics, CatFeatureStatistics, TextFeatureStatistics, FloatFeaturePairwiseProduct) ==
        std::tie(a.FloatFeatureStatistics, a.CatFeatureStatistics, a.TextFeatureStatistics, FloatFeaturePairwiseProduct)
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
    result.InsertValue("ObjectCount", TargetsStatistics.GetObjectCount());

    return result;
}

void TDatasetStatistics::Update(const TDatasetStatistics& update) {
    FeatureStatistics.Update(update.FeatureStatistics);
    TargetsStatistics.Update(update.TargetsStatistics);
    SampleIdStatistics.Update(update.SampleIdStatistics);
}
