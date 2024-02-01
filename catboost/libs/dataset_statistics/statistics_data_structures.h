#pragma once

#include "histograms.h"

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/json_helpers.h>

#include <library/cpp/json/json_writer.h>
#include <library/cpp/binsaver/bin_saver.h>

#include <util/ysaveload.h>
#include <util/generic/hash.h>
#include <util/system/mutex.h>
#include <util/generic/algorithm.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
#include <util/generic/set.h>
#include <util/stream/fwd.h>


#include <limits>

namespace NCB {

using TFeatureCustomBorders = THashMap<ui32, std::pair<double, double>>;

template <typename T>
NJson::TJsonValue AggregateStatistics(const TVector<T>& data) {
    TVector<NJson::TJsonValue> statistics;
    for (const auto& item : data) {
        statistics.emplace_back(item.ToJson());
    }
    return VectorToJson(statistics);
}

struct IStatistics {
public:
    virtual ~IStatistics() = default;

    virtual NJson::TJsonValue ToJson() const = 0;
};

struct TFloatFeatureStatistics : public IStatistics {
    TFloatFeatureStatistics(TFloatFeatureStatistics&&) noexcept = default;

    TFloatFeatureStatistics();

    TFloatFeatureStatistics(const TFloatFeatureStatistics& a)
        : MinValue(a.MinValue)
        , MaxValue(a.MaxValue)
        , CustomMin(a.CustomMin)
        , CustomMax(a.CustomMax)
        , OutOfDomainValuesCount(a.OutOfDomainValuesCount)
        , Underflow(a.Underflow)
        , Overflow(a.Overflow)
        , Sum(a.Sum)
        , SumSqr(a.SumSqr)
        , ObjectCount(a.ObjectCount)
    {}

    void Update(float feature);

    NJson::TJsonValue ToJson() const override;

    void Update(const TFloatFeatureStatistics& update);

    bool operator==(const TFloatFeatureStatistics& rhs) const;

    ui64 GetObjectCount() const {
        return ObjectCount;
    }

    void SetCustomBorders(const std::pair<float, float>& customBorders) {
        CustomMin = customBorders.first;
        CustomMax = customBorders.second;
    }

    double GetMinBorder() const;
    double GetMaxBorder() const;

    Y_SAVELOAD_DEFINE(
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
    );

    SAVELOAD(
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
    );

public:
    double MinValue;
    double MaxValue;
    TMaybe<double> CustomMin;
    TMaybe<double> CustomMax;
    ui64 OutOfDomainValuesCount;
    ui64 Underflow;
    ui64 Overflow;
    long double Sum;
    long double SumSqr;
    ui64 ObjectCount;
private:
    TMutex Mutex;
};

struct TSampleIdStatistics : public IStatistics {
    TSampleIdStatistics()
        : ObjectCount(0)
        , SumLen(0)
    {}

    TSampleIdStatistics(TSampleIdStatistics&&) noexcept = default;

    TSampleIdStatistics(const TSampleIdStatistics& a)
        : ObjectCount(a.ObjectCount)
        , SumLen(a.SumLen)
    {}

    bool operator==(const TSampleIdStatistics& rhs) const {
        return (
            std::tie(SumLen, ObjectCount) == std::tie(rhs.SumLen, rhs.ObjectCount)
        );
    }

    void Update(const TString& value);

    NJson::TJsonValue ToJson() const override;

    void Update(const TSampleIdStatistics& update);

    Y_SAVELOAD_DEFINE(SumLen, ObjectCount);

    SAVELOAD(SumLen, ObjectCount);

public:
    ui64 ObjectCount;
    ui64 SumLen;
private:
    TMutex Mutex;
};


struct TCatFeatureStatistics: public IStatistics {
    TCatFeatureStatistics(TCatFeatureStatistics&&) noexcept = default;

    TCatFeatureStatistics() = default;

    TCatFeatureStatistics(const TCatFeatureStatistics& a)
        : ImperfectHashSet(a.ImperfectHashSet)
    {}

    bool operator==(const TCatFeatureStatistics& rhs) const {
        return ImperfectHashSet == rhs.ImperfectHashSet;
    }

    void Update(TStringBuf value);
    void Update(ui32 value);

    NJson::TJsonValue ToJson() const override;

    void Update(const TCatFeatureStatistics& update);

    Y_SAVELOAD_DEFINE(ImperfectHashSet);

    SAVELOAD(ImperfectHashSet);

public:
    TSet<ui32> ImperfectHashSet;

private:
    TMutex Mutex;
};

struct TTextFeatureStatistics: public IStatistics {
    TTextFeatureStatistics(TTextFeatureStatistics&&) noexcept = default;

    TTextFeatureStatistics()
        : IsConst(true)
        , Example(Nothing())
    {}

    TTextFeatureStatistics(const TTextFeatureStatistics& a)
        : IsConst(a.IsConst)
        , Example(a.Example)
    {}

    bool operator==(const TTextFeatureStatistics& rhs) const {
        return (
            std::tie(IsConst, Example) == std::tie(rhs.IsConst, rhs.Example)
        );
    }

    void Update(TStringBuf value);

    NJson::TJsonValue ToJson() const override;

    void Update(const TTextFeatureStatistics& update);

    Y_SAVELOAD_DEFINE(IsConst, Example);

    SAVELOAD(IsConst, Example);

public:
    bool IsConst;
    TMaybe<TString> Example;
private:
    TMutex Mutex;
};

struct TFloatTargetStatistic : TFloatFeatureStatistics {
};

struct TDiscreteTargetStatistic : public IStatistics {
    TDiscreteTargetStatistic() = default;

    TDiscreteTargetStatistic(const TDiscreteTargetStatistic& a)
        : StringTargets(a.StringTargets)
        , IntegerTargets(a.IntegerTargets)
        , TargetType(a.TargetType)
    {}

    bool operator==(const TDiscreteTargetStatistic& a) const {
        return (
            std::tie(TargetType, StringTargets, IntegerTargets) ==
            std::tie(a.TargetType, a.StringTargets, a.IntegerTargets)
        );
    }

    void Update(TStringBuf feature);

    void Update(ui32 feature);

    NJson::TJsonValue ToJson() const override;

    void Update(const TDiscreteTargetStatistic& update);

    Y_SAVELOAD_DEFINE(StringTargets, IntegerTargets, TargetType);

    SAVELOAD(StringTargets, IntegerTargets, TargetType);

    ui64 GetObjectCount() const {
        ui64 result = 0;
        for (const auto& x : IntegerTargets) {
            result += x.second;
        }
        for (const auto& x : StringTargets) {
            result += x.second;
        }
        return result;
    }

public:
    THashMap<TString, ui64> StringTargets;
    THashMap<ui32, ui64> IntegerTargets;
    ERawTargetType TargetType;
    TMutex Mutex;
};

struct TTargetsStatistics : public IStatistics {
public:
    TTargetsStatistics() {};

    bool operator==(const TTargetsStatistics& a) const;

    void Init(const TDataMetaInfo& metaInfo, const TFeatureCustomBorders& customBorders);

    NJson::TJsonValue ToJson() const override;

    void Update(ui32 flatTargetIdx, TStringBuf value);
    void Update(ui32 flatTargetIdx, ui32 value);    // use for boolean as well
    void Update(ui32 flatTargetIdx, float value);

    void Update(const TTargetsStatistics& update) {
        CB_ENSURE(FloatTargetStatistics.size() == update.FloatTargetStatistics.size());
        CB_ENSURE(DiscreteTargetStatistics.size() == update.DiscreteTargetStatistics.size());
        for (size_t i = 0; i < FloatTargetStatistics.size(); ++i) {
            FloatTargetStatistics[i].Update(update.FloatTargetStatistics[i]);
        }
        for (size_t i = 0; i < DiscreteTargetStatistics.size(); ++i) {
            DiscreteTargetStatistics[i].Update(update.DiscreteTargetStatistics[i]);
        }
    }

    ui64 GetObjectCount() const {
        if (FloatTargetStatistics.empty() && DiscreteTargetStatistics.empty()) {
            return 0;
        }
        switch (TargetType) {
            case ERawTargetType::Float:
                return FloatTargetStatistics[0].GetObjectCount();
            case ERawTargetType::Boolean:
            case ERawTargetType::Integer:
            case ERawTargetType::String:
                return DiscreteTargetStatistics[0].GetObjectCount();
            default:
                break;
        }
        Y_ASSERT(false);
        return 0;
    }

    Y_SAVELOAD_DEFINE(
        FloatTargetStatistics,
        DiscreteTargetStatistics,
        TargetType,
        TargetCount
    );

    SAVELOAD(
        FloatTargetStatistics,
        DiscreteTargetStatistics,
        TargetType,
        TargetCount
    );

public:
    TVector<TFloatTargetStatistic> FloatTargetStatistics;
    TVector<TDiscreteTargetStatistic> DiscreteTargetStatistics;
    ERawTargetType TargetType;
    ui32 TargetCount;
};

struct TFeatureStatistics : public IStatistics {
public:
    bool operator==(const TFeatureStatistics& a) const;

    Y_SAVELOAD_DEFINE(
        FloatFeatureStatistics,
        CatFeatureStatistics,
        TextFeatureStatistics
    );

    SAVELOAD(
        FloatFeatureStatistics,
        CatFeatureStatistics,
        TextFeatureStatistics
    );

    TVector<TFloatFeatureStatistics> FloatFeatureStatistics;
    TVector<TCatFeatureStatistics> CatFeatureStatistics;
    TVector<TTextFeatureStatistics> TextFeatureStatistics;

    void Init(
        const TDataMetaInfo& metaInfo,
        const TFeatureCustomBorders& customBorders);

    NJson::TJsonValue ToJson() const override;

    void Update(const TFeatureStatistics& update);
};

struct TGroupwiseStats {
    TGroupwiseStats() = default;
    TGroupwiseStats(TGroupwiseStats& rhs)
        : GroupsTotalSize(rhs.GroupsTotalSize)
        , GroupsTotalSqrSize(rhs.GroupsTotalSqrSize)
        , GroupsMaxSize(rhs.GroupsMaxSize)
        , GroupsCount(rhs.GroupsCount)
    {}

    TGroupwiseStats(TGroupwiseStats&& rhs) = default;

    TGroupwiseStats& operator=(TGroupwiseStats& rhs);
    TGroupwiseStats& operator=(TGroupwiseStats&& rhs);

    bool operator==(const TGroupwiseStats& a) const {
        return std::tie(GroupsTotalSize, GroupsTotalSqrSize, GroupsMaxSize, GroupsCount) ==
            std::tie(a.GroupsTotalSize, a.GroupsTotalSqrSize, a.GroupsMaxSize, a.GroupsCount);
    }

    double GetAverageGroupSize() const {
        return static_cast<long double>(GroupsTotalSize) / static_cast<long double>(GroupsCount);
    }

    double GetAverageGroupSqrSize() const {
        return static_cast<long double>(GroupsTotalSqrSize) / static_cast<long double>(GroupsCount);
    }

    void Update(TGroupId groupId);

    void Flush();

    NJson::TJsonValue ToJson() const;

    void InfoLog() const;

    Y_SAVELOAD_DEFINE(
        GroupsTotalSize,
        GroupsTotalSqrSize,
        GroupsMaxSize,
        GroupsCount
    );

    SAVELOAD(
        GroupsTotalSize,
        GroupsTotalSqrSize,
        GroupsMaxSize,
        GroupsCount
    );

public:
    // for updating: groupId -> size
    THashMap<TGroupId, ui64> GroupSizes;

    // result
    ui64 GroupsTotalSize = 0;
    ui64 GroupsTotalSqrSize = 0;
    ui64 GroupsMaxSize = 0;
    ui64 GroupsCount = 0;
private:
    TMutex Mutex;
};

struct TDatasetStatistics {
public:
    bool operator==(const TDatasetStatistics& a) const {
        return std::tie(
            FeatureStatistics,
            TargetsStatistics,
            SampleIdStatistics,
            GroupwiseStats,
            TargetHistogram,
            ClassNames,
            ObjectsCount
        )
        == std::tie(
            a.FeatureStatistics,
            a.TargetsStatistics,
            a.SampleIdStatistics,
            a.GroupwiseStats,
            a.TargetHistogram,
            a.ClassNames,
            a.ObjectsCount
        );
    }

    Y_SAVELOAD_DEFINE(
        FeatureStatistics,
        TargetsStatistics,
        SampleIdStatistics,
        GroupwiseStats,
        TargetHistogram,
        ClassNames,
        ObjectsCount
    );

    SAVELOAD(
        FeatureStatistics,
        TargetsStatistics,
        SampleIdStatistics,
        GroupwiseStats,
        TargetHistogram,
        ClassNames,
        ObjectsCount
    );

    TFeatureStatistics FeatureStatistics;
    TTargetsStatistics TargetsStatistics;
    TSampleIdStatistics SampleIdStatistics;
    TMaybe<TGroupwiseStats> GroupwiseStats;

    TMaybe<TVector<TFloatFeatureHistogram>> TargetHistogram;
    TVector<TString> ClassNames;
    ui64 ObjectsCount = 0;

    void Init(
        const TDataMetaInfo& metaInfo,
        const TFeatureCustomBorders& customBorders,
        const TFeatureCustomBorders& targetCustomBorders
    ) {
        FeatureStatistics.Init(metaInfo, customBorders);
        TargetsStatistics.Init(metaInfo, targetCustomBorders);
        if (metaInfo.HasGroupId) {
            GroupwiseStats = MakeMaybe(TGroupwiseStats());
        }
        ObjectsCount = 0;
    }

    void SetTargetHistogram(const TVector<TFloatFeatureHistogram>& targetHistogram) {
        TargetHistogram = targetHistogram;
    }

    NJson::TJsonValue ToJson() const;

    void Update(const TDatasetStatistics& update);
    ui64 GetObjectCount() const {
        return ObjectsCount;
    }
};
}
