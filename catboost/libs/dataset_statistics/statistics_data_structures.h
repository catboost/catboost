#pragma once

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
template<typename T>
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

    virtual NJson::TJsonValue ToJson() const {
        CB_ENSURE(false, "Not implemented");
        NJson::TJsonValue result;
        return result;
    }
};

struct TFloatFeatureStatistics : public IStatistics {
    TFloatFeatureStatistics(TFloatFeatureStatistics&&) noexcept = default;

    TFloatFeatureStatistics();

    TFloatFeatureStatistics(const TFloatFeatureStatistics& a)
        : MinValue(a.MinValue), MaxValue(a.MaxValue), Sum(a.Sum), SumSqr(a.SumSqr), ObjectCount(a.ObjectCount) {}

    void Update(float feature);

    NJson::TJsonValue ToJson() const;

    void Update(const TFloatFeatureStatistics& update);

    bool operator==(const TFloatFeatureStatistics& rhs) const;

    ui64 GetObjectCount() const {
        return ObjectCount;
    }

    Y_SAVELOAD_DEFINE(MinValue, MaxValue, Sum, SumSqr, ObjectCount);

    SAVELOAD(MinValue, MaxValue, Sum, SumSqr, ObjectCount);

    double MinValue;
    double MaxValue;
    long double Sum;
    long double SumSqr;
    ui64 ObjectCount;
private:
    TMutex Mutex;
};

struct TFloatFeaturePairwiseProduct {
    TFloatFeaturePairwiseProduct() = default;
    TFloatFeaturePairwiseProduct(TFloatFeaturePairwiseProduct&&) noexcept = default;

    TFloatFeaturePairwiseProduct(const TFloatFeaturePairwiseProduct& a)
        : PairwiseProduct(a.PairwiseProduct)
        , PairwiseProductDocsUsed(a.PairwiseProductDocsUsed)
        , FeatureCount(a.FeatureCount)
    {}

    void Init(ui32 featureCount);

    void Update(TConstArrayRef<float> features);
    void Update(const TFloatFeaturePairwiseProduct& update);

    bool operator==(const TFloatFeaturePairwiseProduct& rhs) const;

    NJson::TJsonValue ToJson(const TVector<TFloatFeatureStatistics>& featureStats) const;

    TVector<long double> PairwiseProduct;
    ui64 PairwiseProductDocsUsed;
    ui32 FeatureCount;

    Y_SAVELOAD_DEFINE(PairwiseProduct, PairwiseProductDocsUsed, FeatureCount);

    SAVELOAD(PairwiseProduct, PairwiseProductDocsUsed, FeatureCount);
private:
    TMutex Mutex;
};

struct TSampleIdStatistics : public IStatistics {
    TSampleIdStatistics() : ObjectCount(0), SumLen(0) {}

    TSampleIdStatistics(TSampleIdStatistics&&) noexcept = default;

    TSampleIdStatistics(const TSampleIdStatistics& a)
        : ObjectCount(a.ObjectCount)
        , SumLen(a.SumLen)
    {}

    void Update(const TString& value);

    NJson::TJsonValue ToJson() const;

    void Update(const TSampleIdStatistics& update);

    bool operator==(const TSampleIdStatistics& rhs) const {
        return (
            std::tie(SumLen, ObjectCount) == std::tie(rhs.SumLen, rhs.ObjectCount)
        );
    }

    Y_SAVELOAD_DEFINE(SumLen, ObjectCount);

    SAVELOAD(SumLen, ObjectCount);

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

    void Update(TStringBuf value);
    void Update(ui32 value);

    NJson::TJsonValue ToJson() const;

    void Update(const TCatFeatureStatistics& update);

    bool operator==(const TCatFeatureStatistics& rhs) const {
        return ImperfectHashSet == rhs.ImperfectHashSet;
    }

    Y_SAVELOAD_DEFINE(ImperfectHashSet);

    SAVELOAD(ImperfectHashSet);

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

    void Update(TStringBuf value);

    NJson::TJsonValue ToJson() const;

    void Update(const TTextFeatureStatistics& update);

    bool operator==(const TTextFeatureStatistics& rhs) const {
        return (
            std::tie(IsConst, Example) == std::tie(rhs.IsConst, rhs.Example)
        );
    }

    Y_SAVELOAD_DEFINE(IsConst, Example);

    SAVELOAD(IsConst, Example);

    bool IsConst;
    TMaybe<TString> Example;
private:
    TMutex Mutex;
};

struct TFloatTargetStatistic : TFloatFeatureStatistics {
};

struct TStringTargetStatistic : public IStatistics {
    TStringTargetStatistic() = default;

    TStringTargetStatistic(const TStringTargetStatistic& a)
        : StringTargets(a.StringTargets), IntegerTargets(a.IntegerTargets), TargetType(a.TargetType) {}

    void Update(TStringBuf feature);

    void Update(ui32 feature);

    NJson::TJsonValue ToJson() const override;

    void Update(const TStringTargetStatistic& update);

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

    bool operator==(const TStringTargetStatistic& a) const {
        return (
            std::tie(TargetType, StringTargets, IntegerTargets) ==
            std::tie(a.TargetType, a.StringTargets, a.IntegerTargets)
        );
    }

    THashMap<TString, ui64> StringTargets;
    THashMap<ui32, ui64> IntegerTargets;
    ERawTargetType TargetType;
    TMutex Mutex;
};

struct TTargetsStatistics : public IStatistics {
public:
    TTargetsStatistics() {};

    void Init(const TDataMetaInfo& metaInfo);

    NJson::TJsonValue ToJson() const override;

    void Update(ui32 flatTargetIdx, TStringBuf value);

    void Update(ui32 flatTargetIdx, float value);

    bool operator==(const TTargetsStatistics& a) const;

    void Update(const TTargetsStatistics& update) {
        CB_ENSURE(FloatTargetStatistics.size() == update.FloatTargetStatistics.size());
        CB_ENSURE(StringTargetStatistics.size() == update.StringTargetStatistics.size());
        for (size_t i = 0; i < FloatTargetStatistics.size(); ++i) {
            FloatTargetStatistics[i].Update(update.FloatTargetStatistics[i]);
        }
        for (size_t i = 0; i < StringTargetStatistics.size(); ++i) {
            StringTargetStatistics[i].Update(update.StringTargetStatistics[i]);
        }
    }

    ui64 GetObjectCount() const {
        if (FloatTargetStatistics.empty() && StringTargetStatistics.empty()) {
            return 0;
        }
        switch (TargetType) {
            case ERawTargetType::Float:
                return FloatTargetStatistics[0].GetObjectCount();
            case ERawTargetType::Integer:
            case ERawTargetType::String:
                return StringTargetStatistics[0].GetObjectCount();
            default:
                break;
        }
        Y_ASSERT(false);
        return 0;
    }

    Y_SAVELOAD_DEFINE(
        FloatTargetStatistics,
        StringTargetStatistics,
        TargetType,
        TargetCount
    );

    SAVELOAD(
        FloatTargetStatistics,
        StringTargetStatistics,
        TargetType,
        TargetCount
    );

    TVector<TFloatTargetStatistic> FloatTargetStatistics;
    TVector<TStringTargetStatistic> StringTargetStatistics;
    ERawTargetType TargetType;
    ui32 TargetCount;
};

struct TFeatureStatistics : public IStatistics {
public:
    Y_SAVELOAD_DEFINE(
        FloatFeatureStatistics,
        CatFeatureStatistics,
        TextFeatureStatistics,
        FloatFeaturePairwiseProduct
    );

    SAVELOAD(
        FloatFeatureStatistics,
        CatFeatureStatistics,
        TextFeatureStatistics,
        FloatFeaturePairwiseProduct
    );

    TVector<TFloatFeatureStatistics> FloatFeatureStatistics;
    TVector<TCatFeatureStatistics> CatFeatureStatistics;
    TVector<TTextFeatureStatistics> TextFeatureStatistics;
    TFloatFeaturePairwiseProduct FloatFeaturePairwiseProduct;

    void Init(const TDataMetaInfo& metaInfo);

    NJson::TJsonValue ToJson() const override;

    void Update(const TFeatureStatistics& update);

    bool operator==(const TFeatureStatistics& a) const;
};

struct TGroupwiseStats {
    double GetAverageGroupSize() const {
        return static_cast<long double>(GroupsTotalSize) / static_cast<long double>(GroupsCount);
    }

    NJson::TJsonValue ToJson() const;

    void InfoLog() const;

    Y_SAVELOAD_DEFINE(
        GroupsTotalSize,
        GroupsCount
    );

    SAVELOAD(
        GroupsTotalSize,
        GroupsCount
    );

    ui64 GroupsTotalSize = 0;
    ui64 GroupsCount = 0;
};

struct TDatasetStatistics {
public:
    Y_SAVELOAD_DEFINE(
        FeatureStatistics,
        TargetsStatistics,
        SampleIdStatistics,
        GroupwiseStats
    );

    SAVELOAD(
        FeatureStatistics,
        TargetsStatistics,
        SampleIdStatistics,
        GroupwiseStats
    );

    bool operator==(const TDatasetStatistics& a) const {
        return std::tie(FeatureStatistics, TargetsStatistics, SampleIdStatistics) ==
               std::tie(a.FeatureStatistics, a.TargetsStatistics, a.SampleIdStatistics);
    }

    TFeatureStatistics FeatureStatistics;
    TTargetsStatistics TargetsStatistics;
    TSampleIdStatistics SampleIdStatistics;
    TMaybe<TGroupwiseStats> GroupwiseStats;

    void Init(const TDataMetaInfo& metaInfo) {
        FeatureStatistics.Init(metaInfo);
        TargetsStatistics.Init(metaInfo);
    }

    NJson::TJsonValue ToJson() const;

    void Update(const TDatasetStatistics& update);
    ui64 GetObjectCount() const {
        return TargetsStatistics.GetObjectCount();
    }
};
}
