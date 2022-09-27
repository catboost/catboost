#pragma once

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/json_helpers.h>

#include <library/cpp/json/json_writer.h>
#include <library/cpp/binsaver/bin_saver.h>

#include <util/ysaveload.h>
#include <util/generic/hash.h>
#include <util/system/mutex.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>
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
        : MinValue(a.MinValue), MaxValue(a.MaxValue), Sum(a.Sum), ObjectCount(a.ObjectCount) {}

    void Update(float feature);

    NJson::TJsonValue ToJson() const;

    void Update(const TFloatFeatureStatistics& update);

    bool operator==(const TFloatFeatureStatistics& rhs) const {
        return (MinValue == rhs.MinValue) && (MaxValue == rhs.MaxValue) &&
               (Sum == rhs.Sum) && (ObjectCount == rhs.ObjectCount);
    }

    ui64 GetObjectCount() const {
        return ObjectCount;
    }

    Y_SAVELOAD_DEFINE(MinValue, MaxValue, Sum, ObjectCount);

    SAVELOAD(MinValue, MaxValue, Sum, ObjectCount);

    double MinValue;
    double MaxValue;
    double Sum;
    ui64 ObjectCount;
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
        return (TargetType == a.TargetType &&
                StringTargets == a.StringTargets &&
                IntegerTargets == a.IntegerTargets);
    }

    THashMap<TString, ui64> StringTargets;
    THashMap<ui32, ui64> IntegerTargets;
    ERawTargetType TargetType;
    TMutex Mutex;
};

struct TTargetsStatistics : public IStatistics {
public:
    TTargetsStatistics() {};

    void Init(const TDataMetaInfo& MetaInfo);

    NJson::TJsonValue ToJson() const override;

    void Update(ui32 flatTargetIdx, TStringBuf value);

    void Update(ui32 flatTargetIdx, float value);

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

    bool operator==(const TTargetsStatistics& a) const {
        if (TargetType != a.TargetType || TargetCount != a.TargetCount ||
            FloatTargetStatistics.size() != a.FloatTargetStatistics.size() ||
            StringTargetStatistics.size() != a.StringTargetStatistics.size()
            ) {
            return false;
        }
        return std::equal(FloatTargetStatistics.begin(), FloatTargetStatistics.end(),
                          a.FloatTargetStatistics.begin()) &&
               std::equal(StringTargetStatistics.begin(), StringTargetStatistics.end(),
                          a.StringTargetStatistics.begin());
    }

private:
    TVector<TFloatTargetStatistic> FloatTargetStatistics;
    TVector<TStringTargetStatistic> StringTargetStatistics;
    ERawTargetType TargetType;
    ui32 TargetCount;
};

struct TGroupStatistics : public IStatistics {
};

struct TFeatureStatistics : public IStatistics {
public:
    Y_SAVELOAD_DEFINE(
        FloatFeatureStatistics
    );

    SAVELOAD(
        FloatFeatureStatistics
    );

    TVector<TFloatFeatureStatistics> FloatFeatureStatistics;         // [floatFeatureIdx]

    void Init(const TDataMetaInfo& metaInfo) {
        FloatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetFloatFeatureCount());
    }

    NJson::TJsonValue ToJson() const override {
        NJson::TJsonValue result;
        result.InsertValue("FloatFeatureStatistics", AggregateStatistics(FloatFeatureStatistics));
        //  ToDo: add statistics for Cat, Text and Embedding features
        return result;
    }

    void Update(const TFeatureStatistics& update) {
        CB_ENSURE(FloatFeatureStatistics.size() == update.FloatFeatureStatistics.size());
        for (ui32 i = 0; i < FloatFeatureStatistics.size(); ++i) {
            FloatFeatureStatistics[i].Update(update.FloatFeatureStatistics[i]);
        }
    }

    bool operator==(const TFeatureStatistics& rhs) const {
        if (FloatFeatureStatistics.size() != rhs.FloatFeatureStatistics.size()) {
            return false;
        }
        for (ui32 i = 0; i < FloatFeatureStatistics.size(); ++i) {
            if (!(FloatFeatureStatistics[i] == rhs.FloatFeatureStatistics[i])) {
                return false;
            }
        }
        return true;
    }
};

struct TDatasetStatistics {
public:
    Y_SAVELOAD_DEFINE(
        CatFeaturesHashToString,
        FeatureStatistics,
        TargetsStatistics
    );

    SAVELOAD(
        CatFeaturesHashToString,
        FeatureStatistics,
        TargetsStatistics
    );

    bool operator==(const TDatasetStatistics& a) const {
        if (CatFeaturesHashToString.size() != a.CatFeaturesHashToString.size() ||
            FeatureStatistics != a.FeatureStatistics ||
            TargetsStatistics != a.TargetsStatistics) {
            return false;
        }
        return CatFeaturesHashToString == a.CatFeaturesHashToString;
    }

    TVector<THashMap<ui32, TString>> CatFeaturesHashToString; // [catFeatureIdx]

    TDataMetaInfo MetaInfo;
    TFeatureStatistics FeatureStatistics;
    TTargetsStatistics TargetsStatistics;
    // ToDo: maybe add GroupStatistics

    void Init(const TDataMetaInfo& metaInfo) {
        MetaInfo = metaInfo;

        FeatureStatistics.Init(metaInfo);
        TargetsStatistics.Init(metaInfo);
    }

    NJson::TJsonValue ToJson() const {
        NJson::TJsonValue result;

        result.InsertValue("TargetsStatistics", TargetsStatistics.ToJson());
        result.InsertValue("FeatureStatistics", FeatureStatistics.ToJson());

        result.InsertValue("ObjectCount", TargetsStatistics.GetObjectCount());

        return result;
    }

    void Update(const TDatasetStatistics& update) {
        CB_ENSURE_INTERNAL(MetaInfo == update.MetaInfo, "Inconsistent metainfo");
        FeatureStatistics.Update(update.FeatureStatistics);
        TargetsStatistics.Update(update.TargetsStatistics);
    }

    ui64 GetObjectCount() const {
        return TargetsStatistics.GetObjectCount();
    }
};
}