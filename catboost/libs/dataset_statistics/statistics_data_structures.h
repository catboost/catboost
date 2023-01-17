#pragma once

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/json_helpers.h>

#include <library/cpp/json/json_writer.h>

#include <util/generic/hash.h>
#include <util/system/mutex.h>
#include <util/generic/vector.h>

#include <limits>

using namespace NCB;

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

    virtual NJson::TJsonValue ToJson() const {
        CB_ENSURE(false, "Not implemented");
        NJson::TJsonValue result;
        return result;
    }
};

struct TFloatFeatureStatistics: public IStatistics {
    TFloatFeatureStatistics(TFloatFeatureStatistics&&) noexcept = default;

    TFloatFeatureStatistics();

    void Update(float feature);

    NJson::TJsonValue ToJson() const;

private:
    double MinValue;
    double MaxValue;
    double Sum;
    ui32 ObjectCount;
    TMutex Mutex;
};

struct TCatFeatureStatistics: public IStatistics {
    void Update(TStringBuf feature) {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(feature);
    }
    void Update(ui32 feature) {
        CB_ENSURE(false, "Not implemented");
        Y_UNUSED(feature);
    }
};

struct TFloatTargetStatistic: TFloatFeatureStatistics {};

struct TStringTargetStatistic: public IStatistics {
    TStringTargetStatistic(ERawTargetType targetType);

    void Update(TStringBuf feature);
    void Update(ui32 feature);

    NJson::TJsonValue ToJson() const override;

private:
    THashMap<TString, ui32> StringTargets;
    THashMap<ui32, ui32> IntegerTargets;
    ERawTargetType TargetType;
    TMutex Mutex;
};

struct TTargetsStatistics: public IStatistics {
public:
    void Init(const TDataMetaInfo& MetaInfo);

    NJson::TJsonValue ToJson() const override;

    void Update(ui32 flatTargetIdx, TStringBuf value);
    void Update(ui32 flatTargetIdx, float value);

private:
    TVector<TFloatTargetStatistic> FloatTargetStatistics;
    TVector<TStringTargetStatistic> StringTargetStatistics;
    ERawTargetType TargetType;
    ui32 TargetCount;
};

struct TTextFeatureStatistics: public IStatistics {};

struct TEmbeddingFeatureStatistics: public IStatistics {};

struct TGroupStatistics: public IStatistics {};

struct TFeatureStatistics: public IStatistics {
public:
    TVector<TFloatFeatureStatistics> FloatFeatureStatistics;         // [floatFeatureIdx]
    TVector<TCatFeatureStatistics> CatFeatureStatistics;             // [catFeatureIdx]
    TVector<TTextFeatureStatistics> TextFeatureStatistics;           // [textFeatureIdx]
    TVector<TEmbeddingFeatureStatistics> EmbeddingFeatureStatistics; // [embeddingFeatureIdx]

    void Init(const TDataMetaInfo& metaInfo) {
        FloatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetFloatFeatureCount());
        CatFeatureStatistics.resize(metaInfo.FeaturesLayout->GetCatFeatureCount());
        TextFeatureStatistics.resize(metaInfo.FeaturesLayout->GetTextFeatureCount());
        EmbeddingFeatureStatistics.resize(metaInfo.FeaturesLayout->GetEmbeddingFeatureCount());
    }

    NJson::TJsonValue ToJson() const override {
        NJson::TJsonValue result;
        result.InsertValue("FloatFeatureStatistics", AggregateStatistics(FloatFeatureStatistics));
        //  ToDo: add statistics for Cat, Text and Embedding features
        return result;
    }
};

struct TDatasetStatistics {
public:
    TAtomicSharedPtr<TVector<THashMap<ui32, TString>>> CatFeaturesHashToString; // [catFeatureIdx]

    TDataMetaInfo MetaInfo;
    TFeatureStatistics FeatureStatistics;
    TTargetsStatistics TargetsStatistics;
    // ToDo: maybe add GroupStatistics

    void Init(const TDataMetaInfo& metaInfo) {
        MetaInfo = metaInfo;

        FeatureStatistics.Init(MetaInfo);
        TargetsStatistics.Init(MetaInfo);
    }
};
