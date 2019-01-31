#pragma once

#include <catboost/libs/algo/split.h>
#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/model/model.h>
#include <catboost/libs/options/enums.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/digest/multi.h>
#include <util/system/yassert.h>

#include <utility>


struct TRegularFeature {
    EFeatureType Type;
    int Index;

public:
    TRegularFeature(EFeatureType type, int index)
        : Type(type)
        , Index(index)
    {}
};

struct TFeatureEffect {
    double Score = 0;
    TRegularFeature Feature;

public:
    TFeatureEffect() = default;

    TFeatureEffect(double score, EFeatureType type, int index)
        : Score(score)
        , Feature{type, index}
    {}
};

struct TFeatureInteraction {
    double Score = 0;
    TRegularFeature FirstFeature, SecondFeature;

public:
    TFeatureInteraction() = default;

    TFeatureInteraction(double score, EFeatureType firstFeatureType, int firstFeatureIndex,
                   EFeatureType secondFeatureType, int secondFeatureIndex)
        : Score(score)
        , FirstFeature{firstFeatureType, firstFeatureIndex}
        , SecondFeature{secondFeatureType, secondFeatureIndex}
    {}
};

struct TFeature {
    ESplitType Type;
    int FeatureIdx;
    TModelCtr Ctr;
    static constexpr size_t FloatFeatureBaseHash = 12321;
    static constexpr size_t CtrBaseHash = 89321;
    static constexpr size_t OneHotFeatureBaseHash = 517931;

public:
    TFeature() = default;
    TFeature(const TFloatFeature& feature) : Type(ESplitType::FloatFeature), FeatureIdx(feature.FeatureIndex) {}
    TFeature(const TOneHotFeature& feature) : Type(ESplitType::OneHotFeature), FeatureIdx(feature.CatFeatureIndex) {}
    TFeature(const TCtrFeature& feature) : Type(ESplitType::OnlineCtr), Ctr(feature.Ctr) {}

    bool operator==(const TFeature& other) const {
        if (Type != other.Type) {
            return false;
        }
        if (Type == ESplitType::OnlineCtr) {
            return Ctr == other.Ctr;
        } else {
            return FeatureIdx == other.FeatureIdx;
        }
    }
    bool operator!=(const TFeature& other) const {
        return !(*this == other);
    }
    size_t GetHash() const {
        if (Type == ESplitType::FloatFeature) {
            return MultiHash(FloatFeatureBaseHash, FeatureIdx);
        } else if (Type == ESplitType::OnlineCtr) {
            return MultiHash(CtrBaseHash, Ctr.GetHash());
        } else {
            Y_ASSERT(Type == ESplitType::OneHotFeature);
            return MultiHash(OneHotFeatureBaseHash, FeatureIdx);
        }
    }
    TString BuildDescription(const NCB::TFeaturesLayout& layout) const;
};

struct TInternalFeatureInteraction {
    double Score = 0;
    TFeature FirstFeature, SecondFeature;

public:
    TInternalFeatureInteraction(double score, const TFeature& firstFeature, const TFeature& secondFeature)
        : Score(score)
        , FirstFeature(firstFeature)
        , SecondFeature(secondFeature)
    {}
};

TVector<std::pair<double, TFeature>> CalcFeatureEffect(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset, // can be nullptr
    EFstrType type,
    NPar::TLocalExecutor* localExecutor);

TVector<TFeatureEffect> CalcRegularFeatureEffect(
    const TVector<std::pair<double, TFeature>>& effect,
    int catFeaturesCount,
    int floatFeaturesCount);

TVector<double> CalcRegularFeatureEffect(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset, // can be nullptr
    EFstrType type,
    NPar::TLocalExecutor* localExecutor);

TVector<TInternalFeatureInteraction> CalcInternalFeatureInteraction(const TFullModel& model);
TVector<TFeatureInteraction> CalcFeatureInteraction(
    const TVector<TInternalFeatureInteraction>& internalFeatureInteraction,
    const NCB::TFeaturesLayout& layout);

TVector<TVector<double>> CalcInteraction(const TFullModel& model);
TVector<TVector<double>> GetFeatureImportances(
    const TString& type,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset, // can be nullptr
    int threadCount,
    int logPeriod = 0);

TVector<TVector<TVector<double>>> GetFeatureImportancesMulti(
    const TString& type,
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset,
    int threadCount,
    int logPeriod = 0);


/*
 * model is the primary source of featureIds,
 * if model does not contain featureIds data then try to get this data from pool (if provided (non nullptr))
 * for all remaining features without id generated featureIds will be just their external indices
 * (indices in original training dataset)
 */
TVector<TString> GetMaybeGeneratedModelFeatureIds(
    const TFullModel& model,
    const NCB::TDataProviderPtr dataset); // can be nullptr
