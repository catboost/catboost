#pragma once

#include <catboost/libs/algo/features_layout.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/algo/tree_print.h>

#include <catboost/libs/data/pool.h>
#include <catboost/libs/model/model.h>

#include <util/digest/multi.h>
#include <util/string/builder.h>
#include <catboost/libs/options/enums.h>

struct TRegularFeature {
    EFeatureType Type;
    int Index;
    TRegularFeature(EFeatureType type, int index)
        : Type(type)
        , Index(index) {}
};

struct TFeatureEffect {
    double Score = 0;
    TRegularFeature Feature;

    TFeatureEffect() = default;

    TFeatureEffect(double score, EFeatureType type, int index)
        : Score(score)
        , Feature{type, index} {}
};

struct TFeatureInteraction {
    double Score = 0;
    TRegularFeature FirstFeature, SecondFeature;

    TFeatureInteraction() = default;

    TFeatureInteraction(double score, EFeatureType firstFeatureType, int firstFeatureIndex,
                   EFeatureType secondFeatureType, int secondFeatureIndex)
        : Score(score)
        , FirstFeature{firstFeatureType, firstFeatureIndex}
        , SecondFeature{secondFeatureType, secondFeatureIndex} {}
};

struct TFeature {
    ESplitType Type;
    int FeatureIdx;
    TModelCtr Ctr;
    const size_t FloatFeatureBaseHash = 12321;
    const size_t CtrBaseHash = 89321;
    const size_t OneHotFeatureBaseHash = 517931;
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
    TString BuildDescription(const TFeaturesLayout& layout) const;
};

struct TInternalFeatureInteraction {
    double Score = 0;
    TFeature FirstFeature, SecondFeature;
    TInternalFeatureInteraction(double score, const TFeature& firstFeature, const TFeature& secondFeature)
        : Score(score)
        , FirstFeature(firstFeature)
        , SecondFeature(secondFeature) {}
};

TVector<std::pair<double, TFeature>> CalcFeatureEffect(const TFullModel& model, const TPool& pool, int threadCount = 1);
TVector<TFeatureEffect> CalcRegularFeatureEffect(const TVector<std::pair<double, TFeature>>& effect,
                                                 int catFeaturesCount, int floatFeaturesCount);
TVector<double> CalcRegularFeatureEffect(const TFullModel& model, const TPool& pool, int threadCount = 1);

TVector<TInternalFeatureInteraction> CalcInternalFeatureInteraction(const TFullModel& model);
TVector<TFeatureInteraction> CalcFeatureInteraction(const TVector<TInternalFeatureInteraction>& internalFeatureInteraction,
                                                                                const TFeaturesLayout& layout);

TVector<TVector<double>> CalcFstr(const TFullModel& model, const TPool& pool, int threadCount);
TVector<TVector<double>> CalcInteraction(const TFullModel& model, const TPool& pool);
TVector<TVector<double>> GetFeatureImportances(const TFullModel& model, const TPool& pool, const TString& type, int threadCount);
