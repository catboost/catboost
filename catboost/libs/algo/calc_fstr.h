#pragma once

#include "params.h"
#include <catboost/libs/data/pool.h>
#include "features_layout.h"
#include <catboost/libs/model/model.h>
#include <catboost/libs/model/split.h>
#include <util/digest/multi.h>

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

using TFeature = TSplitCandidate;

struct TInternalFeatureInteraction {
    double Score = 0;
    TFeature FirstFeature, SecondFeature;
    TInternalFeatureInteraction(double score, const TFeature& firstFeature, const TFeature& secondFeature)
        : Score(score)
        , FirstFeature(firstFeature)
        , SecondFeature(secondFeature) {}
};

yvector<std::pair<double, TFeature>> CalcFeatureEffect(const TFullModel& model, const TPool& pool, int threadCount = 1);
yvector<TFeatureEffect> CalcRegularFeatureEffect(const yvector<std::pair<double, TFeature>>& effect,
                                                 int catFeaturesCount, int floatFeaturesCount);
yvector<double> CalcRegularFeatureEffect(const TFullModel& model, const TPool& pool, int threadCount = 1);

yvector<TInternalFeatureInteraction> CalcInternalFeatureInteraction(const TFullModel& model);
yvector<TFeatureInteraction> CalcFeatureInteraction(const yvector<TInternalFeatureInteraction>& internalFeatureInteraction,
                                                                                const TFeaturesLayout& layout);
