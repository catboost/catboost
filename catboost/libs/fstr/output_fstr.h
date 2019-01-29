#pragma once

#include "calc_fstr.h"

#include <catboost/libs/algo/tree_print.h>

#include <util/stream/file.h>
#include <util/system/yassert.h>

#include <utility>


inline void OutputFstr(
    const NCB::TFeaturesLayout& layout,
    const TVector<std::pair<double, TFeature>>& effect,
    const TString& path)
{
    TFileOutput out(path);
    for (const auto& effectWithSplit : effect) {
        out << effectWithSplit.first << "\t" << effectWithSplit.second.BuildDescription(layout) << Endl;
    }
}

inline void OutputRegularFstr(
    const NCB::TFeaturesLayout& layout,
    const TVector<TFeatureEffect>& regularEffect,
    const TString& path)
{
    TFileOutput out(path);
    for (const auto& initialFeatureScore : regularEffect) {
        out << initialFeatureScore.Score << "\t"
            << BuildFeatureDescription(
                layout,
                initialFeatureScore.Feature.Index,
                initialFeatureScore.Feature.Type) << Endl;
    }
}

inline void OutputInteraction(
    const NCB::TFeaturesLayout& layout,
    const TVector<TInternalFeatureInteraction>& interactionValues,
    const TString& path)
{
    TFileOutput out(path);
    for (const auto& interaction : interactionValues) {
        out << interaction.Score << "\t" << interaction.FirstFeature.BuildDescription(layout) << "\t"
            << interaction.SecondFeature.BuildDescription(layout) << Endl;
    }
}

inline void OutputRegularInteraction(
    const NCB::TFeaturesLayout& layout,
    const TVector<TFeatureInteraction>& interactionValues,
    const TString& path)
{
    TFileOutput out(path);
    for (const auto& interaction : interactionValues) {
        out << interaction.Score << "\t"
            << BuildFeatureDescription(layout, interaction.FirstFeature.Index, interaction.FirstFeature.Type)
            << "\t"
            << BuildFeatureDescription(
                layout,
                interaction.SecondFeature.Index,
                interaction.SecondFeature.Type) << Endl;
    }
}

inline void OutputFeatureImportanceMatrix(
    const TVector<TVector<double>>& featureImportance,
    const TString& path)
{
    Y_ASSERT(!featureImportance.empty());
    TFileOutput out(path);
    const int docCount = featureImportance[0].ysize();
    const int featureCount = featureImportance.ysize();
    for (int docId = 0; docId < docCount; ++docId) {
        for (int featureId = 0; featureId < featureCount; ++featureId) {
            out << featureImportance[featureId][docId] << (featureId + 1 == featureCount ? '\n' : '\t');
        }
    }
}

inline void CalcAndOutputFstr(const TFullModel& model,
                              const NCB::TDataProviderPtr dataset, // can be nullptr
                              NPar::TLocalExecutor* localExecutor,
                              const TString* regularFstrPath,
                              const TString* internalFstrPath) {
    const NCB::TFeaturesLayout layout(
        model.ObliviousTrees.FloatFeatures,
        model.ObliviousTrees.CatFeatures);

    TVector<std::pair<double, TFeature>> internalEffect = CalcFeatureEffect(model, dataset, localExecutor);
    if (internalFstrPath != nullptr && !internalFstrPath->empty()) {
        OutputFstr(layout, internalEffect, *internalFstrPath);
    }

    if (regularFstrPath != nullptr && !regularFstrPath->empty()) {
        TVector<TFeatureEffect> regularEffect = CalcRegularFeatureEffect(
            internalEffect,
            model.GetNumCatFeatures(),
            model.GetNumFloatFeatures());
        OutputRegularFstr(layout, regularEffect, *regularFstrPath);
    }
}

inline void CalcAndOutputInteraction(
    const TFullModel& model,
    const TString* regularFstrPath,
    const TString* internalFstrPath)
{
    const NCB::TFeaturesLayout layout(
        model.ObliviousTrees.FloatFeatures,
        model.ObliviousTrees.CatFeatures);

    TVector<TInternalFeatureInteraction> internalInteraction = CalcInternalFeatureInteraction(model);
    if (internalFstrPath != nullptr) {
        OutputInteraction(layout, internalInteraction, *internalFstrPath);
    }

    if (regularFstrPath != nullptr) {
        TVector<TFeatureInteraction> interaction = CalcFeatureInteraction(internalInteraction, layout);
        OutputRegularInteraction(layout, interaction, *regularFstrPath);
    }
}
