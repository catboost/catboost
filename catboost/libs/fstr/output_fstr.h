#pragma once

#include "calc_fstr.h"

#include <catboost/libs/data/features_layout_helpers.h>
#include <catboost/private/libs/algo/tree_print.h>

#include <util/stream/file.h>
#include <util/system/yassert.h>

#include <utility>

inline TVector<std::pair<double, TString>> ExpandFeatureDescriptions(
    const NCB::TFeaturesLayout& layout,
    const TVector<std::pair<double, TFeature>>& effect
) {
    TVector<std::pair<double, TString>> result;
    result.reserve(effect.size());
    for (const auto& effectWithSplit : effect) {
        result.emplace_back(effectWithSplit.first, effectWithSplit.second.BuildDescription(layout));
    }
    return result;
}

inline TVector<std::pair<double, TString>> ExpandFeatureDescriptions(
    const NCB::TFeaturesLayout& layout,
    const TVector<TFeatureEffect>& regularEffect
) {
    TVector<std::pair<double, TString>> result;
    result.reserve(regularEffect.size());
    for (const auto& initialFeatureScore : regularEffect) {
        const auto& description = BuildFeatureDescription(
            layout,
            initialFeatureScore.Feature.Index,
            initialFeatureScore.Feature.Type);
        result.emplace_back(initialFeatureScore.Score, description);
    }
    return result;
}

inline void OutputStrengthDescriptions(
    const TVector<std::pair<double, TString>>& strengthDescriptions,
    const TString& path
) {
    TFileOutput out(path);
    for (const auto& strengthDescription : strengthDescriptions) {
        out << strengthDescription.first << "\t" << strengthDescription.second << Endl;
    }
}

inline void OutputFstr(
    const NCB::TFeaturesLayout& layout,
    const TVector<std::pair<double, TFeature>>& effect,
    const TString& path)
{
    OutputStrengthDescriptions(ExpandFeatureDescriptions(layout, effect), path);
}

inline void OutputRegularFstr(
    const NCB::TFeaturesLayout& layout,
    const TVector<TFeatureEffect>& regularEffect,
    const TString& path)
{
    OutputStrengthDescriptions(ExpandFeatureDescriptions(layout, regularEffect), path);
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
                              NPar::ILocalExecutor* localExecutor,
                              const TString* regularFstrPath,
                              const TString* internalFstrPath,
                              EFstrType type) {
    const NCB::TFeaturesLayout layout = MakeFeaturesLayout(model);

    TVector<std::pair<double, TFeature>> internalEffect = CalcFeatureEffect(model, dataset, type, localExecutor);
    if (internalFstrPath != nullptr && !internalFstrPath->empty()) {
        OutputFstr(layout, internalEffect, *internalFstrPath);
    }

    if (regularFstrPath != nullptr && !regularFstrPath->empty()) {
        TVector<TFeatureEffect> regularEffect = CalcRegularFeatureEffect(
            internalEffect,
            model);
        OutputRegularFstr(layout, regularEffect, *regularFstrPath);
    }
}

inline void CalcAndOutputInteraction(
    const TFullModel& model,
    const TString* regularFstrPath,
    const TString* internalFstrPath)
{
    const NCB::TFeaturesLayout layout = MakeFeaturesLayout(model);

    TVector<TInternalFeatureInteraction> internalInteraction = CalcInternalFeatureInteraction(model);
    if (internalFstrPath != nullptr) {
        OutputInteraction(layout, internalInteraction, *internalFstrPath);
    }

    if (regularFstrPath != nullptr) {
        TVector<TFeatureInteraction> interaction = CalcFeatureInteraction(internalInteraction, layout);
        OutputRegularInteraction(layout, interaction, *regularFstrPath);
    }
}
