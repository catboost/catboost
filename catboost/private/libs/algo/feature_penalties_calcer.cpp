#include "feature_penalties_calcer.h"

namespace NCB {
    static inline float GetFeaturePenalty(
        const NCatboostOptions::TPerFeaturePenalty& featurePenalties,
        const TFeaturesLayout& layout,
        const ui32 internalFeatureIndex,
        const EFeatureType type
    ) {
        const auto externalFeatureIndex = layout.GetExternalFeatureIdx(internalFeatureIndex, type);
        auto it = featurePenalties.find(externalFeatureIndex);
        return (it != featurePenalties.end() ? it->second : NCatboostOptions::DEFAULT_FEATURE_WEIGHT);
    }

    static float GetSplitFeatureWeight(
        const TSplit& split,
        const TFeaturesLayout& layout,
        const NCatboostOptions::TPerFeaturePenalty& featureWeights
    ) {
        float result = 1;

        const auto addPenaltyFunc = [&](const int internalFeatureIdx, const EFeatureType type) {
            result *= GetFeaturePenalty(featureWeights, layout, internalFeatureIdx, type);
        };
        split.IterateOverUsedFeatures(addPenaltyFunc);

        return result;
    }

    static inline float GetFeatureFirstUsePenalty(
        const NCatboostOptions::TPerFeaturePenalty& featurePenalties,
        const TFeaturesLayout& layout,
        const TVector<bool>& usedFeatures,
        const ui32 internalFeatureIndex,
        const EFeatureType type
    ) {
        const auto externalFeatureIndex = layout.GetExternalFeatureIdx(internalFeatureIndex, type);
        float result = NCatboostOptions::DEFAULT_FEATURE_PENALTY;
        if (!usedFeatures[externalFeatureIndex]) {
            auto it = featurePenalties.find(externalFeatureIndex);
            if (it != featurePenalties.end()) {
                result = it->second;
            }
        }
        return result;
    }

    static float GetSplitFirstFeatureUsePenalty(
        const TSplit& split,
        const TFeaturesLayout& layout,
        const TVector<bool>& usedFeatures,
        const NCatboostOptions::TPerFeaturePenalty& featurePenalties,
        const float penaltiesCoefficient
    ) {
        float result = 0;

        const auto addPenaltyFunc = [&](const int internalFeatureIdx, const EFeatureType type) {
            result += GetFeatureFirstUsePenalty(featurePenalties, layout, usedFeatures, internalFeatureIdx, type);
        };
        split.IterateOverUsedFeatures(addPenaltyFunc);

        result *= penaltiesCoefficient;
        return result;
    }

    void AddFeaturePenaltiesToBestSplits(
        TLearnContext* ctx,
        const NCB::TQuantizedForCPUObjectsDataProvider& objectsData,
        ui32 oneHotMaxSize,
        TVector<TCandidateInfo>* candidates
    ) {
        const NCatboostOptions::TPerFeaturePenalty& featureWeights = ctx->Params.ObliviousTreeOptions->FeaturePenalties->FeatureWeights;
        const float penaltiesCoefficient = ctx->Params.ObliviousTreeOptions->FeaturePenalties->PenaltiesCoefficient;
        const NCatboostOptions::TPerFeaturePenalty& firstFeatureUsePenalty = ctx->Params.ObliviousTreeOptions->FeaturePenalties->FirstFeatureUsePenalty;

        const TFeaturesLayout& layout = *ctx->Layout;
        const TVector<bool>& usedFeatures = ctx->LearnProgress->UsedFeatures;

        for (auto& cand : *candidates) {
            double& score = cand.BestScore.Val;
            const auto bestSplit = cand.GetBestSplit(objectsData, oneHotMaxSize);

            score *= GetSplitFeatureWeight(
                bestSplit,
                layout,
                featureWeights
            );
            score -= GetSplitFirstFeatureUsePenalty(
                bestSplit,
                layout,
                usedFeatures,
                firstFeatureUsePenalty,
                penaltiesCoefficient
            );
        }
    }
}