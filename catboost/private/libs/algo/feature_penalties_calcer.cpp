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

    float GetSplitFeatureWeight(
        const TSplit& split,
        const TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
        const TFeaturesLayout& layout,
        const NCatboostOptions::TPerFeaturePenalty& featureWeights
    ) {
        float result = 1;

        const auto addPenaltyFunc = [&](const int internalFeatureIdx, const EFeatureType type) {
            result *= GetFeaturePenalty(featureWeights, layout, internalFeatureIdx, type);
        };
        split.IterateOverUsedFeatures(estimatedFeaturesContext, addPenaltyFunc);

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
        const TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
        const TFeaturesLayout& layout,
        const TVector<bool>& usedFeatures,
        const NCatboostOptions::TPerFeaturePenalty& featurePenalties,
        const float penaltiesCoefficient
    ) {
        float result = 0;

        const auto addPenaltyFunc = [&](const int internalFeatureIdx, const EFeatureType type) {
            result += GetFeatureFirstUsePenalty(featurePenalties, layout, usedFeatures, internalFeatureIdx, type);
        };
        split.IterateOverUsedFeatures(estimatedFeaturesContext, addPenaltyFunc);

        result *= penaltiesCoefficient;
        return result;
    }

    class TPerObjectFeaturePenaltiesCalcer {
    public:
        explicit TPerObjectFeaturePenaltiesCalcer(
            const NCatboostOptions::TPerFeaturePenalty& penalties,
            const EGrowPolicy growPolicy,
            const TVector<bool>& usedFeatures,
            const TMap<ui32, TVector<bool>>& usedFeaturesPerObject,
            const TCalcScoreFold& calcScoreFold,
            const TVector<TIndexType>& leaves
        )
            : Penalties(penalties)
            , GrowPolicy(growPolicy)
            , UsedFeatures(usedFeatures)
            , UsedFeaturesPerObject(usedFeaturesPerObject)
            , CalcScoreFold(calcScoreFold)
            , Leaves(leaves)
        {
        }

        float GetPenalty(const ui32 externalFeatureIdx) {
            const auto it = PenaltySum.find(externalFeatureIdx);
            if (it != PenaltySum.end()) { //already calculated
                return it->second;
            }

            return PenaltySum[externalFeatureIdx] = CalculatePenalty(externalFeatureIdx);
        }

    private:
        float CalculatePenalty(const ui32 externalFeatureIdx) const {
            const auto penaltyIt = Penalties.find(externalFeatureIdx);
            if (penaltyIt == Penalties.end()) { //not penalized
                return NCatboostOptions::DEFAULT_FEATURE_PENALTY;
            }
            const float penaltyPerObject = penaltyIt->second;

            ui64 objectsCount = 0;
            if (GrowPolicy == EGrowPolicy::SymmetricTree) {
                if (!UsedFeatures[externalFeatureIdx]) {
                    objectsCount = CalcScoreFold.GetDocCount();
                }
            } else {
                auto it = UsedFeaturesPerObject.find(externalFeatureIdx);
                CB_ENSURE_INTERNAL(
                    it != UsedFeaturesPerObject.end(),
                    "No feature usage stat for penalized feature number " << externalFeatureIdx
                );
                const auto& featureUsage = it->second;
                for (const auto leafNumber : Leaves) {
                    const auto& leafBounds = CalcScoreFold.LeavesBounds[leafNumber];
                    for (const auto idxInCalcScoreFold : xrange(leafBounds.Begin, leafBounds.End)) {
                        const auto globalObjectIdx = CalcScoreFold.IndexInFold[idxInCalcScoreFold];
                        if (!featureUsage[globalObjectIdx]) {
                            ++objectsCount;
                        }
                    }
                }
            }

            const float penaltySum = penaltyPerObject * objectsCount;
            return penaltySum;
        }

        const NCatboostOptions::TPerFeaturePenalty& Penalties;
        const EGrowPolicy GrowPolicy;
        const TVector<bool>& UsedFeatures;
        const TMap<ui32, TVector<bool>>& UsedFeaturesPerObject;
        const TCalcScoreFold& CalcScoreFold;
        const TVector<TIndexType>& Leaves;
        TMap<ui32, float> PenaltySum;
    };

    static float GetSplitPerObjectPenalty(
        const TSplit& split,
        const TCombinedEstimatedFeaturesContext& estimatedFeaturesContext,
        const TFeaturesLayout& layout,
        const float penaltiesCoefficient,
        TPerObjectFeaturePenaltiesCalcer* perObjectFeaturePenaltiesCalcer
    ) {
        float result = 0;

        const auto addPenaltyFunc = [&](const int internalFeatureIdx, const EFeatureType type) {
            const auto externalFeatureIndex = layout.GetExternalFeatureIdx(internalFeatureIdx, type);
            result += perObjectFeaturePenaltiesCalcer->GetPenalty(externalFeatureIndex);
        };
        split.IterateOverUsedFeatures(estimatedFeaturesContext, addPenaltyFunc);

        result *= penaltiesCoefficient;
        return result;
    }

    void PenalizeBestSplits(
        const TVector<TIndexType>& leaves,
        const TLearnContext& ctx,
        const TTrainingDataProviders& trainingData,
        const TFold& fold,
        ui32 oneHotMaxSize,
        TVector<TCandidateInfo>* candidates
    ) {
        const auto& featurePenaltiesOptions = ctx.Params.ObliviousTreeOptions->FeaturePenalties.Get();
        const float penaltiesCoefficient = featurePenaltiesOptions.PenaltiesCoefficient;
        const NCatboostOptions::TPerFeaturePenalty& firstFeatureUsePenalty = featurePenaltiesOptions.FirstFeatureUsePenalty;
        const NCatboostOptions::TPerFeaturePenalty& perObjectPenalty = featurePenaltiesOptions.PerObjectFeaturePenalty;

        const TFeaturesLayout& layout = *ctx.Layout;
        const TVector<bool>& usedFeatures = ctx.LearnProgress->UsedFeatures;
        const auto& usedFeaturesPerObject = ctx.LearnProgress->UsedFeaturesPerObject;
        TPerObjectFeaturePenaltiesCalcer perObjectFeaturePenaltiesCalcer(
            perObjectPenalty,
            ctx.Params.ObliviousTreeOptions->GrowPolicy,
            usedFeatures,
            usedFeaturesPerObject,
            ctx.SampledDocs,
            leaves
        ); // caches results for already calculated features

        for (auto& cand : *candidates) {
            if (cand.BestScore.Val == MINIMAL_SCORE) {
                continue;
            }
            const auto bestSplit = cand.GetBestSplit(trainingData, fold, oneHotMaxSize);

            double& score = cand.BestScore.Val;
            score -= GetSplitFirstFeatureUsePenalty(
                bestSplit,
                ctx.LearnProgress->EstimatedFeaturesContext,
                layout,
                usedFeatures,
                firstFeatureUsePenalty,
                penaltiesCoefficient
            );
            score -= GetSplitPerObjectPenalty(
                bestSplit,
                ctx.LearnProgress->EstimatedFeaturesContext,
                layout,
                penaltiesCoefficient,
                &perObjectFeaturePenaltiesCalcer
            );
        }
    }
}
