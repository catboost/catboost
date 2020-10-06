#include "runtime_embedding_options.h"
#include "json_helper.h"

#include <util/string/builder.h>

namespace NCatboostOptions {
    TEmbeddingFeatureDescription::TEmbeddingFeatureDescription()
        : EmbeddingFeatureId("embedding_feature_id", -1)
        , FeatureEstimators("feature_estimators", TVector<TFeatureCalcerDescription>{})
    {
    }

    TEmbeddingFeatureDescription::TEmbeddingFeatureDescription(
        ui32 embeddingFeatureIdx,
        TConstArrayRef<TFeatureCalcerDescription> featureEstimators
    )
        : TEmbeddingFeatureDescription()
    {
        EmbeddingFeatureId.Set(embeddingFeatureIdx);
        FeatureEstimators.Set(
            TVector<TFeatureCalcerDescription>(featureEstimators.begin(), featureEstimators.end())
        );
    }

    void TEmbeddingFeatureDescription::Save(NJson::TJsonValue* optionsJson) const {
        SaveFields(optionsJson, EmbeddingFeatureId, FeatureEstimators);
    }

    void TEmbeddingFeatureDescription::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &EmbeddingFeatureId, &FeatureEstimators);
        CB_ENSURE(
            EmbeddingFeatureId.IsSet(),
            "EmbeddingFeatureDescription: embedding_feature_id is not specified "
                << EmbeddingFeatureId.Get()
        );
    }

    bool TEmbeddingFeatureDescription::operator==(const TEmbeddingFeatureDescription& rhs) const {
        return std::tie(EmbeddingFeatureId, FeatureEstimators)
            == std::tie(rhs.EmbeddingFeatureId, rhs.FeatureEstimators);
    }

    bool TEmbeddingFeatureDescription::operator!=(const TEmbeddingFeatureDescription& rhs) const {
        return !(*this == rhs);
    }

    TEmbeddingProcessingOptions::TEmbeddingProcessingOptions()
        : EmbeddingFeatureProcessing("feature_processing", {})
    {
        EmbeddingFeatureProcessing.SetDefault(
            TMap<TString, TVector<TFeatureCalcerDescription>>{
                {DefaultProcessingName(), {
                    {
                     TFeatureCalcerDescription{EFeatureCalcerType::LDA}
                    },
                }
            }
        });
    }

    const TVector<TFeatureCalcerDescription>& TEmbeddingProcessingOptions::GetCalcersDescriptions(
        ui32 embeddingFeatureIdx
    ) const {
        TString embeddingFeatureId = ToString(embeddingFeatureIdx);
        if (EmbeddingFeatureProcessing->contains(embeddingFeatureId)) {
            return EmbeddingFeatureProcessing->at(embeddingFeatureId);
        }
        return EmbeddingFeatureProcessing->at(DefaultProcessingName());
    }

    TRuntimeEmbeddingOptions::TRuntimeEmbeddingOptions(
        const TVector<ui32>& embeddingFeatureIndices,
        const TEmbeddingProcessingOptions& embeddingOptions
    )
        : TRuntimeEmbeddingOptions()
    {

        TVector<TEmbeddingFeatureDescription> embeddingFeatures;
        for (ui32 embeddingFeatureIdx: embeddingFeatureIndices) {
            embeddingFeatures.emplace_back(embeddingFeatureIdx, embeddingOptions.GetCalcersDescriptions(embeddingFeatureIdx));
        }
        EmbeddingFeatures.Set(embeddingFeatures);
    }
};
