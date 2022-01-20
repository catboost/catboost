#include "embedding_processing_options.h"

#include "json_helper.h"


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
        : EmbeddingFeatureProcessing("embedding_processing", {})
    {
        EmbeddingFeatureProcessing.SetDefault(
            TMap<TString, TVector<TFeatureCalcerDescription>>{
                {DefaultProcessingName(), DefaultEmbeddingCalcers()}
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

    void TEmbeddingProcessingOptions::Save(NJson::TJsonValue* optionsJson) const {
        SaveFields(optionsJson, EmbeddingFeatureProcessing);
    }

    void TEmbeddingProcessingOptions::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &EmbeddingFeatureProcessing);
        SetNotSpecifiedOptionsToDefaults();

    }

    bool TEmbeddingProcessingOptions::operator==(
        const TEmbeddingProcessingOptions& rhs
    ) const {
        return std::tie(EmbeddingFeatureProcessing)
            == std::tie(rhs.EmbeddingFeatureProcessing);
    }

    bool TEmbeddingProcessingOptions::operator!=(
        const TEmbeddingProcessingOptions& rhs) const {
        return !(*this == rhs);
    }

    void TEmbeddingProcessingOptions::SetNotSpecifiedOptionsToDefaults() {
        if (EmbeddingFeatureProcessing->empty()) {
            TMap<TString, TVector<TFeatureCalcerDescription>> embeddingFeatureProcessing{{
                DefaultProcessingName(), {DefaultEmbeddingCalcers()}
            }};
            EmbeddingFeatureProcessing.SetDefault(embeddingFeatureProcessing);
        }
        for (auto& [featureId, calcers]: EmbeddingFeatureProcessing.Get()) {
            if (calcers.empty()) {
                calcers = DefaultEmbeddingCalcers();
            }
        }
    }
}

void NCatboostOptions::ParseEmbeddingProcessingOptionsFromPlainJson(
    const NJson::TJsonValue& plainOptions,
    NJson::TJsonValue* embeddingProcessingOptions,
    TSet<TString>* seenKeys
) {
    const TString embeddingProcessingOptionName = "embedding_processing";
    const TString featureCalcersOptionName = "embedding_calcers";

    if (!plainOptions.Has(featureCalcersOptionName) && !plainOptions.Has(embeddingProcessingOptionName)) {
        return;
    }

    CB_ENSURE(
        !plainOptions.Has(featureCalcersOptionName) || !plainOptions.Has(embeddingProcessingOptionName),
        "You should provide either `" << embeddingProcessingOptionName << "` option or `"
        << featureCalcersOptionName << "` options."
    );

    if (plainOptions.Has(embeddingProcessingOptionName)) {
        *embeddingProcessingOptions = plainOptions[embeddingProcessingOptionName];
        seenKeys->insert(embeddingProcessingOptionName);
        return;
    }

    if (plainOptions.Has(featureCalcersOptionName)) {
        auto& processingDescription = (*embeddingProcessingOptions)["embedding_processing"][TEmbeddingProcessingOptions::DefaultProcessingName()];
        processingDescription = plainOptions[featureCalcersOptionName];
        seenKeys->insert(featureCalcersOptionName);
    }
}

void NCatboostOptions::SaveEmbeddingProcessingOptionsToPlainJson(
    const NJson::TJsonValue& embeddingProcessingOptions,
    NJson::TJsonValue* plainOptions
) {
    (*plainOptions)["embedding_processing"] = embeddingProcessingOptions;
}
