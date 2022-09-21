#include "text_processing_options.h"
#include "json_helper.h"

#include <util/generic/cast.h>
#include <util/generic/hash_set.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/split.h>

using namespace NTextProcessing::NDictionary;
using namespace NTextProcessing::NTokenizer;

namespace NCatboostOptions {
    TTextColumnTokenizerOptions::TTextColumnTokenizerOptions()
        : TokenizerId("tokenizer_id", "default_tokenizer")
        , TokenizerOptions("tokenizer_options", TTokenizerOptions())
    {
    }

    TTextColumnTokenizerOptions::TTextColumnTokenizerOptions(TString tokenizerId, TTokenizerOptions tokenizerOptions)
        : TokenizerId("tokenizer_id", tokenizerId)
        , TokenizerOptions("tokenizer_options", tokenizerOptions)
    {
    }

    void TTextColumnTokenizerOptions::Save(NJson::TJsonValue* optionsJson) const {
        TJsonFieldHelper<TOption<TString>>::Write(TokenizerId, optionsJson);
        TokenizerOptionsToJson(TokenizerOptions, optionsJson);
    }

    void TTextColumnTokenizerOptions::Load(const NJson::TJsonValue& options) {
        {
            const bool wasRead = TJsonFieldHelper<TOption<TString>>::Read(options, &TokenizerId);
            CB_ENSURE(wasRead, "DictionaryOptions: no tokenizer_id was specified");
        }
        TokenizerOptions = JsonToTokenizerOptions(options);
    }

    bool TTextColumnTokenizerOptions::operator==(const TTextColumnTokenizerOptions& rhs) const {
        return std::tie(TokenizerOptions) == std::tie(rhs.TokenizerOptions);
    }

    bool TTextColumnTokenizerOptions::operator!=(const TTextColumnTokenizerOptions& rhs) const {
        return !(rhs == *this);
    }

    TTextColumnDictionaryOptions::TTextColumnDictionaryOptions()
        : DictionaryId("dictionary_id", "default_dictionary")
        , DictionaryOptions("dictionary_options", DEFAULT_DICTIONARY_OPTIONS)
        , DictionaryBuilderOptions("dictionary_builder_options", DEFAULT_DICTIONARY_BUILDER_OPTIONS)
    {
    }

    TTextColumnDictionaryOptions::TTextColumnDictionaryOptions(
        TString dictionaryId,
        TDictionaryOptions dictionaryOptions,
        TDictionaryBuilderOptions dictionaryBuilderOptions
    )
        : TTextColumnDictionaryOptions()
    {
        DictionaryId.SetDefault(dictionaryId);
        DictionaryOptions.SetDefault(dictionaryOptions);
        DictionaryBuilderOptions.SetDefault(dictionaryBuilderOptions);
    }

    void TTextColumnDictionaryOptions::Save(NJson::TJsonValue* optionsJson) const {
        TJsonFieldHelper<TOption<TString>>::Write(DictionaryId, optionsJson);
        DictionaryOptionsToJson(DictionaryOptions, optionsJson);
        DictionaryBuilderOptionsToJson(DictionaryBuilderOptions, optionsJson);
    }

    void TTextColumnDictionaryOptions::Load(const NJson::TJsonValue& options) {
        {
            const bool wasRead = TJsonFieldHelper<TOption<TString>>::Read(options, &DictionaryId);
            CB_ENSURE(wasRead, "DictionaryOptions: no dictionary_id was specified");
        }
        JsonToDictionaryOptions(options, &DictionaryOptions.Get());
        JsonToDictionaryBuilderOptions(options, &DictionaryBuilderOptions.Get());
    }

    bool TTextColumnDictionaryOptions::operator==(const TTextColumnDictionaryOptions& rhs) const {
        return std::tie(DictionaryOptions, DictionaryBuilderOptions) ==
               std::tie(rhs.DictionaryOptions, rhs.DictionaryBuilderOptions);
    }

    bool TTextColumnDictionaryOptions::operator!=(const TTextColumnDictionaryOptions& rhs) const {
        return !(rhs == *this);
    }

    TFeatureCalcerDescription::TFeatureCalcerDescription()
        : CalcerType("calcer_type", EFeatureCalcerType::NaiveBayes)
        , CalcerOptions("calcer_options", NJson::TJsonValue())
    {}

    TFeatureCalcerDescription::TFeatureCalcerDescription(
        EFeatureCalcerType featureCalcerType,
        NJson::TJsonValue calcerOptions
    )
        : TFeatureCalcerDescription()
    {
        CalcerType.Set(featureCalcerType);
        if (calcerOptions.IsDefined()) {
            CalcerOptions.Set(calcerOptions);
        }
    }

    void TFeatureCalcerDescription::Save(NJson::TJsonValue* optionsJson) const {
        TStringBuilder calcerDescription;
        calcerDescription << ToString(CalcerType.Get()) << ':';

        for (auto& [key, value]: CalcerOptions->GetMap()) {
            calcerDescription << key << '=' << value << ',';
        }
        calcerDescription.pop_back();

        optionsJson->SetValue(ToString(calcerDescription));
    }

    void TFeatureCalcerDescription::Load(const NJson::TJsonValue& options) {
        if (!options.IsDefined()) {
            return;
        }
        TString calcerName;

        if (options.IsString()) {
            TStringBuf name, calcersOptions;
            TStringBuf(options.GetString()).Split(':', name, calcersOptions);
            calcerName = name;
            CalcerOptions->InsertValue("calcer_type", calcerName);
            for (TStringBuf stringParam : StringSplitter(calcersOptions).Split(':')) {
                TStringBuf key, value;
                stringParam.Split('=', key, value);
                CalcerOptions->InsertValue(key, value);
            }
        } else {
            CB_ENSURE(options.IsMap(),
                      "We only support string and dictionaries as featurization options for value "
                      << options.GetStringRobust() << " with type " << options.GetType());
            calcerName = options["calcer_type"].GetString();
            CalcerOptions.Set(options);
        }

        EFeatureCalcerType calcerType;

        CB_ENSURE(TryFromString<EFeatureCalcerType>(calcerName, calcerType),
                  "Unknown feature estimator type " << calcerName);

        CalcerType.Set(calcerType);
    }

    bool TFeatureCalcerDescription::operator==(const TFeatureCalcerDescription& rhs) const {
        return CalcerType == rhs.CalcerType;
    }

    bool TFeatureCalcerDescription::operator!=(const TFeatureCalcerDescription& rhs) const {
        return !(*this == rhs);
    }

    TFeatureCalcerDescription& TFeatureCalcerDescription::operator=(EFeatureCalcerType featureCalcerType) {
        CalcerType.Set(featureCalcerType);
        return *this;
    }

    TTextProcessingOptions::TTextProcessingOptions()
        : Tokenizers("tokenizers", {})
        , Dictionaries("dictionaries", {})
        , TextFeatureProcessing("feature_processing", {})
    {
        SetDefault();
    }

    void TTextProcessingOptions::SetDefault(bool forClassification) {
        const TString tokenizerName = "Space";
        Tokenizers.SetDefault(TVector<TTextColumnTokenizerOptions>{{tokenizerName, TTokenizerOptions()}});

        const TString unigramDctionaryName = "Word";
        const TDictionaryOptions unigramDctionaryOptions = {ETokenLevelType::Word};
        const TTextColumnDictionaryOptions unigramDctionary(unigramDctionaryName, unigramDctionaryOptions);

        const TString bigramDctionaryName = "BiGram";
        const TDictionaryOptions bigramDctionaryOptions = {ETokenLevelType::Word, /*GramOrder=*/2};
        const TTextColumnDictionaryOptions bigramDctionary(bigramDctionaryName, bigramDctionaryOptions);

        Dictionaries.SetDefault(TVector<TTextColumnDictionaryOptions>{bigramDctionary, unigramDctionary});

        TVector<TTextFeatureProcessing> textFeatureProcessingVector;
        textFeatureProcessingVector.push_back(
            TTextFeatureProcessing{
                {TFeatureCalcerDescription{EFeatureCalcerType::BoW}},
                {tokenizerName},
                {bigramDctionaryName, unigramDctionaryName}
            }
        );
        if (forClassification) {
            textFeatureProcessingVector.push_back(
                TTextFeatureProcessing{
                    {TFeatureCalcerDescription{EFeatureCalcerType::NaiveBayes}},
                    {tokenizerName},
                    {unigramDctionaryName}
                }
            );
        }

        TextFeatureProcessing.SetDefault(
            TMap<TString, TVector<TTextFeatureProcessing>>{{DefaultProcessingName(), std::move(textFeatureProcessingVector)}}
        );
    }

    void TTextProcessingOptions::Validate(bool forClassification) const {
        if (!forClassification) {
            if (TextFeatureProcessing.IsSet()) {
                for (const auto& [featureId, processings]: TextFeatureProcessing.Get()) {
                    for (const auto& processing: processings) {
                        if (!processing.FeatureCalcers->empty()) {
                            for (const auto& featureCalcer : processing.FeatureCalcers.Get()) {
                                CB_ENSURE(
                                    !IsClassificationOnlyEstimator(featureCalcer.CalcerType.Get()),
                                    "Text feature processing feature calcer has type " << featureCalcer.CalcerType.Get()
                                    << " that is supported only for classification"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    void TTextProcessingOptions::SetNotSpecifiedOptionsToDefaults() {
        if (Tokenizers->empty()) {
            Tokenizers.SetDefault(TVector<TTextColumnTokenizerOptions>{{"Space", TTokenizerOptions()}});
        }

        TVector<TString> tokenizerNames;
        for (const auto& tokenizer : Tokenizers.Get()) {
            tokenizerNames.push_back(tokenizer.TokenizerId.Get());
        }

        if (Dictionaries->empty()) {
            Dictionaries.SetDefault(TVector<TTextColumnDictionaryOptions>{{"Word", {ETokenLevelType::Word}}});
        }

        TVector<TString> dictionaryNames;
        for (const auto& dictionary : Dictionaries.Get()) {
            dictionaryNames.push_back(dictionary.DictionaryId.Get());
        }

        if (TextFeatureProcessing->empty()) {
            TMap<TString, TVector<TTextFeatureProcessing>> textFeatureProcessing{{
                DefaultProcessingName(), {TTextFeatureProcessing()}
            }};
            TextFeatureProcessing.SetDefault(textFeatureProcessing);
        }

        TVector<TFeatureCalcerDescription> defaultCalcers{TFeatureCalcerDescription{EFeatureCalcerType::BoW}};

        for (auto& [featureId, processings]: TextFeatureProcessing.Get()) {
            for (auto& processing: processings) {
                if (processing.TokenizersNames->empty()) {
                    processing.TokenizersNames.SetDefault(tokenizerNames);
                }
                if (processing.DictionariesNames->empty()) {
                    processing.DictionariesNames.SetDefault(dictionaryNames);
                }
                if (processing.FeatureCalcers->empty()) {
                    processing.FeatureCalcers.SetDefault(defaultCalcers);
                }
            }
        }
    }

    void TTextProcessingOptions::Save(NJson::TJsonValue* optionsJson) const {
        SaveFields(optionsJson, Tokenizers, Dictionaries, TextFeatureProcessing);
    }

    void TTextProcessingOptions::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &Tokenizers, &Dictionaries, &TextFeatureProcessing);
        SetNotSpecifiedOptionsToDefaults();
    }

    bool TTextProcessingOptions::operator==(
        const TTextProcessingOptions& rhs
    ) const {
        return std::tie(Tokenizers, Dictionaries, TextFeatureProcessing)
            == std::tie(rhs.Tokenizers, rhs.Dictionaries, rhs.TextFeatureProcessing);
    }

    bool TTextProcessingOptions::operator!=(
        const TTextProcessingOptions& rhs) const {
        return !(*this == rhs);
    }

    const TVector<TTextFeatureProcessing>& TTextProcessingOptions::GetFeatureProcessing(
        ui32 textFeatureIdx
    ) const {
        TString textFeatureId = ToString(textFeatureIdx);
        if (TextFeatureProcessing->contains(textFeatureId)) {
            return TextFeatureProcessing->at(textFeatureId);
        }
        return TextFeatureProcessing->at(DefaultProcessingName());
    }

    const TVector<TTextColumnTokenizerOptions>& TTextProcessingOptions::GetTokenizers() const {
        return Tokenizers;
    }

    const TVector<TTextColumnDictionaryOptions>& TTextProcessingOptions::GetDictionaries() const {
        return Dictionaries;
    }

    void TTextProcessingOptions::SetDefaultMinTokenOccurrence(ui64 minTokenOccurrence) {
        for (auto& dictionaryOptions: Dictionaries.Get()) {
            auto& dictionaryBuilderOptions = dictionaryOptions.DictionaryBuilderOptions;
            if (dictionaryBuilderOptions.IsDefault()) {
                dictionaryBuilderOptions->OccurrenceLowerBound = minTokenOccurrence;
            }
        }
    }

    TTextProcessingOptions::TTextProcessingOptions(
        TVector<TTextColumnTokenizerOptions>&& tokenizers,
        TVector<TTextColumnDictionaryOptions>&& dictionaries,
        TMap<TString, TVector<TTextFeatureProcessing>>&& textFeatureProcessing
    )
        : Tokenizers("tokenizers", {})
        , Dictionaries("dictionaries", {})
        , TextFeatureProcessing("feature_processing", {})
    {
        Tokenizers.Set(tokenizers);
        Dictionaries.Set(dictionaries);
        TextFeatureProcessing.Set(textFeatureProcessing);
    }

    TTextFeatureProcessing::TTextFeatureProcessing()
        : FeatureCalcers("feature_calcers", {})
        , TokenizersNames("tokenizers_names", {})
        , DictionariesNames("dictionaries_names", {})
    {}

    void TTextFeatureProcessing::Save(NJson::TJsonValue* optionsJson) const {
        SaveFields(optionsJson, TokenizersNames, DictionariesNames, FeatureCalcers);
    }

    void TTextFeatureProcessing::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &TokenizersNames, &DictionariesNames, &FeatureCalcers);
    }

    bool TTextFeatureProcessing::operator==(const TTextFeatureProcessing& rhs) const {
        return std::tie(TokenizersNames, DictionariesNames, FeatureCalcers)
            == std::tie(rhs.TokenizersNames, rhs.DictionariesNames, rhs.FeatureCalcers);
    }

    bool TTextFeatureProcessing::operator!=(const TTextFeatureProcessing& rhs) const {
        return !(*this == rhs);
    }

    TTextFeatureProcessing::TTextFeatureProcessing(
        TVector<TFeatureCalcerDescription>&& featureCalcers,
        TVector<TString>&& tokenizersNames,
        TVector<TString>&& dictionariesNames
    )
        : TTextFeatureProcessing()
    {
        FeatureCalcers.Set(featureCalcers);
        TokenizersNames.Set(tokenizersNames);
        DictionariesNames.Set(dictionariesNames);
    }
}

void NCatboostOptions::ParseTextProcessingOptionsFromPlainJson(
    const NJson::TJsonValue& plainOptions,
    NJson::TJsonValue* textProcessingOptions,
    TSet<TString>* seenKeys
) {
    const TString textProcessingOptionName = "text_processing";
    const TString tokenizersOptionName = "tokenizers";
    const TString dictionariesOptionName = "dictionaries";
    const TString featureCalcersOptionName = "feature_calcers";

    const bool hasSimpleTextProcessingOptions = (
        plainOptions.Has(tokenizersOptionName) ||
        plainOptions.Has(dictionariesOptionName) ||
        plainOptions.Has(featureCalcersOptionName)
    );

    if (!hasSimpleTextProcessingOptions && !plainOptions.Has(textProcessingOptionName)) {
        return;
    }

    CB_ENSURE(
        !hasSimpleTextProcessingOptions || !plainOptions.Has(textProcessingOptionName),
        "You should provide either `" << textProcessingOptionName << "` option or `" << tokenizersOptionName
        << "`, `" << dictionariesOptionName << "`, `" << featureCalcersOptionName << "` options."
    );

    if (plainOptions.Has(textProcessingOptionName)) {
        *textProcessingOptions = plainOptions[textProcessingOptionName];
        seenKeys->insert(textProcessingOptionName);
        return;
    }

    if (plainOptions.Has(tokenizersOptionName)) {
        (*textProcessingOptions)["tokenizers"] = plainOptions[tokenizersOptionName];
        seenKeys->insert(tokenizersOptionName);
    }

    if (plainOptions.Has(dictionariesOptionName)) {
        (*textProcessingOptions)["dictionaries"] = plainOptions[dictionariesOptionName];
        seenKeys->insert(dictionariesOptionName);
    }

    if (plainOptions.Has(featureCalcersOptionName)) {
        auto& processingDescription = (*textProcessingOptions)["feature_processing"][TTextProcessingOptions::DefaultProcessingName()];
        processingDescription["feature_calcers"] = plainOptions[featureCalcersOptionName];
        seenKeys->insert(featureCalcersOptionName);
    }
}

void NCatboostOptions::SaveTextProcessingOptionsToPlainJson(
    const NJson::TJsonValue& textProcessingOptions,
    NJson::TJsonValue* plainOptions
) {
    (*plainOptions)["text_processing"] = textProcessingOptions;
}
