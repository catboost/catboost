#include "text_processing_options.h"
#include "json_helper.h"

#include <util/generic/cast.h>
#include <util/generic/hash_set.h>
#include <util/string/builder.h>
#include <util/string/cast.h>
#include <util/string/split.h>

#include <tuple>

using namespace NTextProcessing::NDictionary;

static TDictionaryBuilderOptions GetDictionaryBuilderOptions(const NJson::TJsonValue& options) {
    TDictionaryBuilderOptions dictionaryBuilderOptions;
    const TStringBuf minTokenOccurencyName = "min_token_occurrence";
    const TStringBuf maxDictSizeName = "max_dict_size";
    if (options.Has(minTokenOccurencyName)) {
        dictionaryBuilderOptions.OccurrenceLowerBound = FromString<ui64>(options[minTokenOccurencyName].GetString());
    }
    if (options.Has(maxDictSizeName)) {
        dictionaryBuilderOptions.MaxDictionarySize = FromString<i32>(options[maxDictSizeName].GetString());
    }
    return dictionaryBuilderOptions;
}

template <class T>
static void OptionToJson(TStringBuf name, const T& value, const T& defaultValue, NJson::TJsonValue* json) {
    if (value != defaultValue) {
        (*json)[name] = ToString(value);
    }
}

static void DictionaryBuilderOptionsToJson(
    const TDictionaryBuilderOptions& options,
    NJson::TJsonValue* optionsJson
) {
    const TDictionaryBuilderOptions defaultBuilderOptions;
    (*optionsJson)["min_token_occurrence"] = ToString(options.OccurrenceLowerBound);
    OptionToJson(
        "max_dict_size",
        options.MaxDictionarySize,
        defaultBuilderOptions.MaxDictionarySize,
        optionsJson
    );
}

static void DictionaryOptionsToJson(
    const NTextProcessing::NDictionary::TDictionaryOptions& options,
    NJson::TJsonValue* optionsJson
) {
    const NTextProcessing::NDictionary::TDictionaryOptions defaultDictionaryOptions;
    (*optionsJson)["token_level_type"] = ToString(options.TokenLevelType);

    OptionToJson("gram_order", options.GramOrder, defaultDictionaryOptions.GramOrder, optionsJson);
    OptionToJson("skip_step", options.SkipStep, defaultDictionaryOptions.SkipStep, optionsJson);
    OptionToJson("start_token_id", options.StartTokenId, defaultDictionaryOptions.StartTokenId, optionsJson);
    OptionToJson(
        "end_of_word_token_policy",
        options.EndOfWordTokenPolicy,
        defaultDictionaryOptions.EndOfWordTokenPolicy,
        optionsJson
    );
    OptionToJson(
        "end_of_sentence_token_policy",
        options.EndOfSentenceTokenPolicy,
        defaultDictionaryOptions.EndOfSentenceTokenPolicy,
        optionsJson
    );
}

namespace NCatboostOptions {
    TTextColumnDictionaryOptions::TTextColumnDictionaryOptions()
        : DictionaryId("dictionary_id", "default_dictionary")
        , DictionaryOptions("dictionary_options",
            NTextProcessing::NDictionary::TDictionaryOptions{
                /*TokenLevelType*/ETokenLevelType::Word,
                /*GramOrder*/ 1,
                /*SkipStep (skip-grams)*/0,
                /*StartTokenId*/ 0,
                /*EndOfWordTokenPolicy*/EEndOfWordTokenPolicy::Insert,
                /*EndOfSentenceTokenPolicy*/ EEndOfSentenceTokenPolicy::Skip
        })
        , DictionaryBuilderOptions(
            "dictionary_builder_options",
        TDictionaryBuilderOptions{
           /*OccurrenceLowerBound*/ 3,
           /*MaxDictionarySize*/ DefaultMaxDictionarySize()
        })
    {}

    TTextColumnDictionaryOptions::TTextColumnDictionaryOptions(
        TString dictionaryId,
        TDictionaryOptions dictionaryOptions,
        TMaybe<TDictionaryBuilderOptions> dictionaryBuilderOptions
    ) : TTextColumnDictionaryOptions() {
        DictionaryId.SetDefault(dictionaryId);
        DictionaryOptions.SetDefault(dictionaryOptions);
        if (dictionaryBuilderOptions.Defined()) {
            DictionaryBuilderOptions.SetDefault(dictionaryBuilderOptions.GetRef());
        }
    }

    void TTextColumnDictionaryOptions::Save(NJson::TJsonValue* optionsJson) const {
        TJsonFieldHelper<TOption<TString>>::Write(DictionaryId, optionsJson);
        DictionaryOptionsToJson(DictionaryOptions, optionsJson);
        DictionaryBuilderOptionsToJson(DictionaryBuilderOptions, optionsJson);
    }

    void TTextColumnDictionaryOptions::Load(const NJson::TJsonValue& options) {
        {
            const bool wasRead = TJsonFieldHelper<TOption<TString>>::Read(options, &DictionaryId);
            CB_ENSURE(wasRead, "DictionaryOptions: no dictionaryId was specified");
        }
        DictionaryOptions = JsonToDictionaryOptions(options);
        DictionaryBuilderOptions = GetDictionaryBuilderOptions(options);
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

        const TString& calcerDescription = options.GetString();
        TStringBuf calcerDescriptionStrBuf = calcerDescription;

        TStringBuf calcerTypeString;
        TStringBuf calcerOptionsString;
        calcerDescriptionStrBuf.Split(':', calcerTypeString, calcerOptionsString);

        if (EFeatureCalcerType calcerType; TryFromString<EFeatureCalcerType>(calcerTypeString, calcerType)) {
            CalcerType.Set(calcerType);
        } else {
            CB_ENSURE(false, "Unknown feature estimator type " << calcerTypeString);
        }

        CalcerOptions->SetType(NJson::EJsonValueType::JSON_MAP);
        if (!calcerOptionsString.empty()) {
            for (TStringBuf optionString: StringSplitter(calcerOptionsString).Split(',')) {
                TStringBuf name;
                TStringBuf value;
                optionString.Split('=', name, value);
                CalcerOptions->InsertValue(name, value);
            }
        }
    }

    bool TFeatureCalcerDescription::operator==(const TFeatureCalcerDescription& rhs) const {
        return CalcerType == rhs.CalcerType;
    }

    bool TFeatureCalcerDescription::operator!=(const TFeatureCalcerDescription& rhs) const {
        return !(*this == rhs);
    }

    TFeatureCalcerDescription& TFeatureCalcerDescription::operator=(
        EFeatureCalcerType featureCalcerType
    ) {
        CalcerType.Set(featureCalcerType);
        return *this;
    }

    TTextProcessingOptions::TTextProcessingOptions()
        : Dictionaries(
            "dictionaries",
            GetDefaultDictionaries()
        )
        , TextFeatureProcessing(
            "text_processing",
            TMap<TString, TVector<TTextFeatureProcessing>>{{
                DefaultProcessingName(),
                {
                    TTextFeatureProcessing{
                        TFeatureCalcerDescription{EFeatureCalcerType::BoW},
                        GetDefaultCalcerDictionaries(EFeatureCalcerType::BoW)
                    },
                    TTextFeatureProcessing{
                        TFeatureCalcerDescription{EFeatureCalcerType::NaiveBayes},
                        GetDefaultCalcerDictionaries(EFeatureCalcerType::NaiveBayes)
                    }
                }
            }}
        ) {}

    void TTextProcessingOptions::Save(NJson::TJsonValue* optionsJson) const {
        SaveFields(optionsJson, Dictionaries, TextFeatureProcessing);
    }

    void TTextProcessingOptions::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &Dictionaries, &TextFeatureProcessing);
    }

    bool TTextProcessingOptions::operator==(
        const TTextProcessingOptions& rhs
    ) const {
        return std::tie(Dictionaries, TextFeatureProcessing)
               == std::tie(rhs.Dictionaries, rhs.TextFeatureProcessing);
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

    TTextColumnDictionaryOptions TTextProcessingOptions::BiGramDictionaryOptions() {
        return {
            "BiGram",
            NTextProcessing::NDictionary::TDictionaryOptions{
                NTextProcessing::NDictionary::ETokenLevelType::Word,
                2
            }
        };
    }

    TTextColumnDictionaryOptions TTextProcessingOptions::WordDictionaryOptions() {
        return {
            "Word",
            NTextProcessing::NDictionary::TDictionaryOptions{
                NTextProcessing::NDictionary::ETokenLevelType::Word
            }
        };
    }

    bool TTextProcessingOptions::HasOnlyDefaultDictionaries() const {
        return Dictionaries.IsDefault();
    }

    TVector<TTextColumnDictionaryOptions> TTextProcessingOptions::GetDefaultDictionaries() {
        return {BiGramDictionaryOptions(), WordDictionaryOptions()};
    }

    TMap<EFeatureCalcerType, TVector<TString>> TTextProcessingOptions::GetDefaultCalcerDictionaries() {
        static const TMap<EFeatureCalcerType, TVector<TString>> DefaultCalcerDictionary = {
            {EFeatureCalcerType::BoW, {"BiGram", "Word"}},
            {EFeatureCalcerType::NaiveBayes, {"Word"}},
            {EFeatureCalcerType::BM25, {"Word"}}
        };
        return DefaultCalcerDictionary;
    }

    TVector<TString> TTextProcessingOptions::GetDefaultCalcerDictionaries(
        EFeatureCalcerType calcerType
    ) {
        const auto& defaultCalcerDictionaries = GetDefaultCalcerDictionaries();
        CB_ENSURE_INTERNAL(
            defaultCalcerDictionaries.contains(calcerType),
            "No default dictionaries for feature calcer " << ToString(calcerType)
        );
        return defaultCalcerDictionaries.at(calcerType);
    }

    TTextProcessingOptions::TTextProcessingOptions(
        TVector<TTextColumnDictionaryOptions>&& dictionaries,
        TMap<TString, TVector<TTextFeatureProcessing>>&& textFeatureProcessing
    )
    : TTextProcessingOptions() {
        Dictionaries.Set(dictionaries);
        TextFeatureProcessing.Set(textFeatureProcessing);
    }

    TTextFeatureProcessing::TTextFeatureProcessing()
        : FeatureCalcer(
            "feature_calcer",
            TFeatureCalcerDescription{
                EFeatureCalcerType::NaiveBayes
            })
        , DictionariesNames("dictionaries_names", TVector<TString>{})
    {}

    void TTextFeatureProcessing::Save(NJson::TJsonValue* optionsJson) const {
        SaveFields(optionsJson, DictionariesNames, FeatureCalcer);
    }

    void TTextFeatureProcessing::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &DictionariesNames, &FeatureCalcer);
    }

    bool TTextFeatureProcessing::operator==(const TTextFeatureProcessing& rhs) const {
        return std::tie(DictionariesNames, FeatureCalcer)
            == std::tie(rhs.DictionariesNames, rhs.FeatureCalcer);
    }

    bool TTextFeatureProcessing::operator!=(const TTextFeatureProcessing& rhs) const {
        return !(*this == rhs);
    }

    TTextFeatureProcessing::TTextFeatureProcessing(
        TFeatureCalcerDescription&& featureCalcer,
        TVector<TString>&& dictionariesNames
    )
        : TTextFeatureProcessing()
    {
        FeatureCalcer.Set(featureCalcer);
        DictionariesNames.Set(dictionariesNames);
    }
}


static NJson::TJsonValue ParseDictionaryOptionsFromString(
    TStringBuf description,
    TSet<TStringBuf>* seenDictionaryNames
) {
    NJson::TJsonValue dictionaryOptions;
    dictionaryOptions.SetType(NJson::EJsonValueType::JSON_MAP);

    TStringBuf dictionaryId;
    GetNext<TStringBuf>(description, ":", dictionaryId);

    CB_ENSURE(!dictionaryId.empty(), "DictionaryOptions: DictionaryId is not specified");
    CB_ENSURE(
        !seenDictionaryNames->contains(dictionaryId),
        "DictionaryOptions: DictionaryId \"" << dictionaryId << "\" was specified twice"
    );
    dictionaryOptions["dictionary_id"] = dictionaryId;
    seenDictionaryNames->insert(dictionaryId);

    const THashSet<TString> validParams = {
        "token_level_type",
        "gram_order",
        "skip_step",
        "start_token_id",
        "end_of_word_token_policy",
        "end_of_sentence_token_policy",
        "min_token_occurrence",
        "max_dict_size"
    };

    for (const auto configItem : StringSplitter(description).Split(',').SkipEmpty()) {
        TStringBuf key;
        TStringBuf value;

        Split(configItem, '=', key, value);
        if (validParams.contains(key)) {
            dictionaryOptions[key] = value;
        } else {
            ythrow TCatBoostException() << "Unsupported dictionary option: " << key;
        }
    }

    return dictionaryOptions;
}

static void ParseTextProcessingOptionsFromString(
    TStringBuf description,
    const TSet<TStringBuf>& seenDictionaryNames,
    NJson::TJsonValue* options
) {
    TStringBuf textFeatureId;
    TStringBuf featureDescription;
    description.RSplit('~', textFeatureId, featureDescription);

    if (textFeatureId.empty()) {
        textFeatureId = NCatboostOptions::TTextProcessingOptions::DefaultProcessingName();
    }
    CB_ENSURE(!options->Has(textFeatureId), "Feature names in text processing options must be unique");
    NJson::TJsonValue& featureProcessingJson = (*options)[textFeatureId];
    featureProcessingJson.SetType(NJson::EJsonValueType::JSON_ARRAY);

    for (TStringBuf textProcessingUnit: StringSplitter(featureDescription).Split('|').SkipEmpty()) {
        TStringBuf calcer;
        TStringBuf dictionaries;
        textProcessingUnit.Split('+', calcer, dictionaries);

        NJson::TJsonValue unitJson(NJson::EJsonValueType::JSON_MAP);
        unitJson["feature_calcer"] = calcer;

        {
            NJson::TJsonValue& dictionaryNames = unitJson["dictionaries_names"];
            dictionaryNames.SetType(NJson::EJsonValueType::JSON_ARRAY);

            for (TStringBuf dictionaryName: StringSplitter(dictionaries).Split(',').SkipEmpty()) {
                dictionaryNames.AppendValue(dictionaryName);
                CB_ENSURE(
                    seenDictionaryNames.contains(dictionaryName),
                    "DictionaryName " << dictionaryName << " wasn't specified in dictionaries options"
                );
            }
        }

        featureProcessingJson.AppendValue(unitJson);
    }
}

void NCatboostOptions::ParseTextProcessingOptionsFromPlainJson(
    const NJson::TJsonValue& plainOptions,
    NJson::TJsonValue* textProcessingOptions,
    TSet<TString>* seenKeys
) {
    TSet<TStringBuf> seenDictionaryNames;

    const TString dictionaryOptionName = "dictionaries";
    if (plainOptions.Has(dictionaryOptionName)) {
        const NJson::TJsonValue& dictionaryPlainOptions = plainOptions[dictionaryOptionName];

        NJson::TJsonValue& dictionaryOptions = (*textProcessingOptions)[dictionaryOptionName];
        dictionaryOptions.SetType(NJson::EJsonValueType::JSON_ARRAY);

        for (const auto& dictionaryPlainOption: dictionaryPlainOptions.GetArray()) {
            auto dictionaryOption = ParseDictionaryOptionsFromString(
                dictionaryPlainOption.GetString(),
                &seenDictionaryNames
            );
            dictionaryOptions.AppendValue(std::move(dictionaryOption));
        }
        seenKeys->insert(dictionaryOptionName);
    }

    const TString textProcessingOptionsName = "text_processing";
    if (plainOptions.Has(textProcessingOptionsName)) {
        const NJson::TJsonValue& textProcessingPlainOptions = plainOptions[textProcessingOptionsName];

        NJson::TJsonValue& textProcessingJson = (*textProcessingOptions)[textProcessingOptionsName];
        textProcessingJson.SetType(NJson::EJsonValueType::JSON_MAP);

        for (const auto& textProcessingPlainOption: textProcessingPlainOptions.GetArray()) {
            ParseTextProcessingOptionsFromString(
                textProcessingPlainOption.GetString(),
                seenDictionaryNames,
                &textProcessingJson
            );
        }
        seenKeys->insert(textProcessingOptionsName);
    }
}

static TString DictionaryOptionsToString(NJson::TJsonValue options) { // copy here is intentional
    TStringBuilder descriptionBuilder;

    TString dictionaryId = options["dictionary_id"].GetString();
    options.EraseValue("dictionary_id");

    descriptionBuilder << dictionaryId << ':';

    for (auto& [parameterName, parameterValue]: options.GetMap()) {
        descriptionBuilder << parameterName << '=' << parameterValue.GetString() << ',';
    }

    descriptionBuilder.pop_back();
    return descriptionBuilder.data();
}

static TString FeatureProcessingDescriptionToString(
    TStringBuf textFeatureId,
    const NJson::TJsonValue& options
) {
    TStringBuilder descriptionBuilder;
    {
        descriptionBuilder << textFeatureId << '~';

        for (auto& featureCalcerProcessing: options.GetArray()) {
            descriptionBuilder << featureCalcerProcessing["feature_calcer"].GetString() << '+';

            for (auto& dictionaryName: featureCalcerProcessing["dictionaries_names"].GetArray()) {
                descriptionBuilder << dictionaryName.GetString() << ',';
            }
            descriptionBuilder.pop_back();
            descriptionBuilder << '|';
        }

        descriptionBuilder.pop_back();
    }

    return descriptionBuilder.data();
}

void NCatboostOptions::SaveTextProcessingOptionsToPlainJson(
    const NJson::TJsonValue& textProcessingOptions,
    NJson::TJsonValue* plainOptions
) {
    NJson::TJsonValue& dictionaries = (*plainOptions)["dictionaries"];
    dictionaries.SetType(NJson::EJsonValueType::JSON_ARRAY);

    for (const auto& dictionary: textProcessingOptions["dictionaries"].GetArray()) {
        TString description = DictionaryOptionsToString(dictionary);
        dictionaries.AppendValue(NJson::TJsonValue(description));
    }

    NJson::TJsonValue& textProcessing = (*plainOptions)["text_processing"];
    textProcessing.SetType(NJson::EJsonValueType::JSON_ARRAY);

    for (const auto& [textFeatureId, featureProcessing]: textProcessingOptions["text_processing"].GetMap()) {
        TString featureProcessingDescription = FeatureProcessingDescriptionToString(
            textFeatureId,
            featureProcessing
        );
        textProcessing.AppendValue(NJson::TJsonValue{featureProcessingDescription});
    }
}
