#include "text_feature_options.h"
#include "json_helper.h"

#include <util/string/cast.h>
#include <util/string/split.h>
#include <util/generic/hash_set.h>

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

static void DictionaryBuilderOptionsToJson(const TDictionaryBuilderOptions& options, NJson::TJsonValue* optionsJson) {
    (*optionsJson)["min_token_occurence"] = ToString(options.OccurrenceLowerBound);
    (*optionsJson)["max_dict_size"] = ToString(options.MaxDictionarySize);
}

static void DictionaryOptionsToJson(const TDictionaryOptions& options, NJson::TJsonValue* optionsJson) {
    (*optionsJson)["token_level_type"] = ToString(options.TokenLevelType);
    (*optionsJson)["gram_order"] = ToString(options.GramOrder);
    (*optionsJson)["skip_step"] = ToString(options.SkipStep);
    (*optionsJson)["start_token_id"] = ToString(options.StartTokenId);
    (*optionsJson)["end_of_word_token_policy"] = ToString(options.EndOfWordTokenPolicy);
    (*optionsJson)["end_of_sentence_token_policy"] = ToString(options.EndOfSentenceTokenPolicy);
}

NCatboostOptions::TTextProcessingOptions::TTextProcessingOptions()
    : DictionaryOptions("dictionary_options",
        TDictionaryOptions{
            /*TokenLevelType*/ ETokenLevelType::Word,
            /*GramOrder*/ 1,
            /*SkipStep (skip-grams)*/ 0,
            /*StartTokenId*/ 0,
            /*EndOfWordTokenPolicy*/ EEndOfWordTokenPolicy::Insert,
            /*EndOfSentenceTokenPolicy*/ EEndOfSentenceTokenPolicy::Skip})
    , DictionaryBuilderOptions("dictionary_builder_options",
        TDictionaryBuilderOptions{
            /*OccurrenceLowerBound*/ 3,
            /*MaxDictionarySize*/ -1})
    , TokenizerType("tokenizer_type", ETokenizerType::Naive)
{
}

void NCatboostOptions::TTextProcessingOptions::Save(NJson::TJsonValue* optionsJson) const {
    DictionaryOptionsToJson(DictionaryOptions, optionsJson);
    DictionaryBuilderOptionsToJson(DictionaryBuilderOptions, optionsJson);
    (*optionsJson)["tokenizer_type"] = ToString<ETokenizerType>(TokenizerType);
}

void NCatboostOptions::TTextProcessingOptions::Load(const NJson::TJsonValue& options) {
    DictionaryOptions = JsonToDictionaryOptions(options);
    DictionaryBuilderOptions = GetDictionaryBuilderOptions(options);
    if (!TryFromString<ETokenizerType>(options["tokenizer_type"].GetString(), TokenizerType)) {
        TokenizerType = ETokenizerType::Naive;
    }
}

bool NCatboostOptions::TTextProcessingOptions::operator==(const TTextProcessingOptions& rhs) const {
    return std::tie(DictionaryOptions, DictionaryBuilderOptions, TokenizerType) ==
           std::tie(rhs.DictionaryOptions, rhs.DictionaryBuilderOptions, rhs.TokenizerType);
}

bool NCatboostOptions::TTextProcessingOptions::operator!=(const TTextProcessingOptions& rhs) const {
    return !(rhs == *this);
}

std::pair<TStringBuf, NJson::TJsonValue> NCatboostOptions::ParsePerTextFeatureProcessing(TStringBuf description) {
    std::pair<TStringBuf, NJson::TJsonValue> perFeatureProcessing;
    GetNext<TStringBuf>(description, ":", perFeatureProcessing.first);

    const THashSet<TString> validParams = {
        "token_level_type",
        "gram_order",
        "skip_step",
        "start_token_id",
        "end_of_word_token_policy",
        "end_of_sentence_token_policy",
        "min_token_occurence",
        "max_dict_size",
        "tokenizer_type"
    };

    for (const auto configItem : StringSplitter(description).Split(':').SkipEmpty()) {
        TStringBuf key, value;
        Split(configItem, '=', key, value);
        if (validParams.contains(key)) {
            perFeatureProcessing.second[key] = value;
        } else {
            ythrow TCatBoostException() << "Unsupported text processing option: " << key;
        }
    }
    return perFeatureProcessing;
}

NCatboostOptions::TTextProcessingOptionCollection::TTextProcessingOptionCollection()
    : FeatureIdToTextProcessing("per_feature_text_processing", TMap<ui32, TTextProcessingOptions>())
{}

const NCatboostOptions::TTextProcessingOptions& NCatboostOptions::TTextProcessingOptionCollection::GetFeatureTextProcessing(ui32 featureId) const {
    if (FeatureIdToTextProcessing->contains(featureId)) {
        return FeatureIdToTextProcessing->at(featureId);
    }
    return DefaultTextProcessing;
}

void NCatboostOptions::TTextProcessingOptionCollection::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &FeatureIdToTextProcessing);
}

void NCatboostOptions::TTextProcessingOptionCollection::Save(NJson::TJsonValue* optionsJson) const {
    SaveFields(optionsJson, FeatureIdToTextProcessing);
}

bool NCatboostOptions::TTextProcessingOptionCollection::operator==(
    const NCatboostOptions::TTextProcessingOptionCollection& rhs) const {
    return FeatureIdToTextProcessing == rhs.FeatureIdToTextProcessing;
}

bool NCatboostOptions::TTextProcessingOptionCollection::operator!=(
    const NCatboostOptions::TTextProcessingOptionCollection& rhs) const {
    return !(rhs == *this);
}

NCatboostOptions::TTextFeatureOptions::TTextFeatureOptions()
    : FeatureEstimators("text_feature_estimators", {EFeatureCalcerType::BoW, EFeatureCalcerType::NaiveBayes})
{
}

void NCatboostOptions::TTextFeatureOptions::Load(const NJson::TJsonValue& options) {
    CheckedLoad(options, &FeatureEstimators);
}

void NCatboostOptions::TTextFeatureOptions::Save(NJson::TJsonValue* options) const {
    SaveFields(options, FeatureEstimators);
}

bool NCatboostOptions::TTextFeatureOptions::operator==(const TTextFeatureOptions& rhs) const {
    return FeatureEstimators == rhs.FeatureEstimators;
}

bool NCatboostOptions::TTextFeatureOptions::operator!=(const TTextFeatureOptions& rhs) const {
    return !(rhs == *this);
}
