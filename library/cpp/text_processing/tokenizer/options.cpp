#include "options.h"

#include <library/cpp/json/json_reader.h>
#include <util/string/cast.h>

using NTextProcessing::NTokenizer::TTokenizerOptions;

static const TString LOWERCASING_OPTION_NAME = "lowercasing";
static const TString LEMMATIZING_OPTION_NAME = "lemmatizing";
static const TString NUMBER_PROCESS_POLICY_OPTION_NAME = "number_process_policy";
static const TString NUMBER_TOKEN_OPTION_NAME = "number_token";
static const TString SEPARATOR_TYPE_OPTION_NAME = "separator_type";
static const TString DELIMITER_OPTION_NAME = "delimiter";
static const TString SPLIT_BY_SET_OPTION_NAME = "split_by_set";
static const TString SKIP_EMPTY_OPTION_NAME = "skip_empty";
static const TString TOKEN_TYPES_OPTION_NAME = "token_types";
static const TString SUBTOKENS_POLICY_OPTION_NAME = "subtokens_policy";
static const TString LANGUAGES_OPTION_NAME = "languages";
static const TString LEMMER_CACHESIZE_OPTION_NAME = "lemmer_cache_size";

template <typename TType>
static void SetOption(const TType& value, const TString& name, NJson::TJsonValue* optionsJson) {
    (*optionsJson)[name] = ToString(value);
}

template <typename TContainerType>
static void SetContainerOption(const TContainerType& container, const TString& name, NJson::TJsonValue* optionsJson) {
    optionsJson->InsertValue(name, NJson::EJsonValueType::JSON_ARRAY);
    for (const auto& element : container) {
        (*optionsJson)[name].AppendValue(ToString(element));
    }
}

template <typename TType>
static void GetOption(const NJson::TJsonValue& optionsJson, const TString& name, TType* result) {
    if (optionsJson.Has(name)) {
        *result = FromString<TType>(optionsJson[name].GetString());
    }
}

template <typename TContainerType>
static void GetContainerOption(const NJson::TJsonValue& optionsJson, const TString& name, TContainerType* result) {
    if (optionsJson.Has(name)) {
        result->clear();
        for (const auto& element : optionsJson[name].GetArray()) {
            result->insert(result->end(), FromString<typename TContainerType::value_type>(element.GetString()));
        }
    }
}

void NTextProcessing::NTokenizer::TokenizerOptionsToJson(const TTokenizerOptions& options, NJson::TJsonValue* optionsJson) {
    SetOption(options.Lowercasing, LOWERCASING_OPTION_NAME, optionsJson);
    SetOption(options.Lemmatizing, LEMMATIZING_OPTION_NAME, optionsJson);
    SetOption(options.NumberProcessPolicy, NUMBER_PROCESS_POLICY_OPTION_NAME, optionsJson);
    SetOption(options.NumberToken, NUMBER_TOKEN_OPTION_NAME, optionsJson);
    SetOption(options.SeparatorType, SEPARATOR_TYPE_OPTION_NAME, optionsJson);
    SetOption(options.Delimiter, DELIMITER_OPTION_NAME, optionsJson);
    SetOption(options.SplitBySet, SPLIT_BY_SET_OPTION_NAME, optionsJson);
    SetOption(options.SkipEmpty, SKIP_EMPTY_OPTION_NAME, optionsJson);
    SetContainerOption(options.TokenTypes, TOKEN_TYPES_OPTION_NAME, optionsJson);
    SetOption(options.SubTokensPolicy, SUBTOKENS_POLICY_OPTION_NAME, optionsJson);
    TVector<TString> stringLanguages;
    for (const auto& language : options.Languages) {
        stringLanguages.push_back(FullNameByLanguage(language));
    }
    SetContainerOption(stringLanguages, LANGUAGES_OPTION_NAME, optionsJson);
    if (options.LemmerCacheSize != DEFAULT_LEMMER_CACHE_SIZE) {
        SetOption(options.LemmerCacheSize, LEMMER_CACHESIZE_OPTION_NAME, optionsJson);
    }
}

void NTextProcessing::NTokenizer::JsonToTokenizerOptions(const NJson::TJsonValue& optionsJson, TTokenizerOptions* options) {
    GetOption(optionsJson, LOWERCASING_OPTION_NAME, &options->Lowercasing);
    GetOption(optionsJson, LEMMATIZING_OPTION_NAME, &options->Lemmatizing);
    GetOption(optionsJson, NUMBER_PROCESS_POLICY_OPTION_NAME, &options->NumberProcessPolicy);
    GetOption(optionsJson, NUMBER_TOKEN_OPTION_NAME, &options->NumberToken);
    GetOption(optionsJson, SEPARATOR_TYPE_OPTION_NAME, &options->SeparatorType);
    GetOption(optionsJson, DELIMITER_OPTION_NAME, &options->Delimiter);
    GetOption(optionsJson, SPLIT_BY_SET_OPTION_NAME, &options->SplitBySet);
    GetOption(optionsJson, SKIP_EMPTY_OPTION_NAME, &options->SkipEmpty);
    GetContainerOption(optionsJson, TOKEN_TYPES_OPTION_NAME, &options->TokenTypes);
    GetOption(optionsJson, SUBTOKENS_POLICY_OPTION_NAME, &options->SubTokensPolicy);
    TVector<TString> stringLanguages;
    GetContainerOption(optionsJson, LANGUAGES_OPTION_NAME, &stringLanguages);
    options->Languages.clear();
    for (const auto& language : stringLanguages) {
        options->Languages.push_back(LanguageByNameOrDie(language));
    }
    GetOption(optionsJson, LEMMER_CACHESIZE_OPTION_NAME, &options->LemmerCacheSize);
}

NJson::TJsonValue NTextProcessing::NTokenizer::TokenizerOptionsToJson(const TTokenizerOptions& options) {
    NJson::TJsonValue optionsJson;
    TokenizerOptionsToJson(options, &optionsJson);
    return optionsJson;
}

TTokenizerOptions NTextProcessing::NTokenizer::JsonToTokenizerOptions(const NJson::TJsonValue& optionsJson) {
    TTokenizerOptions options;
    JsonToTokenizerOptions(optionsJson, &options);
    return options;
}

bool TTokenizerOptions::operator==(const TTokenizerOptions& rhs) const {
    return std::tie(
        Lowercasing, Lemmatizing, NumberProcessPolicy, NumberToken, SeparatorType, Delimiter, SplitBySet,
        SkipEmpty, TokenTypes, SubTokensPolicy, Languages, LemmerCacheSize
    ) == std::tie(
        rhs.Lowercasing, rhs.Lemmatizing, rhs.NumberProcessPolicy, rhs.NumberToken, rhs.SeparatorType,
        rhs.Delimiter, rhs.SplitBySet, rhs.SkipEmpty, rhs.TokenTypes, rhs.SubTokensPolicy, rhs.Languages,
        rhs.LemmerCacheSize
    );
}

bool TTokenizerOptions::operator!=(const TTokenizerOptions& rhs) const {
    return !(rhs == *this);
}
