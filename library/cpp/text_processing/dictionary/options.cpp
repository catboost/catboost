#include "options.h"

#include <util/string/cast.h>
#include <tuple>

static const TString TOKEN_LEVEL_TYPE = "token_level_type";
static const TString GRAM_ORDER = "gram_order";
static const TString SKIP_STEP = "skip_step";
static const TString START_TOKEN_ID = "start_token_id";
static const TString END_OF_WORD_TOKEN_POLICY = "end_of_word_token_policy";
static const TString END_OF_SENTENCE_TOKEN_POLICY = "end_of_sentence_token_policy";
static const TString OCCURRENCE_LOWER_BOUND = "occurrence_lower_bound";
static const TString MAX_DICTIONARY_SIZE = "max_dictionary_size";

namespace NTextProcessing::NDictionary {
    template <typename TType>
    static void SetOption(const TType& value, const TString& name, NJson::TJsonValue* optionsJson) {
        (*optionsJson)[name] = ToString(value);
    }

    void DictionaryOptionsToJson(const TDictionaryOptions& options, NJson::TJsonValue* optionsJson) {
        SetOption(options.TokenLevelType, TOKEN_LEVEL_TYPE, optionsJson);
        SetOption(options.GramOrder, GRAM_ORDER, optionsJson);
        SetOption(options.SkipStep, SKIP_STEP, optionsJson);
        SetOption(options.StartTokenId, START_TOKEN_ID, optionsJson);
        SetOption(options.EndOfWordTokenPolicy, END_OF_WORD_TOKEN_POLICY, optionsJson);
        SetOption(options.EndOfSentenceTokenPolicy, END_OF_SENTENCE_TOKEN_POLICY, optionsJson);
    }

    NJson::TJsonValue DictionaryOptionsToJson(const TDictionaryOptions& options) {
        NJson::TJsonValue optionsJson;
        DictionaryOptionsToJson(options, &optionsJson);
        return optionsJson;
    }

    void DictionaryBuilderOptionsToJson(const TDictionaryBuilderOptions& options, NJson::TJsonValue* optionsJson) {
        SetOption(options.OccurrenceLowerBound, OCCURRENCE_LOWER_BOUND, optionsJson);
        SetOption(options.MaxDictionarySize, MAX_DICTIONARY_SIZE, optionsJson);
    }

    NJson::TJsonValue DictionaryBuilderOptionsToJson(const TDictionaryBuilderOptions& options) {
        NJson::TJsonValue optionsJson;
        DictionaryBuilderOptionsToJson(options, &optionsJson);
        return optionsJson;
    }

    template <typename TType>
    static void GetOption(const NJson::TJsonValue& optionsJson, const TString& name, TType* result) {
        if (optionsJson.Has(name)) {
            TStringBuf value = (optionsJson[name]).GetString();
            const bool isParsed = TryFromString<TType>(value, *result);
            Y_ABORT_UNLESS(isParsed, "Couldn't parse option \"%s\" with value = %s", name.data(), value.data());
        }
    }

    void JsonToDictionaryOptions(const NJson::TJsonValue& optionsJson, TDictionaryOptions* options) {
        GetOption(optionsJson, TOKEN_LEVEL_TYPE, &options->TokenLevelType);
        GetOption(optionsJson, GRAM_ORDER, &options->GramOrder);
        GetOption(optionsJson, SKIP_STEP, &options->SkipStep);
        GetOption(optionsJson, START_TOKEN_ID, &options->StartTokenId);
        GetOption(optionsJson, END_OF_WORD_TOKEN_POLICY, &options->EndOfWordTokenPolicy);
        GetOption(optionsJson, END_OF_SENTENCE_TOKEN_POLICY, &options->EndOfSentenceTokenPolicy);
    }

    TDictionaryOptions JsonToDictionaryOptions(const NJson::TJsonValue& optionsJson) {
        TDictionaryOptions options;
        JsonToDictionaryOptions(optionsJson, &options);
        return options;
    }

    void JsonToDictionaryBuilderOptions(const NJson::TJsonValue& optionsJson, TDictionaryBuilderOptions* options) {
        GetOption(optionsJson, OCCURRENCE_LOWER_BOUND, &options->OccurrenceLowerBound);
        GetOption(optionsJson, MAX_DICTIONARY_SIZE, &options->MaxDictionarySize);
    }

    TDictionaryBuilderOptions JsonToDictionaryBuliderOptions(const NJson::TJsonValue& optionsJson) {
        TDictionaryBuilderOptions options;
        JsonToDictionaryBuilderOptions(optionsJson, &options);
        return options;
    }

    bool TDictionaryOptions::operator==(const TDictionaryOptions& rhs) const {
        return std::tie(
            TokenLevelType, GramOrder, SkipStep, StartTokenId, EndOfWordTokenPolicy, EndOfSentenceTokenPolicy
        ) == std::tie(
            rhs.TokenLevelType, rhs.GramOrder, rhs.SkipStep, rhs.StartTokenId, rhs.EndOfWordTokenPolicy,
            rhs.EndOfSentenceTokenPolicy
        );
    }

    bool TDictionaryOptions::operator!=(const NTextProcessing::NDictionary::TDictionaryOptions& rhs) const {
        return !(rhs == *this);
    }

    bool TDictionaryBuilderOptions::operator==(const TDictionaryBuilderOptions& rhs) const {
        return OccurrenceLowerBound == rhs.OccurrenceLowerBound &&
           MaxDictionarySize == rhs.MaxDictionarySize;
    }

    bool TDictionaryBuilderOptions::operator!=(const TDictionaryBuilderOptions& rhs) const {
        return !(rhs == *this);
    }
}
