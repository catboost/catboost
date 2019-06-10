#include "options.h"

#include <util/string/cast.h>
#include <tuple>

namespace NTextProcessing::NDictionary {
    template <typename TType>
    static void SetOption(const TType& value, const TString& name, NJson::TJsonValue* optionsJson) {
        (*optionsJson)[name] = ToString(value);
    }

    NJson::TJsonValue DictionaryOptionsToJson(const TDictionaryOptions& options) {
        NJson::TJsonValue optionsJson;
        SetOption(options.TokenLevelType, "token_level_type", &optionsJson);
        SetOption(options.GramOrder, "gram_order", &optionsJson);
        SetOption(options.SkipStep, "skip_step", &optionsJson);
        SetOption(options.StartTokenId, "start_token_id", &optionsJson);
        SetOption(options.EndOfWordTokenPolicy, "end_of_word_token_policy", &optionsJson);
        SetOption(options.EndOfSentenceTokenPolicy, "end_of_sentence_token_policy", &optionsJson);
        return optionsJson;
    }

    template <typename TType>
    static void GetOption(const NJson::TJsonValue& optionsJson, const TString& name, TType* result) {
        if (optionsJson.Has(name)) {
            TStringBuf value = (optionsJson[name]).GetString();
            const bool isParsed = TryFromString<TType>(value, *result);
            Y_VERIFY(isParsed, "Couldn't parse option \"%s\" with value = %s", name.data(), value.data());
        }
    }

    TDictionaryOptions JsonToDictionaryOptions(const NJson::TJsonValue& optionsJson) {
        TDictionaryOptions options;
        GetOption(optionsJson, "token_level_type", &options.TokenLevelType);
        GetOption(optionsJson, "gram_order", &options.GramOrder);
        GetOption(optionsJson, "skip_step", &options.SkipStep);
        GetOption(optionsJson, "start_token_id", &options.StartTokenId);
        GetOption(optionsJson, "end_of_word_token_policy", &options.EndOfWordTokenPolicy);
        GetOption(optionsJson, "end_of_sentence_token_policy", &options.EndOfSentenceTokenPolicy);
        return options;
    }

    bool TDictionaryOptions::operator==(const TDictionaryOptions& rhs) const {
        return std::tie(
            TokenLevelType,
            GramOrder,
            SkipStep,
            StartTokenId,
            EndOfWordTokenPolicy,
            EndOfSentenceTokenPolicy) ==
            std::tie(
                rhs.TokenLevelType,
                rhs.GramOrder,
                rhs.SkipStep,
                rhs.StartTokenId,
                rhs.EndOfWordTokenPolicy,
                rhs.EndOfSentenceTokenPolicy);
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
