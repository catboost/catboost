#include "options.h"

#include <util/string/cast.h>

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
            *result = FromString<TType>(optionsJson[name].GetString());
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

}
