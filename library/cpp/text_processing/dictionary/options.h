#pragma once

#include "types.h"

#include <library/cpp/json/json_value.h>

#include <util/ysaveload.h>

namespace NTextProcessing::NDictionary {

    struct TDictionaryOptions {
        ETokenLevelType TokenLevelType = ETokenLevelType::Word;
        ui32 GramOrder = 1; // 1 for Unigram, 2 for Bigram, ...
        ui32 SkipStep = 0;  // 1 for 1-skip-gram, ...
        TTokenId StartTokenId = 0; // Initial shift for tokenId

        // Used in conjunction with ETokenLevelType::Letter TokenLevelType option.
        EEndOfWordTokenPolicy EndOfWordTokenPolicy = EEndOfWordTokenPolicy::Insert;

        // Used in conjunction with ETokenLevelType::Word TokenLevelType option.
        EEndOfSentenceTokenPolicy EndOfSentenceTokenPolicy = EEndOfSentenceTokenPolicy::Skip;

        Y_SAVELOAD_DEFINE(
            TokenLevelType,
            GramOrder,
            SkipStep,
            StartTokenId,
            EndOfWordTokenPolicy,
            EndOfSentenceTokenPolicy
        );

        bool operator==(const TDictionaryOptions& rhs) const;
        bool operator!=(const TDictionaryOptions& rhs) const;
    };

    struct TDictionaryBuilderOptions {
        ui64 OccurrenceLowerBound = 50; // The lower bound of token occurrences in the text to include it in the dictionary.
        i32 MaxDictionarySize = -1; // The max dictionary size.

        Y_SAVELOAD_DEFINE(OccurrenceLowerBound, MaxDictionarySize);

        bool operator==(const TDictionaryBuilderOptions& rhs) const;
        bool operator!=(const TDictionaryBuilderOptions& rhs) const;
    };

    struct TBpeDictionaryOptions {
        size_t NumUnits = 0;
        bool SkipUnknown = false;

        // Used in conjunction with Letter TokenLevelType.
        //   Word - BPE units are created from each word in isolation.
        //     Letter sequences include letters inside the word and end of word token.
        //     It works faster and requires less memory then Sentence type.
        //   Sentence - BPE units are created from a whole sentence by taking into account spaces between words.
        EContextLevel ContextLevel = EContextLevel::Word;

        Y_SAVELOAD_DEFINE(NumUnits, SkipUnknown);
    };

    void DictionaryOptionsToJson(const TDictionaryOptions& options, NJson::TJsonValue* optionsJson);
    void JsonToDictionaryOptions(const NJson::TJsonValue& optionsJson, TDictionaryOptions* options);
    NJson::TJsonValue DictionaryOptionsToJson(const TDictionaryOptions& options);
    TDictionaryOptions JsonToDictionaryOptions(const NJson::TJsonValue& optionsJson);

    void DictionaryBuilderOptionsToJson(const TDictionaryBuilderOptions& options, NJson::TJsonValue* optionsJson);
    void JsonToDictionaryBuilderOptions(const NJson::TJsonValue& optionsJson, TDictionaryBuilderOptions* options);
    NJson::TJsonValue DictionaryBuilderOptionsToJson(const TDictionaryBuilderOptions& options);
    TDictionaryBuilderOptions JsonToDictionaryBuilderOptions(const NJson::TJsonValue& optionsJson);
}
