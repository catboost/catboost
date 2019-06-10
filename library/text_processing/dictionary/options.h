#pragma once

#include "types.h"

#include <library/json/json_value.h>

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

        Y_SAVELOAD_DEFINE(NumUnits, SkipUnknown);
    };

    NJson::TJsonValue DictionaryOptionsToJson(const TDictionaryOptions& options);
    TDictionaryOptions JsonToDictionaryOptions(const NJson::TJsonValue& optionsJson);
}
