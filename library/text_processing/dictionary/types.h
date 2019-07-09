#pragma once

#include <util/generic/ylimits.h>
#include <util/system/types.h>

namespace NTextProcessing::NDictionary {

    enum class ETokenLevelType {
        Word,
        Letter
    };

    enum class EUnknownTokenPolicy {
        Skip,
        Insert
    };

    enum class EEndOfWordTokenPolicy {
        Skip,
        Insert
    };

    enum class EEndOfSentenceTokenPolicy {
        Skip,
        Insert
    };

    enum class EContextLevel {
        Word,
        Sentence
    };

    enum class EDictionaryType {
        FrequencyBased,
        Bpe
    };

    using TTokenId = ui32;

}
