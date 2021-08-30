#pragma once

#include <library/cpp/json/json_value.h>
#include <library/cpp/langs/langs.h>

#include <util/ysaveload.h>
#include <util/generic/hash_set.h>
#include <util/generic/serialized_enum.h>

namespace NTextProcessing::NTokenizer {

    enum class ESeparatorType {
        ByDelimiter,
        BySense
    };

    enum class ETokenType {
        Word,
        Number,
        Punctuation,
        SentenceBreak,
        ParagraphBreak,
        Unknown
    };

    enum class ESubTokensPolicy {
        SingleToken, // All subtokens are interpreted as single token
        SeveralTokens // All subtokens are interpreted as several token
    };

    enum class ETokenProcessPolicy {
        Skip, // Skip token
        LeaveAsIs, // Leave token as is
        Replace // Replace all tokens with single token
    };

    inline constexpr size_t DEFAULT_LEMMER_CACHE_SIZE = 0u;

    struct TTokenizerOptions {
        bool Lowercasing = false;
        bool Lemmatizing = false;
        ETokenProcessPolicy NumberProcessPolicy = ETokenProcessPolicy::LeaveAsIs;
        TString NumberToken = "🔢"; // Rarely used character. Used in conjunction with Replace NumberProcessPolicy

        ESeparatorType SeparatorType = ESeparatorType::ByDelimiter;

        // Used in conjunction with ByDelimiter SeparatorType.
        TString Delimiter = " ";
        bool SplitBySet = false; // Each single character in delimiter used as individual delimiter.
        bool SkipEmpty = true; // Skip all empty tokens.

        // Used in conjunction with BySense SeparatorType.
        THashSet<ETokenType> TokenTypes = {ETokenType::Word, ETokenType::Number, ETokenType::Unknown};
        ESubTokensPolicy SubTokensPolicy = ESubTokensPolicy::SingleToken;

        // Used in conjunction with true Lemmatizing option.
        TVector<ELanguage> Languages; // Empty for all languages.
        // this options cannot be included to saveload list, because it breaks compatibility for current dumps
        size_t LemmerCacheSize = DEFAULT_LEMMER_CACHE_SIZE;  // lemmer cache size (0 - disabled)

        Y_SAVELOAD_DEFINE(
            Lowercasing,
            Lemmatizing,
            NumberProcessPolicy,
            NumberToken,
            SeparatorType,
            Delimiter,
            SplitBySet,
            SkipEmpty,
            TokenTypes,
            SubTokensPolicy,
            Languages
        );

        bool operator==(const TTokenizerOptions& rhs) const;
        bool operator!=(const TTokenizerOptions& rhs) const;
    };

    void TokenizerOptionsToJson(const TTokenizerOptions& options, NJson::TJsonValue* optionsJson);
    void JsonToTokenizerOptions(const NJson::TJsonValue& optionsJson, TTokenizerOptions* options);
    NJson::TJsonValue TokenizerOptionsToJson(const TTokenizerOptions& options);
    TTokenizerOptions JsonToTokenizerOptions(const NJson::TJsonValue& optionsJson);

}
