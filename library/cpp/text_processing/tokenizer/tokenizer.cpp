#include "tokenizer.h"

#include <library/cpp/cache/cache.h>
#include <library/cpp/tokenizer/tokenizer.h>

#include <util/generic/maybe.h>
#include <util/string/split.h>
#include <util/string/strip.h>
#include <util/string/type.h>
#include <util/system/spinlock.h>
#include <util/system/guard.h>

using namespace NTextProcessing;
using NTextProcessing::NTokenizer::ESubTokensPolicy;
using NTextProcessing::NTokenizer::EImplementationType;
using NTextProcessing::NTokenizer::ILemmerImplementation;
using NTextProcessing::NTokenizer::TLemmerImplementationFactory;


namespace {
    class TStringCapacity {
    public:
        size_t operator()(const TUtf16String& s) const {
            return sizeof(typename TUtf16String::value_type) * s.capacity();
        }
    };

    class TLemmerWithCache : public ILemmerImplementation {
        using TLemmerCache = TLRUCache<TUtf16String, TUtf16String, TNoopDelete, TStringCapacity>;
    public:
        TLemmerWithCache(THolder<ILemmerImplementation> lemmer, size_t cacheSize)
            : Lemmer(std::move(lemmer))
            , LemmerCache(cacheSize)
        {
        }

        void Lemmatize(TUtf16String* token) const override {
            with_lock (Lock) {
                auto it = LemmerCache.Find(*token);
                if (it != LemmerCache.End()) {
                    *token = *it;
                    return;
                }
            }

            const auto key = *token;
            Lemmer->Lemmatize(token);
            with_lock (Lock) {
                LemmerCache.Insert(key, *token);
            }
        }

    private:
        THolder<ILemmerImplementation> Lemmer;
        mutable TAdaptiveLock Lock;
        mutable TLemmerCache LemmerCache;
    };
}

static NTokenizer::ETokenType ConvertTokenType(NLP_TYPE tokenType) {
    switch (tokenType) {
        case NLP_WORD:
            return NTokenizer::ETokenType::Word;
        case NLP_INTEGER:
        case NLP_FLOAT:
            return NTokenizer::ETokenType::Number;
        case NLP_SENTBREAK:
            return NTokenizer::ETokenType::SentenceBreak;
        case NLP_PARABREAK:
            return NTokenizer::ETokenType::ParagraphBreak;
        case NLP_MISCTEXT:
            return NTokenizer::ETokenType::Punctuation;
        default:
            return NTokenizer::ETokenType::Unknown;
    }
}

static void ProcessWordToken(const NTokenizer::TTokenizerOptions& options, ILemmerImplementation* lemmer, TUtf16String* token) {
    if (options.Lowercasing) {
        ToLower(*token);
    }
    if (options.Lemmatizing) {
        lemmer->Lemmatize(token);
    }
}

static std::pair<const wchar16*, size_t> BuildSubToken(const TWideToken& rawToken, const TCharSpan& subTokenInfo) {
    return std::make_pair(
        rawToken.Token + subTokenInfo.Pos - subTokenInfo.PrefixLen,
        subTokenInfo.Len + subTokenInfo.SuffixLen + subTokenInfo.PrefixLen
    );
}

static bool IsWordChanged(const NTokenizer::TTokenizerOptions& options) {
    return options.Lemmatizing || options.Lowercasing;
}

namespace {

    class TTokenHandler : public ITokenHandler {
    public:
        TTokenHandler(
            TVector<TString>* tokens,
            TVector<NTokenizer::ETokenType>* tokenTypes,
            const NTokenizer::TTokenizerOptions& options,
            ILemmerImplementation* lemmer
        )
            : Tokens(tokens)
            , TokenTypes(tokenTypes)
            , Options(options)
            , Lemmer(lemmer)
        {
        }

        void OnToken(const TWideToken& rawToken, size_t, NLP_TYPE tokenNlpType) override {
            const auto tokenType = ConvertTokenType(tokenNlpType);
            if (Options.TokenTypes.contains(tokenType)) {
                if (tokenType == NTokenizer::ETokenType::Word) {
                    if (Options.SubTokensPolicy == ESubTokensPolicy::SeveralTokens) {
                        for (const auto& subTokenInfo : rawToken.SubTokens) {
                            auto [subTokenData, subTokenLen] = BuildSubToken(rawToken, subTokenInfo);
                            if (IsWordChanged(Options)) {
                                TUtf16String subToken(subTokenData, subTokenLen);
                                ProcessWordToken(Options, Lemmer, &subToken);
                                AddTokenInfo(WideToUTF8(subToken), tokenType);
                            } else {
                                AddTokenInfo(WideToUTF8(subTokenData, subTokenLen), tokenType);
                            }
                        }

                    } else {
                        Y_ENSURE(Options.SubTokensPolicy == ESubTokensPolicy::SingleToken,
                            "Unsupported ESubTokensPolicy.");
                        if (IsWordChanged(Options)) {
                            TUtf16String token(rawToken.Token, rawToken.Leng);
                            ProcessWordToken(Options, Lemmer, &token);
                            AddTokenInfo(WideToUTF8(token), tokenType);
                        } else {
                            AddTokenInfo(WideToUTF8(rawToken.Token, rawToken.Leng), tokenType);
                        }
                    }
                } else if (tokenType == NTokenizer::ETokenType::Number) {
                    if (Options.NumberProcessPolicy == NTokenizer::ETokenProcessPolicy::Replace) {
                        AddTokenInfo(Options.NumberToken, tokenType);
                    } else if (Options.NumberProcessPolicy == NTokenizer::ETokenProcessPolicy::LeaveAsIs) {
                        AddTokenInfo(WideToUTF8(rawToken.Token, rawToken.Leng), tokenType);
                    }
                } else if (tokenType == NTokenizer::ETokenType::Punctuation) {
                    TString token(WideToUTF8(rawToken.Token, rawToken.Leng));
                    StripInPlace(token);
                    if (!token.empty()) {
                        AddTokenInfo(std::move(token), tokenType);
                    }
                } else {
                    AddTokenInfo(WideToUTF8(rawToken.Token, rawToken.Leng), tokenType);
                }
            }
        }

    private:
        void AddTokenInfo(TString token, NTokenizer::ETokenType tokenType) {
            Tokens->emplace_back(std::move(token));
            if (TokenTypes) {
                TokenTypes->emplace_back(tokenType);
            }
        }

        TVector<TString>* Tokens;
        TVector<NTokenizer::ETokenType>* TokenTypes;
        NTokenizer::TTokenizerOptions Options;
        ILemmerImplementation* Lemmer;
    };

}

static void SplitBySense(
    TStringBuf inputString,
    const NTokenizer::TTokenizerOptions& options,
    ILemmerImplementation* lemmer,
    TVector<TString>* tokens,
    TVector<NTokenizer::ETokenType>* tokenTypes
) {
    TTokenHandler handler(tokens, tokenTypes, options, lemmer);
    TNlpTokenizer tokenizer(handler);
    tokenizer.Tokenize(UTF8ToWide(inputString));
}

template <typename StringType>
static void SplitByDelimiter(
    TStringBuf inputString,
    const TString& delimiter,
    bool splitBySet,
    bool skipEmpty,
    TVector<StringType>* tokens
) {
    if (splitBySet) {
        if (skipEmpty) {
            *tokens = StringSplitter(inputString).SplitBySet(delimiter.c_str()).SkipEmpty();
        } else {
            *tokens = StringSplitter(inputString).SplitBySet(delimiter.c_str());
        }
    } else {
        if (skipEmpty) {
            *tokens = StringSplitter(inputString).SplitByString(delimiter).SkipEmpty();
        } else {
            *tokens = StringSplitter(inputString).SplitByString(delimiter);
        }
    }
}

template <typename StringType>
static void FilterNumbers(TVector<StringType>* tokens) {
    TVector<StringType> filteredTokens;
    filteredTokens.reserve(tokens->size());
    for (auto& token : *tokens) {
        if (!IsNumber(token)) {
            filteredTokens.emplace_back(token);
        }
    }
    tokens->swap(filteredTokens);
}

static void SplitByDelimiter(
    TStringBuf inputString,
    const NTokenizer::TTokenizerOptions& options,
    ILemmerImplementation* lemmer,
    TVector<TString>* tokens,
    TVector<NTokenizer::ETokenType>* tokenTypes
) {
    SplitByDelimiter(inputString, options.Delimiter, options.SplitBySet, options.SkipEmpty, tokens);

    if (IsWordChanged(options)) {
        for (auto& token : *tokens) {
            TUtf16String wideToken = UTF8ToWide(token);
            ProcessWordToken(options, lemmer, &wideToken);
            token = WideToUTF8(wideToken);
        }
    }

    if (options.NumberProcessPolicy == NTokenizer::ETokenProcessPolicy::Replace) {
        for (auto& token : *tokens) {
            if (IsNumber(token)) {
                token = options.NumberToken;
            }
        }
    } else if (options.NumberProcessPolicy == NTokenizer::ETokenProcessPolicy::Skip) {
        FilterNumbers(tokens);
    }

    if (tokenTypes) {
        tokenTypes->resize(tokens->size());
        Fill(tokenTypes->begin(), tokenTypes->end(), NTokenizer::ETokenType::Unknown);
    }
}

NTokenizer::TTokenizer::TTokenizer()
    : Lemmer(TLemmerImplementationFactory::Construct(EImplementationType::Trivial, {}))
{
}

void NTokenizer::TTokenizer::Initialize() {
    if (TLemmerImplementationFactory::Has(EImplementationType::YandexSpecific)) {
        Lemmer.Reset(TLemmerImplementationFactory::Construct(EImplementationType::YandexSpecific, Options.Languages));
    } else {
        Y_ENSURE(TLemmerImplementationFactory::Has(EImplementationType::Trivial),
            "Lemmer implementation factory should have open source implementation.");
        Y_ENSURE(!Options.Lemmatizing, "Lemmer isn't implemented yet.");
        Lemmer.Reset(TLemmerImplementationFactory::Construct(EImplementationType::Trivial, {}));
    }

    if (Options.LemmerCacheSize != 0) {
        Lemmer.Reset(new TLemmerWithCache(std::move(Lemmer), Options.LemmerCacheSize));
    }

    NeedToModifyTokensFlag |= Options.SeparatorType == NTokenizer::ESeparatorType::BySense;
    NeedToModifyTokensFlag |= IsWordChanged(Options);
    NeedToModifyTokensFlag |= Options.NumberProcessPolicy == ETokenProcessPolicy::Replace;
}

NTokenizer::TTokenizer::TTokenizer(const NTokenizer::TTokenizerOptions& options)
    : Options(options)
{
    Initialize();
}

void NTokenizer::TTokenizer::Tokenize(
    TStringBuf inputString,
    TVector<TString>* tokens,
    TVector<NTokenizer::ETokenType>* tokenTypes
) const {
    Y_ASSERT(tokens);
    tokens->clear();
    if (tokenTypes) {
        tokenTypes->clear();
    }

    if (Options.SeparatorType == NTokenizer::ESeparatorType::BySense) {
        SplitBySense(inputString, Options, Lemmer.Get(), tokens, tokenTypes);
    } else {
        Y_ENSURE(Options.SeparatorType == NTokenizer::ESeparatorType::ByDelimiter, "Unsupported SeparatorType");
        SplitByDelimiter(inputString, Options, Lemmer.Get(), tokens, tokenTypes);
    }
}

TVector<TString> NTokenizer::TTokenizer::Tokenize(TStringBuf inputString) const {
    TVector<TString> tokens;
    Tokenize(inputString, &tokens);
    return tokens;
}

void NTokenizer::TTokenizer::TokenizeWithoutCopy(TStringBuf inputString, TVector<TStringBuf>* tokens) const {
    Y_ASSERT(!NeedToModifyTokensFlag);
    SplitByDelimiter(inputString, Options.Delimiter, Options.SplitBySet, Options.SkipEmpty, tokens);
    if (Options.NumberProcessPolicy == ETokenProcessPolicy::Skip) {
        FilterNumbers(tokens);
    }
}

TVector<TStringBuf> NTokenizer::TTokenizer::TokenizeWithoutCopy(TStringBuf inputString) const {
    TVector<TStringBuf> tokens;
    TokenizeWithoutCopy(inputString, &tokens);
    return tokens;
}

NTokenizer::TTokenizerOptions NTokenizer::TTokenizer::GetOptions() const {
    return Options;
}

bool NTokenizer::TTokenizer::NeedToModifyTokens() const {
    return NeedToModifyTokensFlag;
}

void NTokenizer::TTokenizer::Save(IOutputStream *stream) const {
    Options.Save(stream);
}

void NTokenizer::TTokenizer::Load(IInputStream *stream) {
    Options.Load(stream);
    Initialize();
}

bool NTokenizer::TTokenizer::operator==(const TTokenizer& rhs) const {
    return Options == rhs.Options;
}

bool NTokenizer::TTokenizer::operator!=(const TTokenizer& rhs) const {
    return !(*this == rhs);
}
