#pragma once

#include <library/cpp/enumbitset/enumbitset.h>
#include <library/cpp/langmask/langmask.h>
#include <library/cpp/token/nlptypes.h>

#include <util/generic/bitmap.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

struct TTokenizerSplitParams {
public:
    typedef TEnumBitSet<NLP_TYPE, NLP_END, NLP_MISCTEXT + 1> THandledMask;
    static const THandledMask WORDS;
    static const THandledMask NOT_PUNCT;

public:
    TTokenizerSplitParams(){}

    TTokenizerSplitParams(const THandledMask& mask)
        : HandledMask(mask){}

public:
    /// Token types to handle, not used in SplitIntoSentences
    THandledMask HandledMask = WORDS;

    /// Tokenizer params, see tokenizer.h for detailed explanation
    bool BackwardCompatibility = true;
    bool SpacePreserve = false;
    TLangMask TokenizerLangMask;
    bool UrlDecode = true;
};

TVector<TUtf16String> SplitIntoTokens(const TUtf16String& text, const TTokenizerSplitParams& params = TTokenizerSplitParams());
TVector<TUtf16String> SplitIntoSentences(const TUtf16String& text, const TTokenizerSplitParams& params = TTokenizerSplitParams());
