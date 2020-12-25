#ifndef CATBOOST_OPENSOURCE
#include <library/cpp/charset/wide.h>
#endif

#include <util/charset/wide.h>
#include <util/memory/tempbuf.h>

#include "sentbreakfilter.h"
#include "nlpparser.h"
#include "tokenizer.h"

#include <util/stream/file.h>

void TNlpTokenizer::Tokenize(const wchar16* str,
                             size_t size,
                             const TTokenizerOptions& opts) {
    bool semicolonBreaksSentence = opts.LangMask == TLangMask(LANG_GRE);
    TSentBreakFilter sentBreakFilter(opts.LangMask);
    THolder<TNlpParser> parser;
    switch (opts.Version) {
        case 2:
            parser = MakeHolder<TVersionedNlpParser<2>>(TokenHandler, sentBreakFilter, Buffer, opts.SpacePreserve,
                    BackwardCompatible, semicolonBreaksSentence, opts.UrlDecode);
            break;
        case 3:
            parser = MakeHolder<TVersionedNlpParser<3>>(TokenHandler, sentBreakFilter, Buffer, opts.SpacePreserve,
                    BackwardCompatible, semicolonBreaksSentence, opts.UrlDecode, opts.KeepAffixes);
            break;
        default:
            parser = MakeHolder<TDefaultNlpParser>(TokenHandler, sentBreakFilter, Buffer, opts.SpacePreserve,
                    BackwardCompatible, semicolonBreaksSentence, opts.UrlDecode);
            break;
    }
    try {
        parser->Execute(str, size, &TextStart);
    } catch (const ITokenHandler::TAllDoneException&) {
        // do nothing
    }
}

#ifndef CATBOOST_OPENSOURCE
void TNlpTokenizer::Tokenize(const char* text,
                             size_t len,
                             bool spacePreserve,
                             TLangMask langMask) {
    TCharTemp buf(len);
    wchar16* const data = buf.Data();
    CharToWide(text, len, data, csYandex);
    TTokenizerOptions opts {spacePreserve, langMask, /*decodeUrl=*/true};
    Tokenize(data, len, opts);
}
#endif

void TNlpTokenizer::Tokenize(const wchar16* str,
                             size_t size,
                             bool spacePreserve,
                             TLangMask langMask) {
    TTokenizerOptions opts {spacePreserve, langMask, /*decodeUrl=*/true};
    Tokenize(str, size, opts);
}

bool IsSpecialTokenizerSymbol(const TWtringBuf s) {
    if (s.size() != 1) {
        return false;
    }
    // Only base-plane codepoints can be special tokenizer symbols,
    // and they can be just casted to wchar32.
    // Unicode conversion will be needed to process surrogate pairs.
    return IsSpecialTokenizerSymbol(static_cast<wchar32>(s[0]));
}

bool IsAsciiEmojiPart(const TWtringBuf s) {
    // no worries for surrogates here because of Ascii
    for (auto c : s)
        if (!IsAsciiEmojiPart(c))
            return false;
    return true;
}
