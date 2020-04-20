#include "sentbreakfilter.h"

#include <util/generic/singleton.h>
#include <util/memory/pool.h>

namespace {
    inline bool IsEnglishCapitalLetter(wchar16 c) {
        return (c >= 'A' && c <= 'Z');
    }

    inline bool AreAllDelims(const TWideToken& tok, ETokenDelim delim) {
        const TTokenStructure& subtokens = tok.SubTokens;

        if (subtokens.size() == 1)
            return false; // it isn't multitoken

        const size_t last = subtokens.size() - 1;
        for (size_t i = 0; i < last; ++i) {
            const TCharSpan& s = subtokens[i];
            if (s.TokenDelim != delim)
                return false;
        }

        return true;
    }

}

TSentBreakFilter::TSentBreakFilter(TLangMask langMask)
    : LastType(NLP_END)
    , SentLen(0)
    , Abbreviations(Singleton<TAbbreviationsDictionary>())
    , LangMask(langMask)
{
    LastToken.Token = Buffer;
}

bool TSentBreakFilter::IsAbbrevation(const TWtringBuf& text) {
    const TTokenStructure& subtokens = LastToken.SubTokens;
    Y_ASSERT(Abbreviations);

    wchar16* lastStart = Buffer + subtokens.back().Pos;
    size_t lastLen = subtokens.back().Len;
    ToLower(lastStart, lastLen);
    TWtringBuf lastSubtoken(lastStart, lastLen);

    // it can never be a sentence break
    // @todo sentence break should be in the following cases:
    //       ... dB/km. Next sentence.
    //       ... Gbyte/min. Next sentence.
    //       ... etc.). Next sentence.
    if (Abbreviations->NeverBreak(lastSubtoken, LangMask)) {
        return true;
    }

    // if it is followed by a digit - not a sentence break
    if (IsDigit(text.back()) && Abbreviations->DontBreakIfBeforeDigit(lastSubtoken, LangMask)) {
        return true;
    }

    // test for abbreviations that consist of two subtokens
    if (subtokens.size() > 1) {
        wchar16* lastTwoStart = Buffer + subtokens[subtokens.size() - 2].Pos;
        size_t lastTwoLen = subtokens.back().Pos +
                            subtokens.back().Len -
                            subtokens[subtokens.size() - 2].Pos;
        ToLower(lastTwoStart, lastTwoLen);
        TWtringBuf lastTwoSubtokens(lastTwoStart, lastTwoLen);

        if (Abbreviations->DoubleSubtokenNeverBreak(lastTwoSubtokens, LangMask)) {
            return true;
        }

        if (IsDigit(text.back()) && Abbreviations->DoubleSubtokenDontBreakIfBeforeDigit(lastTwoSubtokens, LangMask)) {
            return true;
        }
    }

    return false;
}

NLP_TYPE TSentBreakFilter::OnSentBreak(const wchar16* text, size_t len) {
    Y_ASSERT(text && len);

    // @todo such cases could be processed correctly:
    //       ... NEC Corp.). Next sentence.
    //       The sentence should be broken because 'Corp.' followed by ').'

    // although '.' in termpunct [!.:?] has a lot of values (yc_2E = 0x002E | 0x037E | 0x0387 | 0x0589 | ..., see charclasses_16.rl)
    // only '.' is used to abbreviate words in russian, english and other european languages
    if (*text == '.') {
        if (LastType == NLP_INTEGER && SentLen == 1) {
            // don't break sent. in case of numbered lists (html tag <OL> - ordered list), for ex. 1. 02. 10. [17]. 22). 125. etc.
            // actually numbers less than 10 would be processed by the next rule - (LastToken.Leng == 1)
            return NLP_MISCTEXT;
            // multilevel numbered lists are not processed, for ex. 1.2. 2.1 10.1.2. etc.
            // @todo probably sent. should not be broken if it contains integer numbers only: "10. 11. 12. 13."
            //       now it is broken after "11.": "10. 11.\n12. 13."
        } else if (LastToken.Leng == 1) { // John F. Kennedy, A. Pushkin; @todo (LastType != NLP_INTEGER && LastToken.Leng == 1) ?
            // ascii emoji parts "hello :)." are processed as 1-character words but should not stop sentence breaks
            //if (!IsDigit(*LastToken.Token)) // what about processing numbered lists?
            return (*LastToken.Token >= 0x80 || IsAlnum(*LastToken.Token)) ? NLP_MISCTEXT : NLP_SENTBREAK;
            // @todo sentence break should be in the following cases:
            //       ... km/s. Next sentence.
            //       ... pict. 1. Next sentence.
            //       ... under OS/2. Next sentence. (at the same time numbered lists should not be a sentence: 1. It is the first item.)
        }

        const TTokenStructure& subtokens = LastToken.SubTokens;
        if (!subtokens.empty()) {
            Y_ASSERT(LastToken.Token == Buffer);
            const TCharSpan& lastSubtokenSpan = subtokens.back();
            if (lastSubtokenSpan.Len == 1) {
                if (LastType != NLP_INTEGER &&
                    lastSubtokenSpan.Type == TOKEN_NUMBER &&
                    AreAllDelims(LastToken, TOKDELIM_DOT))
                {
                    return NLP_MISCTEXT; // Pict.1. Diagram.
                }

                if (LastType == NLP_WORD && AreAllDelims(LastToken, TOKDELIM_DOT)) // && all tokens len == 1)
                {
                    return NLP_MISCTEXT; // P.S.
                }

                if (LastType == NLP_WORD && subtokens.size() == 2) {
                    const TCharSpan& s0 = subtokens[0];
                    const TCharSpan& s1 = subtokens[1];
                    //                    if (s0.Len == 1 && s1.Len == 1 && s0.TokenDelim == TOKDELIM_DOT)
                    //                        return NLP_MISCTEXT;

                    // the following initials can be in English: H-P. Kriegel, J-C. Freytag
                    //const TCharSpan& s0 = subtokens[0];
                    //const TCharSpan& s1 = subtokens[1];
                    if (s0.Len == 1 && s1.Len == 1 &&
                        IsEnglishCapitalLetter(LastToken.Token[s0.Pos]) &&
                        IsEnglishCapitalLetter(LastToken.Token[s1.Pos]) &&
                        IsEnglishCapitalLetter(text[len - 1]))
                    {
                        return NLP_MISCTEXT;
                    }
                }
            }

            if (IsAbbrevation(TWtringBuf(text, len))) {
                return NLP_MISCTEXT;
            }
        }
    }

    return NLP_SENTBREAK;
}
