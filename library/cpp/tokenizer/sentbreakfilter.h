#pragma once

#include "tokenizer.h"

#include <library/cpp/token/nlptypes.h>
#include <library/cpp/langmask/langmask.h>
#include <library/cpp/token/token_structure.h>

#include <util/generic/hash_set.h>
#include <util/generic/noncopyable.h>
#include <util/system/maxlen.h>
#include <util/system/yassert.h>

constexpr int TOKEN_MAX_LEN = 255;
constexpr int TOKEN_MAX_BUF = 256;

/// Dictionary with abbreviations that can prevent sentence from breaking.
/** For each language there are two hash sets, first with abbreviations
  * that can never appear at the end of sentence ant second with
  * abbreviations that doesn't break sentence if there is a digit afterwards.
  * The dictionaries are initialized from static arrays in the constructor (see
  * abbreviations.cpp).
  * @note The LANG_UNK language code is used for common abbreviations; they are
  * checked for any input language mask.
  */
class TAbbreviationsDictionary: private TNonCopyable {
private:
    THashSet<TUtf16String> NeverBreakSets[LANG_MAX];
    THashSet<TUtf16String> DontBreakIfBeforeDigitSets[LANG_MAX];
    THashSet<TUtf16String> DoubleSubtokenNeverBreakSets[LANG_MAX];
    THashSet<TUtf16String> DoubleSubtokenDontBreakIfBeforeDigitSets[LANG_MAX];

    void AddElements(THashSet<TUtf16String>& hashSet,
                     const char* elements[],
                     size_t size);

    bool FindInHashSets(const THashSet<TUtf16String>* hashSets,
                        const TWtringBuf& string,
                        TLangMask langMask) const {
        if (hashSets[LANG_UNK].find(string) != hashSets[LANG_UNK].end()) {
            return true;
        }

        for (ELanguage lang : langMask) {
            if (hashSets[lang].find(string) != hashSets[lang].end()) {
                return true;
            }
        }

        return false;
    }

public:
    TAbbreviationsDictionary();

    /// Functions for testing if the string is an abbreviation
    /** @note The functions return true if the string is an abbreviation in
      * any language from the langMask.
      * @note The common abbreviations are checked on any call, even if the
      * langMask is empty.
      */

    /// Test if the string cannot appear at the end of the sentence
    bool NeverBreak(const TWtringBuf& string, TLangMask langMask) const {
        return FindInHashSets(NeverBreakSets, string, langMask);
    }

    /// Test if the string cannot break sentence before digit
    bool DontBreakIfBeforeDigit(const TWtringBuf& string,
                                TLangMask langMask) const {
        return FindInHashSets(DontBreakIfBeforeDigitSets, string, langMask);
    }

    /// Test if the double-subtoken string cannot appear at the end of the sentence
    bool DoubleSubtokenNeverBreak(const TWtringBuf& string,
                                  TLangMask langMask) const {
        return FindInHashSets(DoubleSubtokenNeverBreakSets, string, langMask);
    }

    /// Test if the double-subtoken string cannot break sentence before digit
    bool DoubleSubtokenDontBreakIfBeforeDigit(const TWtringBuf& string,
                                              TLangMask langMask) const {
        return FindInHashSets(DoubleSubtokenDontBreakIfBeforeDigitSets,
                              string,
                              langMask);
    }
};

//! @todo implement a functionality never generating sentence breaks (for document titles for ex.);
//!       probably controlling of sentence length (<=WORD_LEVEL_Max) should implement in this class.
class TSentBreakFilter: private TNonCopyable {
private:
    TWideToken LastToken;
    NLP_TYPE LastType;
    wchar16 Buffer[TOKEN_MAX_BUF]; //!< text in this buffer is not null-terminated
    size_t SentLen;           //!< counts only whole tokens (not subtokens): 1.1 and S.F - single tokens

    const TAbbreviationsDictionary* const Abbreviations;

    TLangMask LangMask;

private:
    bool IsAbbrevation(const TWtringBuf& text);

public:
    TSentBreakFilter(TLangMask langMask);

    //! @attention if token is WORD, MARK, etc. then its length must not be greater than TOKEN_MAX_LEN, see TNlpParser::MakeEntry()
    void OnToken(const TWideToken& token, NLP_TYPE type) {
        if (type == NLP_WORD || type == NLP_INTEGER || type == NLP_FLOAT || type == NLP_MARK) {
            Y_ASSERT(token.Leng <= TOKEN_MAX_LEN);
            LastType = type;
            std::char_traits<wchar16>::copy(Buffer, token.Token, token.Leng);
            //LastToken.Token = Buffer; assigned in the constructor
            LastToken.Leng = token.Leng;
            LastToken.SubTokens = token.SubTokens;
            ++SentLen; // += token.SubTokens.size(); ?
        } else if (type == NLP_SENTBREAK || type == NLP_PARABREAK) {
            SentLen = 0;
        } else
            Y_ASSERT(type == NLP_MISCTEXT || type == NLP_END);
    }

    //! @note @c len includes the first letter (or digit) of the next sentence
    //! @attention in case of ambiguity it is preferable not to break sentence because
    //!            it will look like a compound sentence,
    //!            at the same time sentence can have maximum 64 words so sentence must not
    //!            be too long or it is broken after arbitrary word
    NLP_TYPE OnSentBreak(const wchar16* text, size_t len);
};
