#pragma once

#include "nlptypes.h"
#include "token_structure.h"

//! merges subtokens of a multitoken to floats and marks to provide backward compatibility for new tokenization of marks
//! @note there are exclusions that not compatible with the old tokenization:
//!       [v1.0] -> [v 1.0] instead of old [v1 0]
//!       [a+b]  -> [a + b] instead of old [a+ b]
class TTokenIterator {
    const TWideToken& Tok;
    const TTokenStructure& Subtokens;
    TTokenStructure Tokens;
    NLP_TYPE NlpType;
    size_t First;
    const size_t Last;

private:
    static bool BreakMultitoken(const TTokenStructure& subtokens, size_t first, size_t last, size_t i, size_t n) {
        Y_ASSERT(i >= first && i <= last);
        if (i == last)
            return true;

        const TCharSpan& s = subtokens[i];

        if (s.SuffixLen != 0 || subtokens[i + 1].PrefixLen != 0)
            return true; // no prefix/suffix in the middle

        if (s.TokenDelim == TOKDELIM_NULL) {
            if (i < (last - 1) && subtokens[i + 1].Type == TOKEN_NUMBER && subtokens[i + 1].TokenDelim == TOKDELIM_DOT && subtokens[i + 2].Type == TOKEN_NUMBER)
                return true; // v1.0 -> v /+1 1.0

            if (i == first || s.Type != TOKEN_NUMBER || subtokens[i - 1].TokenDelim != TOKDELIM_DOT || subtokens[i - 1].Type != TOKEN_NUMBER)
                return false; // if the current token '2': 1-2a then the current token is a part of a mark

            return true; // 1.2a -> 1.2 a
        }

        if (s.Type == TOKEN_NUMBER) {
            if (n == 2) {
                Y_ASSERT(i > first && (i + 1) == (first + n));
                if (subtokens[i - 1].TokenDelim == TOKDELIM_DOT) // && subtokens[i + 1].Type == TOKEN_NUMBER)
                    return true;                                 // it is FLOAT
            }

            if (s.TokenDelim == TOKDELIM_DOT && subtokens[i + 1].Type == TOKEN_NUMBER)
                return false; // the current token is a part of a float

            return true; // the current token is number
        }

        // the current token is word

        if (s.TokenDelim != TOKDELIM_APOSTROPHE && s.TokenDelim != TOKDELIM_MINUS)
            return true; // baden-baden, caffrey's

        // delimiter is '-' or '\''

        if (s.Type != subtokens[i + 1].Type)
            return true; // types of tokens are different

        if (i > first && subtokens[i - 1].TokenDelim == TOKDELIM_NULL)
            return true; // the current token 'a' and the previous token '1' has no delimiter: 1a-b

        return (i < (last - 1) && subtokens[i + 1].TokenDelim == TOKDELIM_NULL); // mark follows the current token 'a': a-b2
    }

public:
    explicit TTokenIterator(const TWideToken& tok)
        : Tok(tok)
        , Subtokens(tok.SubTokens)
        , NlpType(NLP_END)
        , First(0)
        , Last(tok.SubTokens.size() - 1)
    {
        Y_ASSERT(tok.SubTokens.size());
    }
    //! returns true if one more multitoken is found
    bool Next() {
        if (Finished())
            return false;

        Tokens.clear();
        size_t i = First;
        do {
            const TCharSpan& s = Subtokens[i];
            if (!Tokens.empty() && Tokens.back().TokenDelim == TOKDELIM_NULL) {
                TCharSpan& mark = Tokens.back();
                mark.Len += s.Len;
                mark.SuffixLen = s.SuffixLen;
                mark.Type = TOKEN_MARK; // change type
                NlpType = NLP_MARK;
            } else {
                Y_ASSERT(Tokens.empty() || Tokens.back().Type == s.Type);
                Tokens.push_back(s);
                NlpType = (s.Type == TOKEN_WORD ? NLP_WORD : NLP_INTEGER);
            }
        } while (!BreakMultitoken(Subtokens, First, Last, i++, Tokens.size()));
        Y_ASSERT(!Tokens.empty());

        if (NlpType == NLP_INTEGER && Tokens.size() == 2) {
            Y_ASSERT(Tokens[0].SuffixLen == 0 && Tokens[0].TokenDelim == TOKDELIM_DOT); // && Tokens[1].SuffixLen == 0);
            NlpType = NLP_FLOAT;
            TCharSpan& first = Tokens[0];
            const TCharSpan& second = Tokens[1];
            first.Len = second.EndPos() - first.Pos;
            first.SuffixLen = second.SuffixLen;
            first.Type = TOKEN_FLOAT;
            first.TokenDelim = TOKDELIM_NULL;
            Tokens.resize(1);
        }

        Tokens.back().TokenDelim = TOKDELIM_NULL; // reset the last delimiter
        First = i;
        return true;
    }
    //! @note positions of subtokens of the original multitoken are not changed;
    //!       all tokens can have suffixes
    const TTokenStructure& Get() const {
        return Tokens;
    }
    bool Finished() const {
        return First > Last;
    }
    //! the first subtoken of multitoken has position equal to 0
    //! @note only word tokens can have suffixes
    void GetMultitoken(TWideToken& tok) const {
        Y_ASSERT(!Tokens.empty());
        tok.SubTokens = Tokens;
        TTokenStructure& subtokens = tok.SubTokens;
        const TCharSpan& first = subtokens[0];
        TCharSpan& last = subtokens.back();
        tok.Token = Tok.Token + first.Pos;
        if (last.Type == TOKEN_WORD) {
            tok.Leng = last.EndPos() + last.SuffixLen - first.Pos;
            if (!Finished() && Subtokens[First].PrefixLen) {
                const ui16 suffixLen = GetAdditionalSuffixLen();
                tok.Leng += suffixLen;
                last.SuffixLen += suffixLen;
            }
        } else {
            tok.Leng = last.EndPos() - first.Pos;
            last.SuffixLen = 0;
            if (NlpType == NLP_INTEGER && !Finished() && Subtokens[First].PrefixLen) {
                const ui16 suffixLen = GetIntegerSuffixLen();
                tok.Leng += suffixLen;
                last.SuffixLen = suffixLen;
            }
        }
        subtokens[0].PrefixLen = 0;
        const size_t diff = first.Pos;
        for (auto& subtoken : subtokens)
            subtoken.Pos -= diff;
    }
    ui16 GetAdditionalSuffixLen() const {
        const TCharSpan& origtok = Subtokens[First - 1];
        Y_ASSERT(origtok.Type == TOKEN_WORD && !Finished() && Subtokens[First].PrefixLen);
        ui16 suffixLen = 0;
        if (origtok.TokenDelim == TOKDELIM_PLUS)
            suffixLen = 1;
        return suffixLen;
    }
    ui16 GetIntegerSuffixLen() const {
        Y_ASSERT(NlpType == NLP_INTEGER && !Finished() && Subtokens[First].PrefixLen && Tokens.size() == 1);
        const TCharSpan& origtok = Subtokens[First - 1];
        ui16 suffixLen = 0;
        if (origtok.TokenDelim == TOKDELIM_PLUS) {
            suffixLen = origtok.SuffixLen;
            if (origtok.SuffixLen < 2)
                suffixLen += 1;
        }
        return suffixLen;
    }
    //! returns NLP type of multitoken returned by GetMultitoken(tok)
    NLP_TYPE GetNlpType() const {
        return NlpType;
    }
    //! called for the first prefix, other prefixes returned as delimiters by GetDelimiter()
    void GetPrefix(TWideToken& tok) const {
        Y_ASSERT(Tokens.empty()); // Next() must NOT be called
        if (Subtokens.empty() || Subtokens[0].PrefixLen == 0) {
            tok.Leng = 0;
            tok.SubTokens.clear();
        } else {
            tok.Token = Tok.Token;
            tok.Leng = Subtokens[0].PrefixLen;
            tok.SubTokens.clear();
        }
    }
    //! @note NLP type of token is NLP_MISCTEXT;
    //!       prefixes always considered as "misctext";
    //!       suffixes of non-words considered as "misctext";
    //!       this function can be called after the last token as well,
    //!       especially when the last non-word token has the suffix
    void GetDelimiter(TWideToken& tok) const {
        Y_ASSERT(!Tokens.empty()); // Next() must be called
        //Y_ASSERT(!Finished());
        const TCharSpan& prev = Tokens.back();
        size_t endpos = prev.EndPos();
        if (prev.Type == TOKEN_WORD) {
            endpos += prev.SuffixLen;
            if (!Finished() && Subtokens[First].PrefixLen)
                endpos += GetAdditionalSuffixLen();
        } else if (NlpType == NLP_INTEGER && !Finished() && Subtokens[First].PrefixLen)
            endpos += GetIntegerSuffixLen();
        tok.Token = Tok.Token + endpos;
        tok.Leng = (Finished() ? Tok.Leng : Tok.SubTokens[First].Pos) - endpos; // length can be equal to 0 in case v1.0 -> v 1.0
        tok.SubTokens.clear();
    }
};
