#pragma once

#include <library/cpp/token/nlptypes.h>
#include <library/cpp/token/token_structure.h>
#include "multitokenutil.h"

// NOTE: this class inherited by request tokenizers only
class TMultitokenParser {
    TWideToken Multitoken; //!< length of the whole token can include suffixes, if difference between pos and endpos of two tokens is greater than 1 then there is suffix
    TCharSpan CurCharSpan;
    wchar16 PrefixChar;
    wchar16 SuffixChar;

protected:
    TMultitokenParser()
        : PrefixChar(0)
        , SuffixChar(0)
    {
    }

    //! used for deferring tokens in TReqTokenizer
    void SetMultitoken(const TWideToken& tok) {
        Multitoken = tok;
    }

    const TWideToken& GetMultitoken() const {
        return Multitoken;
    }

    //! @param token    start of the multitoken
    //! @param len      length includes suffix: 'ab-c+' length is equal to 5, in case of prefix operator: '-!abc' length is equal to 3
    //!                 and prefix: '#hashtag', '@user_name'
    void SetMultitoken(const wchar16* token, size_t len) {
        TTokenStructure& subtokens = Multitoken.SubTokens;
        if (!subtokens.empty()) {
            // positions of the first subtoken can be non-zero in the request parser
            // but 'token' in this case must point to the first character of the first subtoken
            const size_t pos = subtokens[0].Pos - subtokens[0].PrefixLen;
            Y_ASSERT((subtokens.back().EndPos() - pos + subtokens.back().SuffixLen) == len);
            if (pos) {
                const size_t n = subtokens.size();
                for (size_t i = 0; i < n; ++i)
                    subtokens[i].Pos -= pos;
            }
        }
        Multitoken.Token = token;
        Multitoken.Leng = len;

        Y_ASSERT(CheckMultitoken(Multitoken));
    }

    const TTokenStructure& GetSubtokens() const {
        return Multitoken.SubTokens;
    }

    void BeginToken(const wchar16* tokstart, const wchar16* p) {
        // it can be called twice in case "exa&shy;&#x301;mple", see nlptok.rl, mixedtoken, tokfirst/toknext, numfirst/numnext
        if (CurCharSpan.Len == 0)
            CurCharSpan.Pos = p - tokstart;
    }

    void BeginToken(const wchar16* tokstart, const wchar16* p, ETokenType type) {
        BeginToken(tokstart, p);
        CurCharSpan.Type = type;
    }

    void UpdateToken() {
        CurCharSpan.Len += 1;
    }

    void AddToken() {
        Y_ASSERT(CurCharSpan.Len);
        Y_ASSERT(CurCharSpan.Type == TOKEN_WORD || CurCharSpan.Type == TOKEN_NUMBER);

        TTokenStructure& tokens = Multitoken.SubTokens;

        // @todo if number of tokens is greater than 64 then the last token can consist of numbers, letters and delimiters...
        tokens.push_back(CurCharSpan);

        const size_t n = tokens.size();
        if (n > 1)
            CorrectDelimiters(tokens[n - 2], SuffixChar, tokens[n - 1], PrefixChar);

        CurCharSpan.Pos = 0;
        CurCharSpan.Len = 0; // it is checked in AddLastToken()
        CurCharSpan.PrefixLen = 0;
        PrefixChar = 0;
        SuffixChar = 0;
    }

    void AddIdeograph(size_t len) {
        Y_ASSERT(!CurCharSpan.Len && (len == 1 || len == 2));
        TTokenStructure& tokens = Multitoken.SubTokens;
        Y_ASSERT(tokens.empty());
        tokens.push_back(0, len, TOKEN_WORD);
    }

    void AddLastToken(const wchar16* tokstart, const wchar16* tokend) {
        // - CurCharSpan.Len assigned to 0 in AddToken() because in case of multitoken with '.' at the end, for
        //   example: " well-formed. " parser already called to %add_token because '.' can be delimiter of the next token
        if (CurCharSpan.Len) {
            const wchar16* const actualStart = tokstart + CurCharSpan.Pos;
            // for ex. "5% " can have (actualStart == tokend) because '%' could be part of the next token with utf8 characters
            if (actualStart < tokend) {
                const size_t actualLen = tokend - actualStart;
                if (CurCharSpan.Len != actualLen) // for example "WORD% NEXTWORD" - '%' could be part of UTF8 encoded character and already counted...
                    CurCharSpan.Len = actualLen;
                AddToken();
            } else
                CancelToken();
        } else
            CancelToken();

        TTokenStructure& tokens = Multitoken.SubTokens;
        if (!tokens.empty())
            tokens.back().TokenDelim = TOKDELIM_NULL; // reset delimiter if any
    }

    //! correct the last token if it contains words and numbers and changes length of multitoken
    //! @param len      length of multitoken (including all subtokens), for ex. (te - ts)
    //! @return true if the last token is valid, false - last token is cut off and length is changed
    //! @note in case of '+!abc-...-xyz' length includes '+!', this function doesn't take into account offset of the first subtoken
    //!       if number of subtokens equal to 63 all superfluous subtokens are put into the last subtoken TOKEN_MIXED
    //!       which is cut off in this function
    bool CheckLastToken(size_t& len) {
        TTokenStructure& tokens = Multitoken.SubTokens;
        if (tokens.size() == MAX_SUBTOKENS && tokens.back().Type == TOKEN_MIXED) {
            tokens.pop_back();
            // change delimiter (+) to suffix
            TCharSpan& lasttok = tokens.back();
            len = lasttok.EndPos() + lasttok.SuffixLen;
            // actually '+' should be added if subtoken has suffix '+' because '++' is valid suffix as well
            if (lasttok.TokenDelim == TOKDELIM_PLUS && lasttok.SuffixLen == 0) {
                lasttok.SuffixLen = 1;
                len += 1;
            }
            lasttok.TokenDelim = TOKDELIM_NULL;
            return false;
        }
        Y_ASSERT(tokens.empty() || tokens.back().TokenDelim == TOKDELIM_NULL);
        return true;
    }

    //! @return result of CheckLastToken()
    //! @note positions of the first subtoken can be non-zero in case: +abc, -!xyz,
    //!       tokstart in this cases can point to + and - respectively,
    //!       SetMultitoken() resets position of the first subtoken to 0
    //! @param pos      old position value of the first subtoken before resetting it to 0
    //! @param len      new len of multitoken after cutting off the last subtoken if it is invalid
    //! @note in case of '+!abc-efg' returned 'len' includes the prefix operators '+!' and is equal to 9
    bool SetRequestMultitoken(const wchar16* tokstart, const wchar16* tokend, size_t& len) {
        AddLastToken(tokstart, tokend);
        const TTokenStructure& subtokens = GetSubtokens();
        Y_ASSERT(!subtokens.empty());
        const TCharSpan& firsttok = subtokens[0];
        const TCharSpan& lasttok = subtokens.back();
        len = lasttok.EndPos() + lasttok.SuffixLen; // can't use (te - ts) because postfix can be there
        const bool res = CheckLastToken(len);
        const size_t firsttokStart = firsttok.Pos - firsttok.PrefixLen;
        SetMultitoken(tokstart + firsttokStart, len - firsttokStart);
        return res;
    }

    void UpdatePrefix(wchar16 c) {
        Y_ASSERT(c == '#' || c == '@' || c == '$');
        CurCharSpan.PrefixLen = 1; // length of prefix can't be more than 1
        PrefixChar = c;
    }

    void UpdateSuffix(wchar16 c) {
        Y_ASSERT(c == '#' || c == '+');
        TTokenStructure& tokens = Multitoken.SubTokens;
        if (!tokens.empty()) {
            tokens.back().SuffixLen += 1;
            SuffixChar = c;
        } else
            Y_ASSERT(!"can't update suffix: no subtokens");
    }

    void CancelToken() {
        // example: "abc 5% def", '%' can be the first symbol of utf8 encoded character so token is started by call to BeginToken()
        // and then UpdateToken() is called as well but there is no call to AddToken() because '%' is interpreted as a misc character so
        // CurCharSpan.Len must be reset
        CurCharSpan.Len = 0;
        CurCharSpan.PrefixLen = 0;
        PrefixChar = 0;
        SuffixChar = 0;
    }

    void ClearSubtokens() {
        Multitoken.SubTokens.clear();
    }

    //! @param delim    type of delimiter
    //! @param c        delimiter unicode character
    void SetTokenDelim(ETokenDelim delim, wchar16 /*c*/) {
        Y_ASSERT(!Multitoken.SubTokens.empty());
        Multitoken.SubTokens.back().TokenDelim = delim;
        // @todo remove this condition because unicode delimiters are removed before lemmatization
        //        if (c >= 0x7F) // if it is non-ASCII character
        //            SimpleMultitoken = false;
    }
};
