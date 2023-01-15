#pragma once

#include <library/cpp/token/nlptypes.h>
#include <library/cpp/token/token_structure.h>
#include "multitokenutil.h"

// replresents single multitoken in the collection of TNlpParserBase
class TParserToken {
    TTokenStructure Subtokens;
    NLP_TYPE NlpType; // type of multitoken
    bool Hyphen;

public:
    TParserToken()
        : NlpType(NLP_WORD)
        , Hyphen(false)
    {
    }

    explicit TParserToken(const TCharSpan& subtok)
        : NlpType(NLP_WORD)
        , Hyphen(false)
    {
        Subtokens.push_back(subtok);
    }
    bool HasHyphen() const {
        return Hyphen;
    }
    void SetHyphen(EHyphenType type) {
        Y_ASSERT(!Subtokens.empty());
        Subtokens.back().Hyphen = type;
        Hyphen = true;
    }
    void SetTokenDelim(ETokenDelim delim) {
        Y_ASSERT(!Subtokens.empty());
        Subtokens.back().TokenDelim = delim;
    }
    NLP_TYPE GetNlpType() const {
        return NlpType;
    }
    size_t GetSubtokenCount() const {
        return Subtokens.size();
    }
    size_t GetStart() const {
        Y_ASSERT(!Subtokens.empty());
        return Subtokens[0].Pos - Subtokens[0].PrefixLen;
    }
    size_t GetEnd() const {
        Y_ASSERT(!Subtokens.empty());
        return Subtokens.back().EndPos() + Subtokens.back().SuffixLen;
    }
    size_t GetLength() const {
        return GetEnd() - GetStart();
    }
    // allowed suffixes: "+", "++" and "#"
    void AddSubtoken(const TCharSpan& span, size_t prefixLen, wchar16 prefixChar, wchar16 suffixChar) {
        Y_ASSERT(Subtokens.size() < MAX_SUBTOKENS);

        if (Subtokens.empty()) {
            NlpType = (span.Type == TOKEN_WORD ? NLP_WORD : NLP_INTEGER);
        } else {
            if (NlpType != NLP_MARK && span.Type != Subtokens.back().Type)
                NlpType = NLP_MARK;
        }

        Subtokens.push_back(span);
        Subtokens.back().PrefixLen = prefixLen;

        const size_t n = Subtokens.size();
        if (n > 1)
            CorrectDelimiters(Subtokens[n - 2], suffixChar, Subtokens[n - 1], prefixChar);

        Y_ASSERT(NlpType == NLP_WORD || NlpType == NLP_MARK || NlpType == NLP_INTEGER);
    }
    void AddIdeograph(size_t len) {
        Y_ASSERT(Subtokens.empty());
        Subtokens.push_back(0, len, TOKEN_WORD);
        NlpType = NLP_WORD;
    }
    void SwapSubtokens(TTokenStructure& other) {
        Subtokens.swap(other);
    }
    // returns length of all subtokens including prefix of the first subtoken and suffix of the last subtoken
    size_t CorrectPositions() {
        Y_ASSERT(!Subtokens.empty());
        // position of the first subtoken can be non-zero in case of non-first subtoken
        // but TWideToken::Token must point to the first character of the first subtoken (prefix included if any)
        const size_t pos = Subtokens[0].Pos - Subtokens[0].PrefixLen;
        if (pos) {
            const size_t n = Subtokens.size();
            for (size_t i = 0; i < n; ++i)
                Subtokens[i].Pos -= pos;
        }
        Y_ASSERT(Subtokens[0].Pos == Subtokens[0].PrefixLen); // PrefixLen can be equal to 0
        return Subtokens.back().EndPos() + Subtokens.back().SuffixLen;
    }
    void UpdateSuffix() {
        if (Subtokens.empty())
            Y_ASSERT(!"can't update suffix: no subtokens");
        else
            Subtokens.back().SuffixLen += 1;
    }
    size_t GetSuffixLength() const {
        return Subtokens.empty() ? 0 : Subtokens.back().SuffixLen;
    }
    void ResetSuffix() {
        if (!Subtokens.empty()) {
            Subtokens.back().SuffixLen = 0;
        }
    }
    void CorrectLastToken(TCharSpan& last) {
        Y_ASSERT(Subtokens.size() == MAX_SUBTOKENS);
        last = Subtokens.back();
        Subtokens.pop_back();
        // change delimiter (+) to suffix
        TCharSpan& lasttok = Subtokens.back();
        // actually '+' should be added if subtoken has suffix '+' because '++' is valid suffix as well
        if (lasttok.TokenDelim == TOKDELIM_PLUS && lasttok.SuffixLen == 0)
            lasttok.SuffixLen = 1;
        else if (lasttok.TokenDelim == TOKDELIM_AT_SIGN && last.PrefixLen == 0)
            last.PrefixLen = 1;
        lasttok.TokenDelim = TOKDELIM_NULL;
    }
    void Reset() {
        Subtokens.clear();
        NlpType = NLP_WORD;
        Hyphen = false;
    }
};

// represents collection of multitokens in TNlpParser
class TNlpParserBase {
    TVector<TParserToken> Tokens; // it has at least 1 multitoken
    TParserToken* Current;
    TCharSpan CurCharSpan;
    size_t PrefixLen;
    wchar16 PrefixChar;
    wchar16 SuffixChar;

public:
    TNlpParserBase()
        : Tokens(1)
        , Current(&Tokens[0])
        , PrefixLen(0)
        , PrefixChar(0)
        , SuffixChar(0)
    {
    }
    TParserToken& GetToken(size_t i) {
        return Tokens[i];
    }
    size_t GetTokenCount() const {
        return Tokens.size();
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

        if (Current->GetSubtokenCount() == MAX_SUBTOKENS) {
            TCharSpan last; // for backward compatibility the last subtoken of the previous multitoken is popped and ...
            Current->CorrectLastToken(last);
            Tokens.push_back(TParserToken(last)); // ... added to the front of the new multitoken
            Current = &Tokens.back();             // new parser token, Current reassigned immediately after push_back()
        }

        Current->AddSubtoken(CurCharSpan, PrefixLen, PrefixChar, SuffixChar);

        CurCharSpan.Pos = 0;
        CurCharSpan.Len = 0; // it is checked in AddLastToken()
        PrefixLen = 0;
        PrefixChar = 0;
        SuffixChar = 0;
    }
    void AddIdeograph(size_t len) {
        Y_ASSERT(!CurCharSpan.Len && (len == 1 || len == 2));
        Current->AddIdeograph(len);
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

        if (Current->GetSubtokenCount())
            Current->SetTokenDelim(TOKDELIM_NULL); // reset delimiter if any
    }
    void UpdatePrefix(wchar16 c) {
        Y_ASSERT(c == '#' || c == '@' || c == '$');
        PrefixLen = 1; // length of prefix can't be more than 1
        PrefixChar = c;
    }
    void UpdateSuffix(wchar16 c) {
        Y_ASSERT(c == '#' || c == '+');
        Current->UpdateSuffix();
        SuffixChar = c;
    }
    TParserToken* GetCurrentToken() {
        return Current;
    }
    void CancelToken() {
        // example: "abc 5% def", '%' can be the first symbol of utf8 encoded character so token is started by call to BeginToken()
        // and then UpdateToken() is called as well but there is no call to AddToken() because '%' is interpreted as a misc character so
        // CurCharSpan.Len must be reset
        CurCharSpan.Len = 0;
        PrefixLen = 0;
        PrefixChar = 0;
        SuffixChar = 0;
    }
    void ResetTokens() {
        Tokens.resize(1);
        Current = &Tokens[0];
        Current->Reset();
    }
    void SetSoftHyphen() {
        Current->SetHyphen(HYPHEN_SOFT);
    }
    void SetHyphenation() {
        Current->SetHyphen(HYPHEN_ORDINARY);
    }
    void SetTokenDelim(ETokenDelim delim, wchar16 /*c*/) {
        Current->SetTokenDelim(delim);
    }
};
