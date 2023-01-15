#pragma once

#include <util/charset/wide.h>
#include <util/generic/singleton.h>

#include "token_structure.h"

//! represents a set of accent characters
//! @note this class is intended to be a singleton because if has quite large array (65 KB)
class TAccentTable: private TNonCopyable {
public:
    TAccentTable();

    bool operator[](wchar16 c) const {
        return Data[c];
    }

private:
    enum { DATA_SIZE = 0xFFFF };
    unsigned char Data[DATA_SIZE];
};

//! represents an accessor to the accent table, this class is able to be copied
class TAccents {
public:
    TAccents()
        : Table(*HugeSingleton<TAccentTable>())
    {
    }

    //! retruns @c true if character is an accent symbol
    bool Check(wchar16 c) const {
        return Table[c];
    }

private:
    const TAccentTable& Table;
};

//! removes characters from @c TWideToken using a filter
//! @note the checker class must have the default constructor and @c Check member function
template <typename TChecker>
class TCharFilter {
    TWideToken Token;
    TCharTemp Buffer;
    TChecker Checker;

private:
    //! copies already verified data - characters and subtokens
    wchar16* CopyData(const TWideToken& token, size_t n, size_t t, wchar16* const data) {
        std::char_traits<wchar16>::copy(data, token.Token, n); // without current character

        Token.SubTokens.clear();
        for (size_t i = 0; i < t; ++i) {
            Token.SubTokens.push_back(token.SubTokens[i]);
        }

        return data + n;
    }

public:
    explicit TCharFilter(size_t bufSize)
        : Buffer(bufSize)
    {
    }

    //! removes accent characters from token
    //! @return if there is no any accent character the function returns the source token
    //!         otherwise the function returns the internal token
    const TWideToken& Filter(const TWideToken& token) {
        if (token.SubTokens.empty())
            return FilterTokenNoSubtokens(token);
        else
            return FilterTokenWithSubtokens(token);
    }

    static bool HasChars(const TWideToken& token) {
        TChecker checker;
        const TTokenStructure& subtokens = token.SubTokens;
        const wchar16* const s = token.Token; // source character sequence
        size_t i = 0;                       // the index of the current source character
        size_t t = 0;                       // the index of the next source subtoken

        while (i < token.Leng) {
            if (t < subtokens.size()) {
                if (i >= subtokens[t].Pos && i < subtokens[t].EndPos()) {
                    // inside a token
                    if (checker.Check(s[i])) {
                        return true;
                    }
                }

                ++i;

                if (i >= subtokens[t].EndPos()) {
                    ++t;
                }
            } else {
                break;
            }
        }

        return false;
    }

private:
    const TWideToken& FilterTokenWithSubtokens(const TWideToken& token) {
        Y_ASSERT(!token.SubTokens.empty());
        Y_ASSERT(token.SubTokens.back().EndPos() <= token.Leng);

        const TTokenStructure& subtokens = token.SubTokens;
        const wchar16* const s = token.Token; // source character sequence
        size_t i = 0;                       // the index of the current source character
        size_t t = 0;                       // the index of the next source subtoken

        while (i < token.Leng) {
            if (t < subtokens.size()) {
                if (i >= subtokens[t].Pos && i < subtokens[t].EndPos()) {
                    // inside a token
                    if (Checker.Check(s[i]))
                        return FilterTokenWithSubtokens(token, s, i, t, Buffer.Data());
                }

                ++i;

                if (i >= subtokens[t].EndPos()) {
                    ++t;
                }
            } else {
                break;
            }
        }

        return token;
    }

    const TWideToken& FilterTokenWithSubtokens(
        const TWideToken& token, const wchar16* s, size_t i, size_t t, wchar16* const buffer) {
        Y_ASSERT(i < token.Leng && t < token.SubTokens.size() && s >= token.Token);

        const TTokenStructure& subtokens = token.SubTokens;
        wchar16* d = CopyData(token, i, t, buffer); // destination character
        TCharSpan span = subtokens[t];

        while (i < token.Leng) {
            if (t < subtokens.size()) {
                if (i >= subtokens[t].Pos && i < subtokens[t].EndPos()) {
                    // inside a token
                    if (Checker.Check(s[i])) {
                        Y_ASSERT(span.Len);
                        --span.Len;
                    } else {
                        *d++ = s[i];
                    }
                } else {
                    // outside of tokens
                    *d++ = s[i];
                }

                ++i;

                if (i >= subtokens[t].EndPos()) {
                    ++t;

                    if (span.Len)
                        Token.SubTokens.push_back(span);

                    if (t < subtokens.size()) {
                        const size_t diff = i - (d - buffer);
                        Y_ASSERT(subtokens[t].Pos >= diff);
                        span.Pos = subtokens[t].Pos - diff;
                        span.Len = subtokens[t].Len;
                    }
                }
            } else {
                // copy the remainder of characters
                const size_t n = token.Leng - i;
                std::char_traits<wchar16>::copy(d, &s[i], n);
                d += n;
                break;
            }
        }

        Token.Token = buffer;
        Token.Leng = d - buffer;
        Y_ASSERT(!Token.SubTokens.size() || (Token.SubTokens.size() && Token.Leng >= Token.SubTokens.back().EndPos()));
        return Token;
    }

    const TWideToken& FilterTokenNoSubtokens(const TWideToken& token) {
        Y_ASSERT(token.SubTokens.empty());
        const wchar16* s = token.Token;
        const wchar16* const e = s + token.Leng;

        for (; s != e; ++s) {
            if (Checker.Check(*s))
                return FilterTokenNoSubtokens(token.Token, s, e, Buffer.Data());
        }

        return token;
    }

    const TWideToken& FilterTokenNoSubtokens(
        const wchar16* const token, const wchar16* s, const wchar16* const e, wchar16* const buffer) {
        const size_t n = s - token;
        std::char_traits<wchar16>::copy(buffer, token, n);
        wchar16* d = buffer + n;

        for (; s != e; ++s) {
            if (!Checker.Check(*s))
                *d++ = *s;
        }

        Token.Token = buffer;
        Token.Leng = d - buffer;
        Y_ASSERT(Token.Leng);
        return Token;
    }
};

const wchar32* LemmerDecomposition(wchar32 ch, bool advancedGermanUmlauts = true, bool extTable = false);
size_t NormalizeUnicode(const wchar16* word, size_t length, wchar16* converted, size_t bufLen, bool advancedGermanUmlauts = true, bool extTable = false);
TUtf16String NormalizeUnicode(const TUtf16String& w, bool advancedGermanUmlauts = true, bool extTable = false);
TUtf16String NormalizeUnicode(const TWtringBuf& wbuf, bool advancedGermanUmlauts = true, bool extTable = false);
