#pragma once

#include <util/str_stl.h>
#include <util/system/defaults.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/charset/wide.h>

const size_t MAX_SUBTOKENS = 63; // 128; it must not be greater than number of words in a sentence

//! @note tokenizer produces multitokens containins TOKEN_WORD and TOKEN_NUMBER subtokens
enum ETokenType {
    TOKEN_WORD,
    TOKEN_NUMBER,
    TOKEN_MARK,  //!< token contains letters and digits: a1b2
    TOKEN_FLOAT, //!< '1.23' - single token with no delimiter, length is equal to 4; introduced for backward compatibility
    TOKEN_MIXED  //!< actually the very last token can have this type, it must be tokenized again; it can contain delimiters, suffixes, different token types
};

enum EHyphenType {
    HYPHEN_NONE,     //!< means that there is no hyphenation
    HYPHEN_ORDINARY, //!< an ordinary hyphenation, "-\n"
    HYPHEN_SOFT,     //!< U+00AD soft hyphen, "&shy;"
    HYPHEN_HARD      //!< U+2011 non-breaking hyphen, it isn't processed yet, in case of "black-\nwhite" it will be "blackwhite" but it should be "black-white"
};

enum ETokenDelim {
    TOKDELIM_NULL,       //!< means that there is no token delimiter
    TOKDELIM_APOSTROPHE, //!< ' 0x27
    TOKDELIM_MINUS,      //!< - 0x2D
    TOKDELIM_PLUS,       //!< + 0x2B
    TOKDELIM_UNDERSCORE, //!< _ 0x5F
    TOKDELIM_SLASH,      //!< / 0x2F
    TOKDELIM_AT_SIGN,    //!< @ 0x40
    TOKDELIM_DOT,        //!< . 0x2E
    TOKDELIM_UNKNOWN,    //!< used between concatenated multitokens (such delimiters can have length greater than 1 or equal to 0)
};

// Position of a single token in the word
struct TCharSpan {
    TCharSpan()
        : Pos(0)
        , Len(0)
        , PrefixLen(0)
        , SuffixLen(0)
        , Type(TOKEN_WORD)
        , Hyphen(HYPHEN_NONE)
        , TokenDelim(TOKDELIM_NULL)
    {
    }

    TCharSpan(size_t offset, size_t len, ETokenType type = TOKEN_WORD, ETokenDelim tokenDelim = TOKDELIM_NULL, EHyphenType hyphen = HYPHEN_NONE,
              ui16 suffixLen = 0, ui16 prefixLen = 0)
        : Pos(offset)
        , Len(len)
        , PrefixLen(prefixLen)
        , SuffixLen(suffixLen)
        , Type(type)
        , Hyphen(hyphen)
        , TokenDelim(tokenDelim)
    {
    }

    size_t Pos;             //!< offset from the beginning of the word to the start of the token
    size_t Len;             //!< token length, it does not include suffix length
    ui16 PrefixLen;         //!< Len doesn't include PrefixLen
    ui16 SuffixLen;         //!< suffix length
    ETokenType Type;        //!< type of this token
    EHyphenType Hyphen;     //!< hyphenation after the token, it should be verified by the lemmer and concatenated if the word is correct
    ETokenDelim TokenDelim; //!< if true then apostrophe or minus follows this token

    //! @note end position does not include suffix length
    inline size_t EndPos() const {
        return Pos + Len;
    }

    bool operator==(const TCharSpan& span) const {
        return ((span.Pos == Pos) && (span.Len == Len) &&
                (span.PrefixLen == PrefixLen) && (span.SuffixLen == SuffixLen) &&
                (span.Type == Type) && (span.Hyphen == Hyphen) && (span.TokenDelim == TokenDelim));
    }

    size_t Hash() const {
        size_t result = Pos;
        result = CombineHashes(result, Len);
        result = CombineHashes(result, size_t(PrefixLen));
        result = CombineHashes(result, size_t(SuffixLen));
        result = CombineHashes(result, size_t(Type));
        result = CombineHashes(result, size_t(Hyphen));
        result = CombineHashes(result, THash<size_t>()(TokenDelim));
        return result;
    }
};

template <class T>
struct THash;

template <>
struct THash<TCharSpan> {
    size_t operator()(const TCharSpan& s) const {
        return s.Hash();
    }
};
// Vector of immutable TCharSpan elements, with upper limit on size: MAX_SUBTOKENS
// Mimics std::vector<TCharSpan> as closely as possible

class TTokenStructure {
public:
    TTokenStructure() {
        //        Tokens.reserve(MAX_SUBTOKENS); don't reserve memory
    }
    void push_back(const TCharSpan& s) {
        push_back(s.Pos, s.Len, s.Type, s.TokenDelim, s.Hyphen, s.SuffixLen, s.PrefixLen);
    }
    void push_back(size_t offset, size_t len, ETokenType type = TOKEN_WORD, ETokenDelim tokenDelim = TOKDELIM_NULL, EHyphenType hyphen = HYPHEN_NONE,
                   ui16 suffixLen = 0, ui16 prefixLen = 0) {
        Y_ASSERT(type == TOKEN_WORD || type == TOKEN_NUMBER || type == TOKEN_MARK || type == TOKEN_FLOAT || type == TOKEN_MIXED);
        if (Tokens.size() == MAX_SUBTOKENS) {
            // no more room - append to the last subtoken
            TCharSpan& span = Tokens.back();
            span.Len = len + offset - span.Pos;
            // PrefixLen not changed
            span.SuffixLen = suffixLen;
            span.Type = TOKEN_MIXED; // token gets invalid: it can contain delimiters, suffixes and different token types
            span.Hyphen = HYPHEN_NONE;
            span.TokenDelim = TOKDELIM_NULL;
        } else {
            // insert a new one
            Tokens.push_back(
                TCharSpan(offset, len, type, tokenDelim, hyphen, suffixLen, prefixLen));
        }
    }
    void pop_back() {
        Tokens.pop_back();
    }
    const TCharSpan* begin() const {
        return Tokens.begin();
    }
    const TCharSpan* end() const {
        return Tokens.end();
    }
    TCharSpan* begin() {
        return Tokens.begin();
    }
    TCharSpan* end() {
        return Tokens.end();
    }
    const TCharSpan* cbegin() const {
        return Tokens.cbegin();
    }
    const TCharSpan* cend() const {
        return Tokens.cend();
    }
    const TCharSpan& operator[](size_t idx) const {
        return Tokens[idx];
    }
    TCharSpan& operator[](size_t idx) {
        return Tokens[idx];
    }
    const TCharSpan& back() const {
        return Tokens.back();
    }
    TCharSpan& back() {
        return Tokens.back();
    }
    size_t size() const {
        return Tokens.size();
    }
    void resize(size_t newsize) {
        Tokens.resize(Min(newsize, MAX_SUBTOKENS));
    }
    size_t capacity() const {
        return MAX_SUBTOKENS;
    }
    bool empty() const {
        return Tokens.empty();
    }
    void clear() {
        Tokens.clear();
    }
    void swap(TTokenStructure& other) {
        Tokens.swap(other.Tokens);
    }

private:
    TVector<TCharSpan> Tokens;
};

//! @note actually it is multi-token containing several tokens
struct TWideToken {
    const wchar16* Token;
    size_t Leng;
    TTokenStructure SubTokens;

    TWideToken()
        : Token(nullptr)
        , Leng(0)
    {
    }

    TWideToken(const wchar16* s, size_t len)
        : Token(s)
        , Leng(len)
    {
        SubTokens.push_back(0, len);
    }

    TWideToken(const wchar16* token, size_t len, const TTokenStructure& subTokens)
        : Token(token)
        , Leng(len)
        , SubTokens(subTokens)
    {
    }

    void Concat(const TWideToken& tok) {
        Y_ASSERT(Token && Leng && !SubTokens.empty() && Token + Leng <= tok.Token);
        const size_t offset = tok.Token - Token;
        Leng = offset + tok.Leng;
        SubTokens.back().TokenDelim = TOKDELIM_UNKNOWN;
        for (const auto& s : tok.SubTokens) {
            SubTokens.push_back(s.Pos + offset, s.Len, s.Type, s.TokenDelim, s.Hyphen, s.SuffixLen, s.PrefixLen);
        }
    }

    const TWtringBuf Text() const {
        return TWtringBuf(Token, Leng);
    }

    TWideToken Extract(size_t begin, size_t end) const {
        TWideToken ret;
        if (end > SubTokens.size())
            end = SubTokens.size();
        if (begin >= end)
            return ret;

        const size_t shift = SubTokens[begin].Pos - SubTokens[begin].PrefixLen;
        ret.Token = Token + shift;
        for (size_t i = begin; i < end; ++i) {
            ret.SubTokens.push_back(SubTokens[i]);
            ret.SubTokens.back().Pos -= shift;
        }
        ret.SubTokens.back().Hyphen = HYPHEN_NONE;
        ret.SubTokens.back().TokenDelim = TOKDELIM_NULL;
        ret.Leng = ret.SubTokens.back().EndPos() + ret.SubTokens.back().SuffixLen;
        Y_ASSERT(ret.Leng + shift <= Leng);
        return ret;
    }
};

inline bool operator==(const TWideToken& a, const TWideToken& b) {
    if ((a.Leng != b.Leng) || memcmp(a.Token, b.Token, a.Leng * sizeof(wchar16)))
        return false;
    if (a.SubTokens.size() != b.SubTokens.size())
        return false;
    for (size_t i = 0; i < a.SubTokens.size(); ++i) {
        if (!(a.SubTokens[i] == b.SubTokens[i]))
            return false;
    }
    return true;
}

inline bool GetLeftTokenDelim(const TWideToken& tok, size_t index) {
    const TTokenStructure& subtokens = tok.SubTokens;
    Y_ASSERT(index < subtokens.size());
    if (index == 0)
        return false;
    const TCharSpan& subtok = subtokens[index];
    const TCharSpan& prev = subtokens[index - 1];
    return (prev.EndPos() + prev.SuffixLen < subtok.Pos - subtok.PrefixLen);
}

inline wchar16 GetRightTokenDelim(const TWideToken& tok, size_t index) {
    const TTokenStructure& subtokens = tok.SubTokens;
    Y_ASSERT(index < subtokens.size());
    if (index + 1 >= subtokens.size())
        return 0;
    const TCharSpan& subtok = subtokens[index];
    const size_t delimPos = subtok.EndPos() + subtok.SuffixLen;
    const TCharSpan& next = subtokens[index + 1];
    return (delimPos == next.Pos - next.PrefixLen ? 0 : tok.Token[delimPos]);
}

//! @note yc_minus (0x2D) can be 0xB7 that is an ASCII character as well
inline char GetTokenDelimChar(ETokenDelim delim) {
    switch (delim) {
        case TOKDELIM_APOSTROPHE:
            return '\'';
        case TOKDELIM_MINUS:
            return '-';
        case TOKDELIM_PLUS:
            return '+';
        case TOKDELIM_UNDERSCORE:
            return '_';
        case TOKDELIM_SLASH:
            return '/';
        case TOKDELIM_AT_SIGN:
            return '@';
        case TOKDELIM_DOT:
            return '.';
        case TOKDELIM_NULL:
        case TOKDELIM_UNKNOWN:
        default:
            Y_ASSERT(!"invalid delimiter");
            return 0;
    }
}
