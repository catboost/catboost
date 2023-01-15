#pragma once

#include <util/generic/noncopyable.h>
#include <util/generic/vector.h>
#include <util/system/maxlen.h>
#include <library/cpp/token/token_structure.h>
#include <library/cpp/token/nlptypes.h>
#include "nlpparserbase.h"

class ITokenHandler;
class TSentBreakFilter;

//! this class is implemented to be used by @c TNlpTokenizer only
class TNlpParser: private TNonCopyable {
protected:
    TNlpParserBase Base;
    ITokenHandler& TokenHandler;
    const wchar16* SentenceBreak;
    TSentBreakFilter& SentBreakFilter;
    const wchar16* OrigText;
    const unsigned char* Text;
    const unsigned char* EndOfText; //!< points to null-terminator, it must be assigned if ExtraLen != 0
    TVector<std::pair<ui32, ui32>> ExtraLen;
    size_t ExtraLenIndex;
    const bool SpacePreserve;
    const bool BackwardCompatible;
    const bool SemicolonBreaksSentence;
    const bool UrlDecode;
    TTempArray<wchar16>& Buffer;

    static const unsigned char CharClasses[65536];

public:
    // see also charclasses_8.rl
    enum ECharClass {
        CC_ZERO = 0x00,            // (EOF) [\0]
        CC_TAB = 0x09,             // [\t]
        CC_LINE_FEED = 0x0A,       // [\n]
        CC_CARRIAGE_RETURN = 0x0D, // [\r]
        CC_SPACE = 0x20,           // [ ]
        CC_QUOTATION_MARK = 0x22,  // ["]
        CC_NUMBER_SIGN = 0x23,     // [#]
        CC_DOLLAR_SIGN = 0x24,     // [$]
        CC_PERCENT = 0x25,         // [%]
        CC_AMPERSAND = 0x26,       // [&]
        CC_APOSTROPHE = 0x27,      // [']
        CC_ASTERISK = 0x2A,        // [*]
        CC_PLUS = 0x2B,            // [+]
        CC_COMMA = 0x2C,           // [,]
        CC_MINUS = 0x2D,           // [-]
        CC_FULL_STOP = 0x2E,       // [.]
        CC_SLASH = 0x2F,           // [/]
        CC_DIGIT = 0x31,           // [1]
        CC_AT_SIGN = 0x40,         // [@]
        CC_CAPITAL_LETTER = 0x41,  // [A]
        CC_UNDERSCORE = 0x5F,      // [_]
        CC_SMALL_LETTER = 0x61,    // [a]
        CC_COMBINING_MARK = 0x80,
        CC_UNICASE_ALPHA = 0x81,
        CC_SOFT_HYPHEN = 0x8F,
        CC_IDEOGRAPH = 0x9F,
        CC_NON_BREAKING_SPACE = 0xA0,
        CC_SECTION_SIGN = 0xA7,
        CC_COPYRIGHT_SIGN = 0xA9,
        CC_SPECIAL = 0xB0,

        CC_OPENING_PUNCT = 0xB2, // [(\[{'"]
        CC_CLOSING_PUNCT = 0xB3, // [)\]}'"]
        CC_SURROGATE_LEAD = 0xB4,
        CC_SURROGATE_TAIL = 0xB5,
        CC_WHITESPACE = 0xB6, // [\t\n\v\f\r ]
        CC_NUMERO_SIGN = 0xB7,
        CC_CJK_TERM_PUNCT = 0xBA, // 0x3002 and other CJK full stops
        CC_TERM_PUNCT = 0xBB,     // terminating punctuation [!.:?]
        CC_CURRENCY = 0xBC,
        CC_CONTROL = 0xBD, // 0x01 - 0x1F, 0x7F excluding \t \n \r
        CC_MISC_TEXT = 0xBE,

        CC_UNASSIGNED = 0xFF
    };

    TNlpParser(ITokenHandler& handler, TSentBreakFilter& sentBreakFilter, TTempArray<wchar16>& buffer,
               bool spacePreserve, bool backwardCompatible, bool semicolonBreaksSentence = false,
               bool urlDecode = true);
    virtual ~TNlpParser() = default;

    //depending on content, tokens can point to data text or Buffer
    //textStart points to string passed to ragel
    void Execute(const wchar16* text, size_t len, const wchar16** textStart=nullptr);

    static int GetCharClass(wchar16 ch) {
        return CharClasses[ch];
    }

protected:
    virtual void ExecuteImpl(const unsigned char* text, size_t len) = 0;

    // the size of the buffer must not be less than (len + 1)
    // text can contain zeros
    void ConvertTextToCharClasses(const wchar16* text, size_t len, unsigned char* buffer);

    //! @todo now the following phrases processed incorrectly:
    //!       HTML-\nand VRML-documents -> HTMLand VRML-documents
    //!       son-\nin-law -> sonin-law
    //!       to fix this problem such phrases could be analyzed by the lemmer
    virtual void MakeMultitokenEntry(TParserToken& token, const wchar16* entry) = 0;

    void CutTooLongMultitoken(TTokenStructure& subtokens, const wchar16*& entry, size_t& leng, size_t& origleng, NLP_TYPE& type);

    void PassBackwardCompatibleToken(const TWideToken& multitoken, NLP_TYPE type, size_t totalLen);

    virtual void MakeEntry(const wchar16* entry, size_t entryLen, NLP_TYPE type);

    size_t GetExtraLen(const wchar16* entry, size_t entryLen);

    //! @note @c leng must include the first letter (or digit) of the next sentence
    size_t MakeSentenceBreak(const wchar16* entry, size_t leng);

    //! marks the end of sentence before any punctuation at the beginning of the next sentence
    void MarkSentenceBreak(const wchar16* p) {
        SentenceBreak = p;
    }

    //! marks the end of sentence if there is no punctuation at the beginning of the next sentence
    void EnsureSentenceBreak(const wchar16* p) {
        if (!SentenceBreak)
            SentenceBreak = p;
    }

    //! resets members describing sentence break
    void ResetSentenceBreak() {
        SentenceBreak = nullptr;
    }

    void ProcessMultitoken(const wchar16* ts, const wchar16* te);

    void ProcessSurrogatePairs(const wchar16* ts, const wchar16* te);

    void ProcessIdeographs(const wchar16* ts, const wchar16* te);

    const wchar16* GetOrigText(const unsigned char* p) {
        return OrigText + (p - Text);
    }

    //////////////////////////////////////////////////////////////////////////////////////////
    // functions wrappers {{
    void ProcessMultitoken(const unsigned char* ts, const unsigned char* te) {
        ProcessMultitoken(GetOrigText(ts), GetOrigText(te));
    }
    void ProcessSurrogatePairs(const unsigned char* ts, const unsigned char* te) {
        ProcessSurrogatePairs(GetOrigText(ts), GetOrigText(te));
    }
    void ProcessIdeographs(const unsigned char* ts, const unsigned char* te) {
        ProcessIdeographs(GetOrigText(ts), GetOrigText(te));
    }
    size_t MakeSentenceBreak(const unsigned char* entry, size_t leng) {
        return MakeSentenceBreak(GetOrigText(entry), leng);
    }
    void MarkSentenceBreak(const unsigned char* p) {
        MarkSentenceBreak(GetOrigText(p));
    }
    void EnsureSentenceBreak(const unsigned char* p) {
        EnsureSentenceBreak(GetOrigText(p));
    }
    void MakeEntry(const unsigned char* entry, size_t len, NLP_TYPE type) {
        MakeEntry(GetOrigText(entry), len, type);
    }
    void BeginToken(const unsigned char* tokstart, const unsigned char* p) {
        Base.BeginToken(GetOrigText(tokstart), GetOrigText(p));
    }
    void BeginToken(const unsigned char* tokstart, const unsigned char* p, ETokenType type) {
        Base.BeginToken(GetOrigText(tokstart), GetOrigText(p), type);
    }
    void UpdateToken() {
        Base.UpdateToken();
    }
    void AddToken() {
        Base.AddToken();
    }
    void CancelToken() {
        Base.CancelToken();
    }
    void UpdatePrefix(wchar16 c) {
        Y_ASSERT(c == '#' || c == '@' || c == '$');
        Base.UpdatePrefix(c);
    }
    void UpdateSuffix(wchar16 c) {
        Y_ASSERT(c == '#' || c == '+');
        Base.UpdateSuffix(c);
    }
    void SetSoftHyphen() {
        Base.SetSoftHyphen();
    }
    void SetHyphenation() {
        Base.SetHyphenation();
    }
    void SetTokenDelim(ETokenDelim delim, unsigned char c) {
        Base.SetTokenDelim(delim, c);
    }
    // }}
    //////////////////////////////////////////////////////////////////////////////////////////
};

template<size_t VERSION> class TVersionedNlpParser: public TNlpParser {
public:
    using TNlpParser::TNlpParser;
protected:
    void ExecuteImpl(const unsigned char* text, size_t len) override;
    void MakeMultitokenEntry(TParserToken& token, const wchar16* entry) override;
};
template<> class TVersionedNlpParser<3>: public TNlpParser {
private:
    size_t LastTokenSuffixLength = 0;
    const bool KeepAffixes = true;
    //we have no look-ahead, so if symbol can be part of prefix we can generate token from it only when we understand that it is not included in prefix
    const wchar16* KeepedPotentialPrefix = nullptr;
public:
    using TNlpParser::TNlpParser;
    TVersionedNlpParser(ITokenHandler& handler, TSentBreakFilter& sentBreakFilter, TTempArray<wchar16>& buffer,
               bool spacePreserve, bool backwardCompatible, bool semicolonBreaksSentence = false,
               bool urlDecode = true, bool keepAffixes = false)
        : TNlpParser(handler, sentBreakFilter, buffer, spacePreserve, backwardCompatible, semicolonBreaksSentence, urlDecode)
        , KeepAffixes(keepAffixes)
    {
    }
protected:
    void ExecuteImpl(const unsigned char* text, size_t len) override;
    //returns number of chars to shift after: -1 if we need to re-process last symbol, 0 normally
    int MakeMisctextEntry(const unsigned char* entry, size_t len, size_t availableAfter);
    void MakeMultitokenEntry(TParserToken& token, const wchar16* entry) override;
    void FlushKeepedPotentialPrefix();

    using TNlpParser::MakeEntry;
    void MakeEntry(const wchar16* entry, size_t entryLen, NLP_TYPE type) override;
};
using TDefaultNlpParser = TVersionedNlpParser<2>;
