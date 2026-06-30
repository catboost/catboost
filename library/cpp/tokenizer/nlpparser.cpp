#include "nlpparser.h"
#include "sentbreakfilter.h"
#include "special_tokens.h"

#include <library/cpp/token/charfilter.h>
#include <library/cpp/token/token_iterator.h>

#include <util/charset/utf8.h>
#include <util/charset/wide.h>
#include <util/generic/utility.h>

namespace {
    const char PERCENT_CHAR = '%';

    //! returns pointer to the first non-accent symbol, if it is not found - returns NULL
    const wchar16* FindNonAccent(const wchar16* p, size_t n) {
        const TAccents accents;
        const wchar16* const e = p + n;
        for (; p != e; ++p) {
            if (!accents.Check(*p))
                break;
        }
        return p;
    }

    inline char HexToChar(char h) {
        Y_ASSERT(isxdigit(h));
        return (
            h >= 'a' ? h - 'a' + 10 : h >= 'A' ? h - 'A' + 10 : h >= '0' ? h - '0' : 0);
    }
}

#ifdef ROBOT_OLDTOK
//compilation error
TNlpParser::TNlpParser(ITokenHandler& handler, TSentBreakFilter& sentBreakFilter, bool spacePreserve, bool, bool)
#else
TNlpParser::TNlpParser(ITokenHandler& handler, TSentBreakFilter& sentBreakFilter, TTempArray<wchar16>& buffer,
                       bool spacePreserve, bool backwardCompatible, bool semicolonBreaksSentence,
                       bool urlDecode)
#endif
    : TokenHandler(handler)
    , SentenceBreak(nullptr)
    , SentBreakFilter(sentBreakFilter)
    , OrigText(nullptr)
    , Text(nullptr)
    , EndOfText(nullptr)
    , ExtraLenIndex(0)
    , SpacePreserve(spacePreserve)
#ifdef ROBOT_OLDTOK
    , BackwardCompatible(true)
    , SemicolonBreaksSentence(false)
#else
    , BackwardCompatible(backwardCompatible)
    , SemicolonBreaksSentence(semicolonBreaksSentence)
#endif
    , UrlDecode(urlDecode)
    , Buffer(buffer)
{
}

void TNlpParser::ProcessMultitoken(const wchar16* ts, const wchar16* te) {
    Base.AddLastToken(ts, te);
    const wchar16* p = ts;
    size_t prevEnd = 0;
    size_t tokenCount = Base.GetTokenCount();
    for (size_t i = 0; i < tokenCount; ++i) {
        TParserToken& tok = Base.GetToken(i);
        if (i) {
            size_t n = tok.GetStart() - prevEnd;
            if (n) {
                MakeEntry(p, n, NLP_MISCTEXT);
                p += n;
            }
        }
        prevEnd = tok.GetEnd();
        MakeMultitokenEntry(tok, p);
        p = ts + prevEnd;
    }
    Base.ResetTokens();
}

template<> void TVersionedNlpParser<2>::MakeMultitokenEntry(TParserToken& token, const wchar16* entry) {
    size_t entryLen = token.GetLength();

    TTokenStructure subtokens;
    token.CorrectPositions();
    Y_ASSERT(token.GetLength() == entryLen); // check after the correction
    token.SwapSubtokens(subtokens);

    const wchar16* tokenText = entry;
    size_t tokenLen = entryLen;
    NLP_TYPE type = token.GetNlpType();
    wchar16 buffer[TOKEN_MAX_BUF];
    if (token.HasHyphen()) {
        type = PrepareMultitoken(subtokens, buffer, TOKEN_MAX_BUF, entry, tokenLen);
        Y_ASSERT(tokenLen <= TOKEN_MAX_LEN); // multitoken isn't longer than TOKEN_MAX_LEN
        tokenText = buffer;
    } else {
        // NLP_WORD, NLP_INTEGER, NLP_FLOAT, NLP_MARK are needed to be cut off
        Y_ASSERT(type == NLP_WORD || type == NLP_INTEGER || type == NLP_FLOAT || type == NLP_MARK);
        if (tokenLen > TOKEN_MAX_LEN) {
            // entry here always points to the original text
            CutTooLongMultitoken(subtokens, entry, tokenLen, entryLen, type);
            tokenText = entry; // in case if leading accents were removed...
        }
    }

    TWideToken multitoken; // don't call to constructor TWideToken(entry, leng)!
    multitoken.Token = tokenText;
    multitoken.Leng = tokenLen;
    multitoken.SubTokens.swap(subtokens);
    Y_ASSERT(CheckMultitoken(multitoken));

    const size_t totalLen = entryLen + GetExtraLen(entry, entryLen);

    Y_ASSERT(multitoken.SubTokens.size());
    if (BackwardCompatible) {
        PassBackwardCompatibleToken(multitoken, type, totalLen);
    } else {
        SentBreakFilter.OnToken(multitoken, type);
        TokenHandler.OnToken(multitoken, totalLen, type);
    }
}

void TVersionedNlpParser<3>::MakeMultitokenEntry(TParserToken& token, const wchar16* entry) {
    size_t entryLen = token.GetLength();

    TTokenStructure subtokens;
    token.CorrectPositions();
    Y_ASSERT(token.GetLength() == entryLen); // check after the correction
    token.SwapSubtokens(subtokens);

    KeepedPotentialPrefix = nullptr;
    if (!subtokens.empty() && subtokens[0].PrefixLen > 0) {
        Y_ASSERT(subtokens[0].PrefixLen == 1);
        // in case x#y we have already tokenized # as suffix
        if (!KeepAffixes && LastTokenSuffixLength == 0) {
            MakeEntry(entry, 1, NLP_WORD);
        }
        if (!KeepAffixes || LastTokenSuffixLength != 0) {
            subtokens[0].PrefixLen = 0;
            --entryLen;
            ++entry;
            for (auto& subtoken : subtokens) {
                --subtoken.Pos;
            }
        }
        LastTokenSuffixLength = 0;
    }

    const wchar16* tokenText = entry;
    size_t tokenLen = entryLen;
    NLP_TYPE type = token.GetNlpType();
    wchar16 buffer[TOKEN_MAX_BUF];
    if (token.HasHyphen()) {
        type = PrepareMultitoken(subtokens, buffer, TOKEN_MAX_BUF, entry, tokenLen);
        Y_ASSERT(tokenLen <= TOKEN_MAX_LEN); // multitoken isn't longer than TOKEN_MAX_LEN
        tokenText = buffer;
    } else {
        // NLP_WORD, NLP_INTEGER, NLP_FLOAT, NLP_MARK are needed to be cut off
        Y_ASSERT(type == NLP_WORD || type == NLP_INTEGER || type == NLP_FLOAT || type == NLP_MARK);
        if (tokenLen > TOKEN_MAX_LEN) {
            // entry here always points to the original text
            CutTooLongMultitoken(subtokens, entry, tokenLen, entryLen, type);
            tokenText = entry; // in case if leading accents were removed...
        }
    }

    TWideToken multitoken; // don't call to constructor TWideToken(entry, leng)!
    multitoken.Token = tokenText;
    multitoken.Leng = tokenLen;
    multitoken.SubTokens.swap(subtokens);
    Y_ASSERT(CheckMultitoken(multitoken));

    const size_t totalLen = entryLen + GetExtraLen(entry, entryLen);

    Y_ASSERT(multitoken.SubTokens.size());
    if (BackwardCompatible) {
        PassBackwardCompatibleToken(multitoken, type, totalLen);
    } else {
        SentBreakFilter.OnToken(multitoken, type);
        TokenHandler.OnToken(multitoken, totalLen, type);
    }

}

size_t TNlpParser::GetExtraLen(const wchar16* entry, size_t entryLen) {
    const size_t offset = entry - OrigText;
    const size_t endOffset = offset + entryLen;
    ui32 extraLen = 0;
    while (ExtraLenIndex < ExtraLen.size() &&
           ExtraLen[ExtraLenIndex].first > offset && ExtraLen[ExtraLenIndex].first <= endOffset) {
        extraLen += ExtraLen[ExtraLenIndex].second;
        ++ExtraLenIndex;
    }
    return extraLen;
}

void TNlpParser::CutTooLongMultitoken(TTokenStructure& subtokens, const wchar16*& entry, size_t& leng, size_t& origleng, NLP_TYPE& type) {
    Y_ASSERT(leng > TOKEN_MAX_LEN);
    if (type == NLP_WORD || type == NLP_INTEGER || type == NLP_MARK) {
        // if too many accent symbols are in the beginning of the token (the number is greater than TOKEN_MAX_LEN)
        // TODO: remove accents before tokenization
        const ptrdiff_t n = FindNonAccent(entry, leng) - entry;
        Y_ASSERT(n >= 0);

        // NLP_WORD contains words only, NLP_INTEGER - integers only, NLP_MARK - words and integers
        Y_ASSERT(!subtokens.empty());

        if (n > 0) {
            const TWideToken miscText(entry, n); // the first part containing accents only
            TokenHandler.OnToken(miscText, n, NLP_MISCTEXT);
            origleng -= n;
            entry += n;
            leng = AdjustSubtokens(subtokens, n, TOKEN_MAX_LEN);
        } else
            leng = AdjustSubtokens(subtokens, TOKEN_MAX_LEN);

        // correct NLP type
        if (type == NLP_MARK) {
            Y_ASSERT(!subtokens.empty());
            ETokenType tokType = subtokens[0].Type;
            Y_ASSERT(tokType == TOKEN_WORD || tokType == TOKEN_NUMBER);
            for (size_t i = 1; i < subtokens.size(); ++i) {
                if (subtokens[i].Type != tokType) {
                    tokType = TOKEN_MARK;
                    break;
                }
            }
            if (tokType != TOKEN_MARK)
                type = (tokType == TOKEN_WORD ? NLP_WORD : NLP_INTEGER);
        }
    } else {
        // no processing of the case when point of a NLP_FLOAT token is cut off (position of the
        // point character is greater than TOKEN_MAX_LEN) and token actually will be integer
        Y_ASSERT(subtokens.empty());
        leng = TOKEN_MAX_LEN;
    }
}

void TNlpParser::PassBackwardCompatibleToken(const TWideToken& multitoken, NLP_TYPE type, size_t totalLen) {
    if (multitoken.SubTokens.size() == 1) {
        const TCharSpan& subtok = multitoken.SubTokens[0];
        TWideToken tok;
        if (subtok.PrefixLen) {
            tok.Token = multitoken.Token;
            tok.Leng = subtok.PrefixLen;
            SentBreakFilter.OnToken(tok, NLP_MISCTEXT);
            TokenHandler.OnToken(tok, tok.Leng, NLP_MISCTEXT);
        }

        const ui16 prefixLen = subtok.PrefixLen;
        tok.Token = multitoken.Token + prefixLen;
        tok.Leng = multitoken.Leng - prefixLen;
        tok.SubTokens.push_back(subtok);
        tok.SubTokens[0].PrefixLen = 0;
        tok.SubTokens[0].Pos -= prefixLen;

        SentBreakFilter.OnToken(tok, type);
        TokenHandler.OnToken(tok, tok.Leng + (totalLen - multitoken.Leng), type);
        // suffix after alone number is kept, for example: 18+ -> [18+]
        // if number with suffix is part of multitoken then suffix will be removed: 16+/18+ -> [16]+/[18]+
        // see also tokenizer_ut.cpp
    } else {
        TWideToken tok;
        TTokenIterator it(multitoken);
        it.GetPrefix(tok); // prefix of the first token
        if (tok.Leng) {
            SentBreakFilter.OnToken(tok, NLP_MISCTEXT);
            TokenHandler.OnToken(tok, tok.Leng, NLP_MISCTEXT);
        }
        while (it.Next()) {
            it.GetMultitoken(tok);
            SentBreakFilter.OnToken(tok, it.GetNlpType());
            if (!it.Finished())
                TokenHandler.OnToken(tok, tok.Leng, it.GetNlpType());
            else
                TokenHandler.OnToken(tok, tok.Leng + (totalLen - multitoken.Leng), it.GetNlpType());
            it.GetDelimiter(tok);
            if (tok.Leng) {
                SentBreakFilter.OnToken(tok, NLP_MISCTEXT);
                TokenHandler.OnToken(tok, tok.Leng, NLP_MISCTEXT);
            }
        }
    }
}

void TNlpParser::MakeEntry(const wchar16* entry, size_t entryLen, NLP_TYPE type) {
    TWideToken token; // don't call to constructor TWideToken(entry, leng)!
    token.Token = entry;
    token.Leng = entryLen;

    const size_t totalLen = entryLen + GetExtraLen(entry, entryLen);

    SentBreakFilter.OnToken(token, type);
    TokenHandler.OnToken(token, totalLen, type);
}

void TVersionedNlpParser<3>::MakeEntry(const wchar16* entry, size_t entryLen, NLP_TYPE type) {
    if (KeepedPotentialPrefix) {
        TWideToken token(KeepedPotentialPrefix, 1);
        SentBreakFilter.OnToken(token, type);
        TokenHandler.OnToken(token, entryLen, type);
        KeepedPotentialPrefix = nullptr;
        entryLen -= 1;
        entry += 1;
        if (entryLen == 0) {
            return;
        }
    }
    if (type == NLP_WORD) {
        TWideToken token(entry, entryLen);
        SentBreakFilter.OnToken(token, type);
        TokenHandler.OnToken(token, entryLen, type);
        return;
    }
    return TNlpParser::MakeEntry(entry, entryLen, type);
}

namespace {
    bool IsWhitespaceClass(const unsigned char c) {
        return c == TNlpParser::CC_LINE_FEED
            || c == TNlpParser::CC_TAB
            || c == TNlpParser::CC_CARRIAGE_RETURN
            || c == TNlpParser::CC_WHITESPACE;
    }

    bool IsNotSpace(wchar16 c) {
        return !IsWhitespaceClass(TNlpParser::GetCharClass(c));
    }
}

int TVersionedNlpParser<3>::MakeMisctextEntry(const unsigned char* entry, size_t len, size_t availableAfter) {
    const wchar16* entry16 = GetOrigText(entry);
    size_t skipFirst = LastTokenSuffixLength;
    LastTokenSuffixLength = 0;
    //if last symbol is #, @ or $ (tokprefix), we may want to create token from it if it is prefix
    bool leftLast = (len > 1) && (entry16[len - 1] == '#' || entry16[len - 1] == '@' || entry16[len - 1] == '$');
    while (len > 0) {
        const wchar16* misctextEnd = std::find_if(entry16, entry16 + len, IsNotSpace);
        size_t interestingLength = 0;
        while (misctextEnd < entry16 + len) {
            interestingLength = GetSpecialTokenLength(misctextEnd, len - (misctextEnd - entry16) + availableAfter);
            if (interestingLength != 0) {
                break;
            }
            misctextEnd = std::find_if(misctextEnd + 1, entry16 + len, IsNotSpace);
        }
        if (misctextEnd > entry16) {
            while (skipFirst && misctextEnd > entry16) {
                ++entry16;
                --len;
                --skipFirst;
            }
            if (leftLast && misctextEnd == len + entry16) {
                if (misctextEnd - entry16 > 1) {
                    MakeEntry(entry16, misctextEnd - entry16 - 1, NLP_MISCTEXT);
                }
                return -1;
            }
            if (misctextEnd > entry16) {
                MakeEntry(entry16, misctextEnd - entry16, NLP_MISCTEXT);
            }
            len -= misctextEnd - entry16;
            entry16 = misctextEnd;
        }
        Y_ASSERT(misctextEnd == entry16);
        if (interestingLength > 0) {
            while (skipFirst && interestingLength && len) {
                ++entry16;
                --interestingLength;
                --len;
                --skipFirst;
            }
            if (KeepAffixes && leftLast && len == interestingLength) {
                for (size_t i = 0; i + 1 < interestingLength; ++i) {
                    MakeEntry(entry16 + i, 1, NLP_WORD);
                }
                KeepedPotentialPrefix = entry16 + interestingLength - 1;
                return -1;
            }
            for (size_t i = 0; i < interestingLength; ++i) {
                MakeEntry(entry16 + i, 1, NLP_WORD);
            }
            if (interestingLength > len) {
                return interestingLength - len;
            }
            len -= interestingLength;
            entry16 += interestingLength;
        }
    }
    return 0;
}

size_t TNlpParser::MakeSentenceBreak(const wchar16* entry, size_t leng) {
    if (!SentenceBreak)
        SentenceBreak = entry + leng - 1; // last symbol is ytitle
    const size_t end = SentenceBreak - entry;
    assert(0 < end && end <= leng);

    MakeEntry(entry, end, SentBreakFilter.OnSentBreak(entry, leng));
    SentenceBreak = nullptr;
    return end; // adjust the current position, excluding the start of the sentence
}

void TNlpParser::ProcessSurrogatePairs(const wchar16* ts, const wchar16* te) {
    const wchar16 brokenRune = BROKEN_RUNE;
    const wchar16* lead = nullptr;
    for (const wchar16* p = ts; p != te; ++p) {
        if (IsW16SurrogateLead(*p)) {
            if (lead)
                MakeEntry(&brokenRune, 1, NLP_MISCTEXT);
            lead = p;
        } else if (IsW16SurrogateTail(*p)) {
            if (lead) {
                Base.AddIdeograph(2);
                Y_ASSERT(Base.GetTokenCount() == 1);
                MakeMultitokenEntry(Base.GetToken(0), lead);
                Base.ResetTokens();
            } else
                MakeEntry(&brokenRune, 1, NLP_MISCTEXT);
            lead = nullptr;
        } else
            Y_ASSERT(!"invalid character");
    }
    if (lead)
        MakeEntry(&brokenRune, 1, NLP_MISCTEXT);
}

void TNlpParser::ProcessIdeographs(const wchar16* ts, const wchar16* te) {
    for (const wchar16* p = ts; p != te; ++p) {
        Base.AddIdeograph(1);
        Y_ASSERT(Base.GetTokenCount() == 1);
        MakeMultitokenEntry(Base.GetToken(0), p);
        Base.ResetTokens();
    }
}

void TNlpParser::Execute(const wchar16* text, size_t len, const wchar16** textStart) {
    if (!len)
        return;
    const wchar16* p = text;
    const wchar16* e = p + len;
    wchar16* data = nullptr;
    wchar16* dest = nullptr;
    ExtraLen.clear();
    ExtraLenIndex = 0;

    while (p != e) {
        if (UrlDecode && *p == PERCENT_CHAR && (p + 3) <= e && IsHexdigit(p[1]) && IsHexdigit(p[2])) {
            if (!dest) {
                Buffer = TTempArray<wchar16>(len + 1);
                data = Buffer.Data();
                dest = data;
                const size_t n = p - text;
                std::char_traits<wchar16>::copy(dest, text, n);
                dest += n;
            }

            const wchar16* start = p; // in case if UTF8 is bad
            TTempBuf buf(e - p);      // for UTF8
            char* const utf8 = buf.Data();
            size_t i = 0;
            while (p != e && *p == PERCENT_CHAR && (p + 3) <= e && IsHexdigit(p[1]) && IsHexdigit(p[2])) {
                const char c = (HexToChar(char(p[1])) << 4) | HexToChar(char(p[2]));
                utf8[i++] = ((unsigned char)c < 0x20 ? ' ' : c); // replace all controlling characters with ' '
                p += 3;
            }

            bool decoded = false;
            // convert at least 2 UTF8 bytes
            if (i > 1) {
                decoded = true;
                Y_ABORT_UNLESS(size_t(p - start) == 3 * i);
                size_t written = 0;
                const size_t extraLenRollback = ExtraLen.size();
                for (size_t j = 0; j < i;) {
                    size_t stepRead = 0;
                    if (RECODE_OK != GetUTF8CharLen(stepRead, reinterpret_cast<const unsigned char*>(utf8) + j, reinterpret_cast<const unsigned char*>(utf8) + i)) {
                        decoded = false;
                        break;
                    }
                    Y_ABORT_UNLESS(stepRead && j + stepRead <= i);
                    size_t stepWritten = 0;
                    if (!UTF8ToWide(utf8 + j, stepRead, dest + written, stepWritten)) {
                        decoded = false;
                        break;
                    }
                    written += stepWritten;
                    ExtraLen.push_back(std::make_pair<ui32>(dest + written - data, 3 * stepRead - stepWritten));
                    j += stepRead;
                }
                if (decoded) {
                    dest += written;
                } else {
                    ExtraLen.resize(extraLenRollback);
                }
            }
            if (!decoded) {
                // UTF8 is bad or too short (for example: %action-%61%62%63)
                // copy text as is:
                size_t n = p - start;
                std::char_traits<wchar16>::copy(dest, start, n);
                dest += n;
            }
        } else if (dest)
            *dest++ = *p++;
        else
            ++p;
    }

    if (dest) {
        if (textStart) {
            *textStart = data;
        }
        *dest = 0; // just in case
        const size_t newLen = dest - data;
        TTempBuf convbuf(newLen + 1);
        unsigned char* conv = (unsigned char*)convbuf.Data();
        ConvertTextToCharClasses(data, newLen, conv);
        OrigText = data;
        ExecuteImpl(conv, newLen);
    } else {
        if (textStart) {
            *textStart = text;
        }
        TTempBuf convbuf(len + 1);
        unsigned char* conv = (unsigned char*)convbuf.Data();
        ConvertTextToCharClasses(text, len, conv);
        OrigText = text;
        ExecuteImpl(conv, len);
    }
}

void TNlpParser::ConvertTextToCharClasses(const wchar16* text, size_t len, unsigned char* buffer) {
    const wchar16* end = text + len;
    while (text != end) {
        // TODO: it would be better to copy the char classes table into the new one in the constructor
        //       and to change required char classes in it instead of checking conditions here (semicolon and whitespaces)
        const unsigned char c = (*text == ';' ? (unsigned char)(SemicolonBreaksSentence ? CC_TERM_PUNCT : CC_MISC_TEXT) : CharClasses[*text]);
        ++text;
        if (SpacePreserve)
            *buffer++ = c;
        else {
            // in case of !SpacePreserve all whitespaces are replaced with space because
            // browsers normalize whitespaces: "a \t\n\r b" -> "a b" if tag <pre></pre> isn't used
            // this change fixes incorrect hyphenations without tag <pre>: "HTML-\nfile" is not "HTMLfile"
            // browser show this text as: "HTML- file"
            *buffer++ = (IsWhitespaceClass(c) ? (unsigned char)CC_SPACE : c);
        }
    }
    *buffer = 0;
}
