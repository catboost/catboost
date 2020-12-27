#include <util/system/maxlen.h>
#include "multitokenutil.h"

namespace {
    inline void GlueTokens(TCharSpan* desttok, const TCharSpan* srctok) {
        Y_ASSERT(srctok->PrefixLen == 0);
        desttok->Len += srctok->Len;
        desttok->SuffixLen = srctok->SuffixLen;
        desttok->Hyphen = srctok->Hyphen; // the next token can have hyphenation
        desttok->TokenDelim = srctok->TokenDelim;
    }

    inline void CopyToken(TCharSpan* desttok, const TCharSpan* srctok, size_t destpos) {
        Y_ASSERT(srctok->PrefixLen == 0);
        desttok->Pos = destpos;
        desttok->Len = srctok->Len;
        desttok->SuffixLen = srctok->SuffixLen;
        desttok->Type = srctok->Type;
        desttok->Hyphen = srctok->Hyphen; // the next token can have hyphenation
        desttok->TokenDelim = srctok->TokenDelim;
    }

    //! processes hyphenation between subtokens if any
    inline void ProcessHyphenation(wchar16* buffer, const wchar16* entry, TCharSpan*& desttok, const TCharSpan* srctok, size_t& destpos, size_t& srcpos) {
        // Check if the space contains hyphenation, and copy the separator
        if (desttok->Hyphen == HYPHEN_ORDINARY) {
            if (desttok->Type == TOKEN_WORD && srctok->Type == TOKEN_WORD) {
                // glue two adjacent tokens if both are TOKEN_WORD
                srcpos = srctok->Pos;
                GlueTokens(desttok, srctok);
            } else {
                // copy delimiter only - remove all following whitespace characters, for example: "sepa- \n rator" (see nlptok.rl, hyphenation/nlfasblank)
                wchar16* const dest = buffer + destpos;
                const wchar16* const src = entry + srcpos; // srcpos - the beginning of source delimiter
                *dest = *src;
                desttok->TokenDelim = TOKDELIM_MINUS;
                ++destpos;
                srcpos = srctok->Pos; // keep it before CopyToken()
                CopyToken(++desttok, srctok, destpos);
            }
        } else if (desttok->Hyphen == HYPHEN_SOFT) {
            if (desttok->Type == srctok->Type) {
                // glue two adjacent tokens if both have the same type
                srcpos = srctok->Pos;
                GlueTokens(desttok, srctok);
            } else {
                // delimiter isn't copied
                srcpos = srctok->Pos; // keep it before CopyToken()
                CopyToken(++desttok, srctok, destpos);
            }
        } else {
            Y_ASSERT(desttok->Hyphen == HYPHEN_NONE); // HYPHEN_HARD isn't processed yet
            Y_ASSERT(srctok->Pos >= srcpos);
            if (srctok->Pos > srcpos) {
                // copy the delimiter and create the next token
                const size_t seplen = srctok->Pos - srcpos; // srcpos - the beginning of source delimiter

                // copy suffix with delimiter
                std::char_traits<wchar16>::copy(buffer + destpos, entry + srcpos, seplen);
                destpos += seplen;

                srcpos = srctok->Pos; // keep it before CopyToken()
            }
            CopyToken(++desttok, srctok, destpos);
        }
    }

    //! @param len    length of text to be copied (it can include suffix length)
    inline void CopyTokenText(wchar16* buffer, const wchar16* entry, size_t len, size_t& destpos, size_t& srcpos) {
        std::char_traits<wchar16>::copy(buffer + destpos, entry + srcpos, len); // srcpos - the beginning of source token
        srcpos += len;
        destpos += len;
    }

    inline bool CutSubtoken(TCharSpan& s, size_t maxLen) {
        if (s.EndPos() > maxLen) {
            s.Len = maxLen - s.Pos;
            s.SuffixLen = 0;
            return true;
        } else if (s.EndPos() + s.SuffixLen > maxLen) {
            s.SuffixLen = maxLen - s.EndPos();
            return true;
        }
        return false;
    }
}

void CorrectDelimiters(TCharSpan& prevtok, wchar16 suffixChar, TCharSpan& lasttok, wchar16 prefixChar) {
    // correct suffix length of the previous token
    switch (lasttok.Pos - prevtok.EndPos()) {
        case 0:
            Y_ASSERT(prevtok.SuffixLen == 0 && prevtok.TokenDelim == TOKDELIM_NULL && lasttok.PrefixLen == 0);
            break;
        case 1:                                                                // only delimiter allowed
            if (prevtok.SuffixLen == 1 && prevtok.TokenDelim == TOKDELIM_PLUS) // 'a+b'
                prevtok.SuffixLen = 0;
            Y_ASSERT(
                (prevtok.SuffixLen == 0 && prevtok.TokenDelim != TOKDELIM_NULL && lasttok.PrefixLen == 0) ||
                (prevtok.SuffixLen == 0 && prevtok.TokenDelim == TOKDELIM_NULL && lasttok.PrefixLen == 0 && prevtok.Hyphen == HYPHEN_SOFT));
            break;
        case 2:                                                     // only suffix + delimiter OR delimiter + prefix
            if (prevtok.SuffixLen == 1 && lasttok.PrefixLen == 1) { // 'a+@b' - delim/prefix, 'a#@b' - suffix/delim
                if (suffixChar == '+') {
                    prevtok.SuffixLen = 0;
                    prevtok.TokenDelim = TOKDELIM_PLUS;
                } else if (suffixChar == '#') {
                    Y_ASSERT(prefixChar == '@'); // only '@' can be delimiter and prefix
                    prevtok.TokenDelim = TOKDELIM_AT_SIGN;
                    lasttok.PrefixLen = 0;
                }
            } else if (prevtok.SuffixLen == 2) {
                prevtok.SuffixLen = 1;
                prevtok.TokenDelim = TOKDELIM_PLUS; // since SuffixLen == 2 can be '++' only
            }
            Y_ASSERT(
                (prevtok.SuffixLen == 1 && prevtok.TokenDelim != TOKDELIM_NULL && lasttok.PrefixLen == 0) ||
                (prevtok.SuffixLen == 0 && prevtok.TokenDelim != TOKDELIM_NULL && lasttok.PrefixLen == 1) ||
                (prevtok.SuffixLen == 0 && prevtok.TokenDelim == TOKDELIM_NULL && lasttok.PrefixLen == 0 &&
                 (prevtok.Hyphen == HYPHEN_ORDINARY || prevtok.Hyphen == HYPHEN_SOFT)));
            break;
        case 3:
            if (prevtok.SuffixLen == 2 && lasttok.PrefixLen == 1) { // a++/b
                if (prevtok.TokenDelim == TOKDELIM_PLUS)            // a++/b
                    prevtok.SuffixLen = 1;
                else if (prevtok.TokenDelim == TOKDELIM_AT_SIGN) { // 'a++@b'
                    prevtok.SuffixLen = 1;
                    prevtok.TokenDelim = TOKDELIM_PLUS;
                }
            }
            Y_ASSERT(
                (prevtok.SuffixLen == 1 && prevtok.TokenDelim != TOKDELIM_NULL && lasttok.PrefixLen == 1) ||
                (prevtok.SuffixLen == 2 && prevtok.TokenDelim != TOKDELIM_NULL && lasttok.PrefixLen == 0) ||
                (prevtok.SuffixLen == 0 && prevtok.TokenDelim == TOKDELIM_NULL && lasttok.PrefixLen == 0 &&
                 (prevtok.Hyphen == HYPHEN_ORDINARY || prevtok.Hyphen == HYPHEN_SOFT)));
            break;
        case 4:
            Y_ASSERT(
                (prevtok.SuffixLen == 2 && prevtok.TokenDelim != TOKDELIM_NULL && lasttok.PrefixLen == 1) ||
                (prevtok.SuffixLen == 0 && prevtok.TokenDelim == TOKDELIM_NULL && lasttok.PrefixLen == 0 &&
                 (prevtok.Hyphen == HYPHEN_ORDINARY || prevtok.Hyphen == HYPHEN_SOFT)));
            break;
        default:
            Y_ASSERT(
                prevtok.SuffixLen == 0 && prevtok.TokenDelim == TOKDELIM_NULL && lasttok.PrefixLen == 0 &&
                (prevtok.Hyphen == HYPHEN_ORDINARY || prevtok.Hyphen == HYPHEN_SOFT));
            break;
    }
}

NLP_TYPE PrepareMultitoken(TTokenStructure& subtokens, wchar16* buffer, size_t buflen, const wchar16* entry, size_t& len) {
    size_t srcpos = subtokens[0].EndPos(); // the beginning of source delimiters and tokens
    if (srcpos > buflen - 1) {
        srcpos = buflen - 1;
        subtokens[0].Len = srcpos - subtokens[0].Pos;
    }
    size_t destpos = srcpos;
    std::char_traits<wchar16>::copy(buffer, entry, srcpos);

    TCharSpan* const firsttok = &subtokens[0];
    TCharSpan* const lasttok = firsttok + subtokens.size();
    TCharSpan* desttok = firsttok;

    // lengths of tokens are not changed, tokens can be moved only to remove delimiters

    for (const TCharSpan* srctok = firsttok + 1; srctok != lasttok; ++srctok) {
        // Check available space; truncate input if not sufficient
        if (srctok->EndPos() >= buflen) {
            break;
        }

        if (srctok->EndPos() + srctok->SuffixLen >= buflen) {
            ProcessHyphenation(buffer, entry, desttok, srctok, destpos, srcpos);

            if (desttok->EndPos() + desttok->SuffixLen >= buflen) {
                Y_ASSERT(desttok->EndPos() < buflen);
                desttok->SuffixLen = 0; // cut off the suffix

                CopyTokenText(buffer, entry, srctok->Len, destpos, srcpos); // copy with no suffix
            } else
                CopyTokenText(buffer, entry, (srctok->Len + srctok->SuffixLen), destpos, srcpos); // copy the token with suffix

            break;
        }

        ProcessHyphenation(buffer, entry, desttok, srctok, destpos, srcpos);
        CopyTokenText(buffer, entry, (srctok->Len + srctok->SuffixLen), destpos, srcpos); // copy the token with suffix
    }

    Y_ASSERT(srcpos >= destpos);

    // Multitoken->Leng
    len = destpos;
    subtokens.resize(desttok - firsttok + 1);
    subtokens.back().TokenDelim = TOKDELIM_NULL;
    return DetectNLPType(subtokens); // after PrepareMultitoken nlpType can be chagned
}

size_t AdjustSubtokens(TTokenStructure& subtokens, size_t maxLen) {
    Y_ASSERT(!subtokens.empty());

    TCharSpan* const first = &subtokens[0];
    TCharSpan* p = first + subtokens.size() - 1;
    while (p != first && p->Pos >= maxLen) {
        --p;
    }

    CutSubtoken(*p, maxLen);
    subtokens.resize(p - first + 1);
    p->TokenDelim = TOKDELIM_NULL;
    return p->EndPos() + p->SuffixLen;
}

size_t AdjustSubtokens(TTokenStructure& subtokens, size_t n, size_t maxLen) {
    Y_ASSERT(n > 0);
    Y_ASSERT(!subtokens.empty());

    // ````````````````ab``-cd```-ef```
    // ---------0---------- --1-- --2--
    //                 ^n

    TCharSpan& first = subtokens[0];
    Y_ASSERT(first.Len > n);
    first.Pos = 0;
    first.Len -= n;
    if (CutSubtoken(first, maxLen)) {
        subtokens.resize(1);
    } else {
        for (size_t i = 1; i < subtokens.size(); ++i) {
            TCharSpan& span = subtokens[i];

            Y_ASSERT(span.Pos > n);
            const size_t newPos = span.Pos - n;

            if (newPos >= maxLen) { // if no symbols left for current token - finish
                subtokens.resize(i);
                break;
            }

            span.Pos = newPos;

            if (CutSubtoken(span, maxLen)) {
                subtokens.resize(i + 1);
                break;
            }
        }
    }

    TCharSpan& last = subtokens.back();
    last.TokenDelim = TOKDELIM_NULL;
    return last.EndPos() + last.SuffixLen;
}

bool CheckMultitoken(const TWideToken& tok) {
    const TTokenStructure& subtokens = tok.SubTokens;
    for (size_t i = 0; i < subtokens.size(); ++i) {
        const TCharSpan& s = subtokens[i];
        if (!s.Len)
            Y_ASSERT(false);
        if (i > 0) {
            const TCharSpan& prev = subtokens[i - 1];
            if (prev.EndPos() + prev.SuffixLen > s.Pos - s.PrefixLen)
                Y_ASSERT(false);
            else if (prev.EndPos() + prev.SuffixLen == s.Pos - s.PrefixLen && prev.TokenDelim != TOKDELIM_NULL)
                Y_ASSERT(false);
        }
        if (i + 1 < subtokens.size()) {
            const TCharSpan& next = subtokens[i + 1];
            if (s.EndPos() + s.SuffixLen > next.Pos - next.PrefixLen)
                Y_ASSERT(false);
            else if (s.EndPos() + s.SuffixLen == next.Pos - next.PrefixLen && s.TokenDelim != TOKDELIM_NULL)
                Y_ASSERT(false);
        } else {
            if (s.EndPos() + s.SuffixLen != tok.Leng || s.TokenDelim != TOKDELIM_NULL)
                Y_ASSERT(false);
        }
    }
    return (subtokens.empty() || tok.Leng == (subtokens.back().EndPos() + subtokens.back().SuffixLen));
}
