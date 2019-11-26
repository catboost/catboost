#include "token_util.h"

#include <util/charset/unidata.h>

TUtf16String RemoveWideTokenPrefix(TWideToken& token) {
    const size_t prefixLen = token.SubTokens[0].PrefixLen;
    TUtf16String res(token.Token, prefixLen);
    token.Token += prefixLen;
    token.Leng -= prefixLen;
    token.SubTokens[0].PrefixLen = 0;
    for (auto& subToken : token.SubTokens) {
        subToken.Pos -= prefixLen;
    }
    return res;
}

TUtf16String RemoveWideTokenSuffix(TWideToken& token) {
    const size_t suffixLen = token.SubTokens.back().SuffixLen;
    TUtf16String res(token.Token + token.SubTokens.back().EndPos(), suffixLen);
    token.Leng -= suffixLen;
    token.SubTokens.back().SuffixLen = 0;
    return res;
}

bool CheckWideTokenSplit(const TWideToken& token, size_t pos) {
    Y_ASSERT(pos < token.SubTokens.size() - 1);

    const TCharSpan& subtoken = token.SubTokens[pos];
    const TCharSpan& subtokenNext = token.SubTokens[pos + 1];

    return (subtoken.Type != subtokenNext.Type) || (subtoken.Type != TOKEN_WORD) || ((subtoken.TokenDelim != TOKDELIM_APOSTROPHE) && (subtoken.TokenDelim != TOKDELIM_MINUS));
}

bool CheckWideTokenDotSplit(const TWideToken& token, size_t pos) {
    Y_ASSERT(pos < token.SubTokens.size() - 1);

    const TCharSpan& token1 = token.SubTokens[pos];
    const TCharSpan& token2 = token.SubTokens[pos + 1];

    if (token1.TokenDelim != TOKDELIM_DOT) {
        return false;
    }

    if ((token1.Type == TOKEN_WORD || (token1.Type == TOKEN_NUMBER && pos == 0)) && token2.Type == TOKEN_WORD && (::IsUpper(token.Token[token2.Pos]) || ::IsTitle(token.Token[token2.Pos]))) {
        return true;
    }

    return token1.Type == TOKEN_WORD && token2.Type == TOKEN_NUMBER;
}

// Check if we can split wide-token after specified sub-token.
// The function uses rich-tree specific heuristics
bool CheckWideTokenReqSplit(const TTokenStructure& subtokens, size_t pos) {
    const size_t last = subtokens.size() - 1;
    Y_ASSERT(pos < last);
    const TCharSpan& s = subtokens[pos];

    if (s.TokenDelim == TOKDELIM_NULL) {
        if (pos < (last - 1) && subtokens[pos + 1].Type == TOKEN_NUMBER && subtokens[pos + 1].TokenDelim == TOKDELIM_DOT && subtokens[pos + 2].Type == TOKEN_NUMBER)
            return true; // v2.0 -> v /+1 2.0

        if (pos == 0 || s.Type != TOKEN_NUMBER || subtokens[pos - 1].TokenDelim != TOKDELIM_DOT || subtokens[pos - 1].Type != TOKEN_NUMBER)
            return false; // the current token is a part of a mark, the current token '2': 1-2a
    }

    if (s.Type == TOKEN_NUMBER && s.TokenDelim == TOKDELIM_DOT && subtokens[pos + 1].Type == TOKEN_NUMBER)
        return false; // the current token is a part of a number sequence

    if (s.TokenDelim != TOKDELIM_APOSTROPHE && s.TokenDelim != TOKDELIM_MINUS)
        return true; // baden-baden, caffrey's

    if (s.Type == TOKEN_NUMBER)
        return true; // the current token is number

    if (s.Type != subtokens[pos + 1].Type)
        return true; // types of tokens are different

    if (pos > 0 && subtokens[pos - 1].TokenDelim == TOKDELIM_NULL)
        return true; // the current token 'a' and the previous token '2' has no delimiter: 2a-b

    return (pos < (last - 1) && subtokens[pos + 1].TokenDelim == TOKDELIM_NULL); // mark follows the current token 'a': a-b2
}

TWideToken ExtractWideTokenRange(const TWideToken& tok, size_t start, size_t end) {
    Y_ASSERT(start < tok.SubTokens.size());
    Y_ASSERT(end < tok.SubTokens.size());

    TWideToken newToken;
    const size_t offset = GetSubTokenOffset(tok, start);
    newToken.Token = tok.Token + offset;
    newToken.Leng = tok.SubTokens[end].EndPos() + tok.SubTokens[end].SuffixLen - offset;
    for (size_t j = start; j <= end; ++j) {
        newToken.SubTokens.push_back(tok.SubTokens[j]);
        newToken.SubTokens.back().Pos -= offset;
    }
    return newToken;
}
