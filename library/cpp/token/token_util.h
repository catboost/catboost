#pragma once

#include "token_structure.h"

#include <util/system/yassert.h>
#include <util/generic/string.h>

TUtf16String RemoveWideTokenPrefix(TWideToken& token);
TUtf16String RemoveWideTokenSuffix(TWideToken& token);

// Check if we can split wide-token after specified sub-token.
// The function does't allow to split on dash and apostrophe betwenn normal words
bool CheckWideTokenSplit(const TWideToken& token, size_t pos);
// Check if we can split wide-token after specified sub-token by dot delimiter.
// The function verifies the following condition:
// <word> <dot> <word with uppercased first character or number>
// <number> <dot> <word with uppercased first character>
bool CheckWideTokenDotSplit(const TWideToken& token, size_t pos);

// Check if we can split wide-token after specified sub-token.
// The function uses rich-tree specific heuristics
bool CheckWideTokenReqSplit(const TTokenStructure& subtokens, size_t pos);

inline size_t GetSubTokenOffset(const TWideToken& tok, size_t subToken) {
    Y_ASSERT(subToken < tok.SubTokens.size());
    return tok.SubTokens[subToken].Pos - tok.SubTokens[subToken].PrefixLen;
}

// Create a new wide-token with the specified inclusive [start, end] sub-token range
TWideToken ExtractWideTokenRange(const TWideToken& tok, size_t start, size_t end);
