#pragma once

#include <library/cpp/token/nlptypes.h>
#include <library/cpp/token/token_structure.h>

void CorrectDelimiters(TCharSpan& prevtok, wchar16 suffixChar, TCharSpan& lasttok, wchar16 prefixChar);

//! removes hyphenations and replaces unicode delimiters
//! @return new length of multitoken
NLP_TYPE PrepareMultitoken(TTokenStructure& subtokens, wchar16* buffer, size_t buflen, const wchar16* entry, size_t& len);

//! cuts off the subtokens according to the specified maximum length
//! @return new length of the subtokens
size_t AdjustSubtokens(TTokenStructure& subtokens, size_t maxLen);

//! corrects positions of subtokens and cuts off their length according to the specified maximum
//! @note the first @c n characters are accents
//! @return new length of the subtokens
size_t AdjustSubtokens(TTokenStructure& subtokens, size_t n, size_t maxLen);

//! for debugging purposes only
bool CheckMultitoken(const TWideToken& tok);
