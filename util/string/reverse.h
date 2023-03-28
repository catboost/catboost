#pragma once

#include <util/generic/fwd.h>

void ReverseInPlace(TString& string);

/** NB. UTF-16 is variable-length encoding because of the surrogate pairs.
 * This function takes this into account and treats a surrogate pair as a single symbol.
 * Ex. if [C D] is a surrogate pair,
 * A B [C D] E
 * will become
 * E [C D] B A
 */
void ReverseInPlace(TUtf16String& string);

void ReverseInPlace(TUtf32String& string);
