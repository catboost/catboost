#pragma once

// Author: smikler@yandex-team.ru

#include <util/generic/string.h>

/* Converts pcre regular expression to pire compatible format:
 *   - replaces "\\#" with "#"
 *   - replaces "\\=" with "="
 *   - replaces "\\:" with ":"
 *   - removes "?P<...>"
 *   - removes "?:"
 *   - removes "()" recursively
 *   - replaces "??" with "?"
 *   - replaces "*?" with "*"
 * NOTE:
 *   - Not fully tested!
 */
TString Pcre2Pire(const TString& src);
