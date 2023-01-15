#pragma once

#include <util/generic/strbuf.h>

Y_PURE_FUNCTION bool IsSpace(const char* s, size_t len) noexcept;

/// Checks if a string is a set of only space symbols.
Y_PURE_FUNCTION static inline bool IsSpace(const TStringBuf s) noexcept {
    return IsSpace(s.data(), s.size());
}

/// Returns "true" if the given string is an arabic number ([0-9]+)
Y_PURE_FUNCTION bool IsNumber(const TStringBuf s) noexcept;

Y_PURE_FUNCTION bool IsNumber(const TWtringBuf s) noexcept;

/// Returns "true" if the given string is a hex number ([0-9a-fA-F]+)
Y_PURE_FUNCTION bool IsHexNumber(const TStringBuf s) noexcept;

Y_PURE_FUNCTION bool IsHexNumber(const TWtringBuf s) noexcept;

/* Tests if the given string is case insensitive equal to one of:
 * - "true",
 * - "t",
 * - "yes",
 * - "y",
 * - "on",
 * - "1",
 * - "da".
 */
Y_PURE_FUNCTION bool IsTrue(const TStringBuf value) noexcept;

/* Tests if the given string is case insensitive equal to one of:
 *  - "false",
 *  - "f",
 *  - "no",
 *  - "n",
 *  - "off",
 *  - "0",
 *  - "net".
 */
Y_PURE_FUNCTION bool IsFalse(const TStringBuf value) noexcept;
