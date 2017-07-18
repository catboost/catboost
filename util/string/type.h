#pragma once

#include <util/generic/strbuf.h>

bool IsSpace(const char* s, size_t len) noexcept;

/// Checks if a string is a set of only space symbols.
static inline bool IsSpace(const TStringBuf s) noexcept {
    return IsSpace(~s, +s);
}

/// Returns "true" if the given string is an arabic number ([0-9]+)
bool IsNumber(const TStringBuf s) noexcept;
bool IsNumber(const TWtringBuf s) noexcept;

/// Returns "true" if the given string is a hex number ([0-9a-fA-F]+)
bool IsHexNumber(const TStringBuf s) noexcept;
bool IsHexNumber(const TWtringBuf s) noexcept;

/// Returns "true" if the given sting is equal to one of "yes", "on", "1", "true", "da".
/// @details case-insensitive.
bool IsTrue(const TStringBuf value);

/// Returns "false" if the given sting is equal to one of "no", "off", "0", "false", "net"
/// @details case-insensitive.
bool IsFalse(const TStringBuf value);
