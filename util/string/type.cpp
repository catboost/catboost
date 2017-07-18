#include "type.h"
#include "ascii.h"

bool IsSpace(const char* s, size_t len) noexcept {
    if (len == 0)
        return false;
    for (const char* p = s; p < s + len; ++p)
        if (!IsAsciiSpace(*p))
            return false;
    return true;
}

template <typename TStroka>
static bool IsNumberT(const TStroka& s) noexcept {
    if (s.empty()) {
        return false;
    }

    for (auto ch : s) {
        if (!IsAsciiDigit(ch)) {
            return false;
        }
    }
    return true;
}

bool IsNumber(const TStringBuf s) noexcept {
    return IsNumberT(s);
}

bool IsNumber(const TWtringBuf s) noexcept {
    return IsNumberT(s);
}

template <typename TStroka>
static bool IsHexNumberT(const TStroka& s) noexcept {
    if (s.empty()) {
        return false;
    }

    for (auto ch : s) {
        if (!IsAsciiHex(ch)) {
            return false;
        }
    }

    return true;
}

bool IsHexNumber(const TStringBuf s) noexcept {
    return IsHexNumberT(s);
}

bool IsHexNumber(const TWtringBuf s) noexcept {
    return IsHexNumberT(s);
}

bool IsTrue(const TStringBuf v) {
    if (!v)
        return false;

    return !strnicmp(~v, "da", v.length()) || !strnicmp(~v, "yes", v.length()) || !strnicmp(~v, "on", v.length()) || !strnicmp(~v, "1", v.length()) || !strnicmp(~v, "true", v.length());
}

bool IsFalse(const TStringBuf v) {
    if (!v)
        return false;

    return !strnicmp(~v, "net", v.length()) || !strnicmp(~v, "no", v.length()) || !strnicmp(~v, "off", v.length()) || !strnicmp(~v, "0", v.length()) || !strnicmp(~v, "false", v.length());
}
