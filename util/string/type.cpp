#include "type.h"
#include "ascii.h"

#include <array>

bool IsSpace(const char* s, size_t len) noexcept {
    if (len == 0)
        return false;
    for (const char* p = s; p < s + len; ++p)
        if (!IsAsciiSpace(*p))
            return false;
    return true;
}

template <typename TStringType>
static bool IsNumberT(const TStringType& s) noexcept {
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

template <typename TStringType>
static bool IsHexNumberT(const TStringType& s) noexcept {
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

namespace {
    template <size_t N>
    bool IsCaseInsensitiveAnyOf(TStringBuf str, const std::array<TStringBuf, N>& options) {
        for (auto option : options) {
            if (str.size() == option.size() && ::strnicmp(str.data(), option.data(), str.size()) == 0) {
                return true;
            }
        }
        return false;
    }
} //anonymous namespace

bool IsTrue(const TStringBuf v) noexcept {
    static constexpr std::array<TStringBuf, 7> trueOptions{
        AsStringBuf("true"),
        AsStringBuf("t"),
        AsStringBuf("yes"),
        AsStringBuf("y"),
        AsStringBuf("on"),
        AsStringBuf("1"),
        AsStringBuf("da")};
    return IsCaseInsensitiveAnyOf(v, trueOptions);
}

bool IsFalse(const TStringBuf v) noexcept {
    static constexpr std::array<TStringBuf, 7> falseOptions{
        AsStringBuf("false"),
        AsStringBuf("f"),
        AsStringBuf("no"),
		AsStringBuf("n"),
        AsStringBuf("off"),
        AsStringBuf("0"),
        AsStringBuf("net")};
    return IsCaseInsensitiveAnyOf(v, falseOptions);
}
