#pragma once

#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/str_stl.h>

/*
 * Zero-terminated string view.
 *
 * Has a c_str() for use with system/cstdlib calls (like TString)
 * but can be constructed from a string literal or command-line arg
 * without memory allocation (like TStringBuf).
 *
 * Use it to reference filenames, thread names, string formats etc.
 */

class TZtStringBuf: public TStringBuf {
public:
    constexpr TZtStringBuf(const char* s Y_LIFETIME_BOUND) noexcept
        : TStringBuf(s)
    {
    }

    TZtStringBuf(const TString& s Y_LIFETIME_BOUND) noexcept
        : TStringBuf(s)
    {
    }

    TZtStringBuf(const std::string& s Y_LIFETIME_BOUND) noexcept
        : TStringBuf(s)
    {
    }

    constexpr TZtStringBuf() noexcept
        : TZtStringBuf("")
    {
    }

    TZtStringBuf(const TStringBuf&) = delete;

    constexpr const char* c_str() const noexcept {
        return data();
    }
};

template <>
struct THash<TZtStringBuf> : public THash<TStringBuf> {
};
