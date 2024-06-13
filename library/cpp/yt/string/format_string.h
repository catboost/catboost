#pragma once

#include "format_analyser.h"

#include <util/generic/strbuf.h>

#if (!__clang__ || __clang_major__ < 16)
    #define YT_DISABLE_FORMAT_STATIC_ANALYSIS
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

// Explicitly create TRuntimeFormat if you wish to
// use runtime/non-literal value as format.
class TRuntimeFormat
{
public:
    explicit TRuntimeFormat(TStringBuf fmt);

    TStringBuf Get() const noexcept;

private:
    TStringBuf Format_;
};

// This class used to properly bind to
// string literals and allow compile-time parsing/checking
// of those. If you need a runtime format, use TRuntimeFormat
template <class... TArgs>
class TBasicFormatString
{
public:
    // Can be used to perform compile-time check of format.
    template <class T>
        requires std::constructible_from<std::string_view, T>
    consteval TBasicFormatString(const T& fmt);

    TBasicFormatString(TRuntimeFormat fmt);

    TStringBuf Get() const noexcept;

    static consteval void CheckFormattability();

private:
    std::string_view Format_;

    template <class T>
    static void CrashCompilerClassIsNotFormattable();
};

// Used to properly infer template arguments if Format.
template <class... TArgs>
using TFormatString = TBasicFormatString<std::type_identity_t<TArgs>...>;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define FORMAT_STRING_INL_H_
#include "format_string-inl.h"
#undef FORMAT_STRING_INL_H_
