#pragma once

#include <util/generic/strbuf.h>

#include <array>
#include <string_view>

namespace NYT::NDetail {

////////////////////////////////////////////////////////////////////////////////

struct TFormatAnalyser
{
public:
    template <class... TArgs>
    static consteval void ValidateFormat(std::string_view fmt);

private:
    // Non-constexpr function call will terminate compilation.
    // Purposefully undefined and non-constexpr/consteval
    template <class T>
    static void CrashCompilerNotFormattable(std::string_view /*msg*/)
    { /*Suppress "internal linkage but undefined" warning*/ }
    static void CrashCompilerNotEnoughArguments(std::string_view msg);
    static void CrashCompilerTooManyArguments(std::string_view msg);
    static void CrashCompilerWrongTermination(std::string_view msg);
    static void CrashCompilerMissingTermination(std::string_view msg);
    static void CrashCompilerWrongFlagSpecifier(std::string_view msg);

    struct TSpecifiers
    {
        std::string_view Conversion;
        std::string_view Flags;
    };
    template <class TArg>
    static consteval auto GetSpecifiers();

    static constexpr char IntroductorySymbol = '%';
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NDetail

#define FORMAT_ANALYSER_INL_H_
#include "format_analyser-inl.h"
#undef FORMAT_ANALYSER_INL_H_
