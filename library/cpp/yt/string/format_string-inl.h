#ifndef FORMAT_STRING_INL_H_
#error "Direct inclusion of this file is not allowed, include format_string.h"
// For the sake of sane code completion.
#include "format_string.h"
#endif

#include <algorithm>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class... TArgs>
template <class T>
    requires std::constructible_from<std::string_view, T>
consteval TBasicFormatString<TArgs...>::TBasicFormatString(const T& fmt)
    : Format_(fmt)
{
    CheckFormattability();
#if !defined(YT_DISABLE_FORMAT_STATIC_ANALYSIS)
    std::tie(Markers, Escapes) = NDetail::TFormatAnalyser::AnalyzeFormat<std::remove_cvref_t<TArgs>...>(Format_);
#else
    std::ranges::fill_n(std::ranges::begin(Escapes), 1, -1);
    if constexpr (sizeof...(TArgs) != 0) {
        std::ranges::fill_n(std::ranges::begin(Markers), 1, std::tuple{0, 0});
    }
#endif
}

template <class... TArgs>
TStringBuf TBasicFormatString<TArgs...>::Get() const noexcept
{
    return {Format_};
}

template <class... TArgs>
consteval void TBasicFormatString<TArgs...>::CheckFormattability()
{
#if !defined(NDEBUG) && !defined(YT_DISABLE_FORMAT_STATIC_ANALYSIS)
    using TTuple = std::tuple<std::remove_cvref_t<TArgs>...>;

    [] <size_t... Idx> (std::index_sequence<Idx...>) {
        ([] {
            if constexpr (!CFormattable<std::tuple_element_t<Idx, TTuple>>) {
                CrashCompilerClassIsNotFormattable<std::tuple_element_t<Idx, TTuple>>();
            }
        } (), ...);
    } (std::index_sequence_for<TArgs...>());
#endif
}

template <class... TArgs>
TBasicFormatString<TArgs...>::TBasicFormatString(TRuntimeFormat fmt)
    : Format_(fmt.Get())
{
    std::ranges::fill_n(std::ranges::begin(Escapes), 1, -1);
    if constexpr (sizeof...(TArgs) != 0) {
        std::ranges::fill_n(std::ranges::begin(Markers), 1, std::tuple{0, 0});
    }

    // NB(arkady-e1ppa): StaticFormat performs the
    // formattability check of the args in a way
    // that provides more useful information
    // than a simple static_assert with conjunction.
    // Additionally, the latter doesn't work properly
    // for older clang version.
    static constexpr auto argsChecker = [] {
        CheckFormattability();
        return 42;
    } ();
    Y_UNUSED(argsChecker);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
