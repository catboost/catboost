#ifndef FORMAT_STRING_INL_H_
#error "Direct inclusion of this file is not allowed, include format_string.h"
// For the sake of sane code completion.
#include "format_string.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class... TArgs>
template <class T>
    requires std::constructible_from<std::string_view, T>
consteval TBasicStaticFormat<TArgs...>::TBasicStaticFormat(const T& fmt)
    : Format_(fmt)
{
    CheckFormattability();
#if !defined(NDEBUG) && !defined(YT_DISABLE_FORMAT_STATIC_ANALYSIS)
    NDetail::TFormatAnalyser::ValidateFormat<std::remove_cvref_t<TArgs>...>(Format_);
#endif
}

template <class... TArgs>
TStringBuf TBasicStaticFormat<TArgs...>::Get() const noexcept
{
    return {Format_};
}

template <class... TArgs>
consteval void TBasicStaticFormat<TArgs...>::CheckFormattability()
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

inline TRuntimeFormat::TRuntimeFormat(TStringBuf fmt)
    : Format_(fmt)
{ }

inline TStringBuf TRuntimeFormat::Get() const noexcept
{
    return Format_;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
