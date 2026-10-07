#pragma once

#include <version>

#if __cplusplus >= 202002L && defined(__cpp_lib_concepts)

// clang-format off
#include <concepts>
#include <ranges>
// clang-format on

template <typename T, typename TOf>
concept CInputRangeOf = std::ranges::input_range<T> && std::same_as<std::ranges::range_value_t<T>, TOf>;

#endif
