#pragma once

#include <util/generic/fwd.h>

#include <cstddef>

/*

A helper include file.

One should not include it directly. It will be included implicitly when you add

    GENERATE_ENUM_SERIALIZATION_WITH_HEADER(your_header_with_your_enum.h)

in your CMakeLists.txt

@see https://st.yandex-team.ru/IGNIETFERRO-333
@see https://wiki.yandex-team.ru/PoiskovajaPlatforma/Build/WritingCmakefiles/#generate-enum-with-header

*/

/**
 * Returns number of distinct items in enum or enum class
 *
 * @tparam EnumT     enum type
 */
template <typename EnumT>
constexpr size_t GetEnumItemsCount();

/**
 * Returns names for items in enum or enum class
 *
 * @tparam EnumT     enum type
 */
template <typename EnumT>
const ymap<EnumT, TString>& GetEnumNames();
