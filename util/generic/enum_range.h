#pragma once

#include <type_traits>

// This tiny header is to define value ranges at compile time in enums and enum class/struct types.
//
// enum class E1 {
//     A,
//     B,
//     C,
// };
// Y_DEFINE_ENUM_MINMAX(E1, A, C);
//
//     or
//
// enum class E2 {
//     A,
//     B,
//     C,
// };
// Y_DEFINE_ENUM_MAX(E2, C);
//
//     Notes:
//         * use Y_DEFINE_ENUM_MINMAX / Y_DEFINE_ENUM_MINMAX if your enum is defined in a namespace
//         * use Y_DEFINE_ENUM_MINMAX_F / Y_DEFINE_ENUM_MINMAX_F if your enum is defined in a class/struct
//         * use shortened version Y_DEFINE_ENUM_MAX / Y_DEFINE_ENUM_MAX_F if enum begin is 0
//         * add Y_DEFINE_xxx macro immediately after enum definition
//
//     Usage examples:
//         TEnumRange<E>::Min               // min value of range in enum type
//         TEnumRange<E>::Max               // max value of range in enum type
//         TEnumRange<E>::UnderlyingMin     // min value of range in underlying type
//         TEnumRange<E>::UnderlyingMax     // max value of range in underlying type

void _YRegisterEnumRange(...);

namespace NDetail::NEnumRange {

    template <typename E, E _Min, E _Max>
    struct TEnumRange {
        static_assert(std::is_enum_v<E>, "");

        using TEnum = E;
        using TUnderlying = std::underlying_type_t<TEnum>;

        static constexpr TEnum Min = _Min;
        static constexpr TEnum Max = _Max;

        static constexpr TUnderlying UnderlyingMin = static_cast<TUnderlying>(Min);
        static constexpr TUnderlying UnderlyingMax = static_cast<TUnderlying>(Max);

        static_assert(UnderlyingMin <= UnderlyingMax, "Invalid enum range");
    };
}

#define Y_DEFINE_ENUM_MINMAX_IMPL(PREFIX, E, Min, Max) \
    PREFIX ::NDetail::NEnumRange::TEnumRange<E, Min, Max> _YRegisterEnumRange(const E*)

#define Y_DEFINE_ENUM_MINMAX(E, Min, Max) \
    Y_DEFINE_ENUM_MINMAX_IMPL([[maybe_unused]], E, E::Min, E::Max)

#define Y_DEFINE_ENUM_MAX(E, Max) \
    Y_DEFINE_ENUM_MINMAX_IMPL([[maybe_unused]], E, E{}, E::Max)

#define Y_DEFINE_ENUM_MINMAX_FRIEND(E, Min, Max) \
    Y_DEFINE_ENUM_MINMAX_IMPL(friend, E, E::Min, E::Max)

#define Y_DEFINE_ENUM_MAX_FRIEND(E, Max) \
    Y_DEFINE_ENUM_MINMAX_IMPL(friend, E, E{}, E::Max)

template <typename E>
using TEnumRange = decltype(_YRegisterEnumRange(static_cast<E*>(nullptr)));
