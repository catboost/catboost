#pragma once

#include <util/generic/cast.h>
#include <util/generic/serialized_enum.h>

#include <optional>
#include <type_traits>
#include <typeinfo>

namespace NPrivate {

    [[noreturn]] void OnSafeCastToEnumUnexpectedValue(const std::type_info& valueTypeInfo);

} // namespace NPrivate

/**
 * Safely cast an integer value to the enum value.
 * @throw yexception is case of unknown enum underlying type value
 *
 * @tparam TEnum     enum type
 */
template <typename TEnum, typename TInteger, typename = std::enable_if_t<std::is_enum_v<TEnum>>>
TEnum SafeCastToEnum(TInteger integerValue) {
    using TUnderlyingEnumType = std::underlying_type_t<TEnum>;

    std::optional<TUnderlyingEnumType> value;
    try {
        value = SafeIntegerCast<TUnderlyingEnumType>(integerValue);
    } catch (const TBadCastException&) {
        // SafeIntegerCast throws TBadCastException when TInteger cannot be cast
        // to TUnderlyingEnumType but the exception message is about integer
        // value cast being unsafe.
        // SafeCastToEnum must throw TBadCastException with its own exception
        // message even if integer cast fails.
    }

    if (value.has_value()) {
        for (TEnum enumValue : GetEnumAllValues<TEnum>()) {
            if (static_cast<TUnderlyingEnumType>(enumValue) == *value) {
                return enumValue;
            }
        }
    }

    NPrivate::OnSafeCastToEnumUnexpectedValue(typeid(TEnum));
}
