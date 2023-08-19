#pragma once

#include "preprocessor.h"

#include <util/generic/strbuf.h>

#include <array>
#include <optional>
#include <type_traits>
#include <vector>

#include <library/cpp/yt/exception/exception.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////
/*
 * Smart enumerations augment C++ enum classes with a bunch of reflection
 * capabilities accessible via TEnumTraits class specialization.
 *
 * Please refer to the unit test for an actual example of usage
 * (unittests/enum_ut.cpp).
 */

// The actual overload must be provided with DEFINE_ENUM* (see below).
template <class T>
void GetEnumTraitsImpl(T);

template <class T>
using TEnumTraitsImpl = decltype(GetEnumTraitsImpl(T()));

template <class T>
constexpr bool IsEnumDomainSizeKnown()
{
    if constexpr(requires{ TEnumTraitsImpl<T>::DomainSize; }) {
        return true;
    } else {
        return false;
    }
}

template <
    class T,
    bool = IsEnumDomainSizeKnown<T>()
>
struct TEnumTraitsWithKnownDomain
{ };

template <
    class T,
    bool = std::is_enum_v<T> && !std::is_same_v<TEnumTraitsImpl<T>, void>
>
struct TEnumTraits
{
    static constexpr bool IsEnum = false;
    static constexpr bool IsBitEnum = false;
    static constexpr bool IsStringSerializableEnum = false;
};

template <class T>
struct TEnumTraitsWithKnownDomain<T, true>
{
    static constexpr int GetDomainSize();

    static constexpr const std::array<TStringBuf, GetDomainSize()>& GetDomainNames();
    static constexpr const std::array<T, GetDomainSize()>& GetDomainValues();

    // For non-bit enums only.
    static constexpr T GetMinValue()
        requires (!TEnumTraitsImpl<T>::IsBitEnum);
    static constexpr T GetMaxValue()
        requires (!TEnumTraitsImpl<T>::IsBitEnum);

    // For bit enums only.
    static std::vector<T> Decompose(T value)
        requires (TEnumTraitsImpl<T>::IsBitEnum);
};

template <class T>
struct TEnumTraits<T, true>
    : public TEnumTraitsWithKnownDomain<T>
{
    static constexpr bool IsEnum = true;
    static constexpr bool IsBitEnum = TEnumTraitsImpl<T>::IsBitEnum;
    static constexpr bool IsStringSerializableEnum = TEnumTraitsImpl<T>::IsStringSerializableEnum;

    static TStringBuf GetTypeName();

    static std::optional<TStringBuf> FindLiteralByValue(T value);
    static std::optional<T> FindValueByLiteral(TStringBuf literal);

    static TString ToString(T value);
    static T FromString(TStringBuf literal);
};

////////////////////////////////////////////////////////////////////////////////

//! Defines a smart enumeration with a specific underlying type.
/*!
 * \param enumType Enumeration enumType.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_ENUM_WITH_UNDERLYING_TYPE(enumType, underlyingType, seq) \
    ENUM__CLASS(enumType, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(enumType, underlyingType, false, false, seq) \
    ENUM__VALIDATE_UNIQUE(enumType) \
    ENUM__END_TRAITS(enumType) \
    static_assert(true)

//! Defines a smart enumeration with a specific underlying type.
//! Duplicate enumeration values are allowed.
#define DEFINE_AMBIGUOUS_ENUM_WITH_UNDERLYING_TYPE(enumType, underlyingType, seq) \
    ENUM__CLASS(enumType, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(enumType, underlyingType, false, false, seq) \
    ENUM__END_TRAITS(enumType) \
    static_assert(true)

//! Defines a smart enumeration with the default |int| underlying type.
#define DEFINE_ENUM(enumType, seq) \
    DEFINE_ENUM_WITH_UNDERLYING_TYPE(enumType, int, seq)

//! Defines a smart enumeration with a specific underlying type.
/*!
 * \param enumType Enumeration enumType.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_BIT_ENUM_WITH_UNDERLYING_TYPE(enumType, underlyingType, seq) \
    ENUM__CLASS(enumType, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(enumType, underlyingType, true, false, seq) \
    ENUM__VALIDATE_UNIQUE(enumType) \
    ENUM__END_TRAITS(enumType) \
    ENUM__BITWISE_OPS(enumType) \
    static_assert(true)

//! Defines a smart enumeration with a specific underlying type.
//! Duplicate enumeration values are allowed.
/*!
 * \param enumType Enumeration enumType.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_AMBIGUOUS_BIT_ENUM_WITH_UNDERLYING_TYPE(enumType, underlyingType, seq) \
    ENUM__CLASS(enumType, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(enumType, underlyingType, true, false, seq) \
    ENUM__END_TRAITS(enumType) \
    ENUM__BITWISE_OPS(enumType) \
    static_assert(true)

//! Defines a smart enumeration with the default |unsigned int| underlying type.
/*!
 * \param enumType Enumeration enumType.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 */
#define DEFINE_BIT_ENUM(enumType, seq) \
    DEFINE_BIT_ENUM_WITH_UNDERLYING_TYPE(enumType, unsigned int, seq)

//! Defines a smart enumeration with a specific underlying type and IsStringSerializable attribute.
/*!
 * \param enumType Enumeration enumType.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_STRING_SERIALIZABLE_ENUM_WITH_UNDERLYING_TYPE(enumType, underlyingType, seq) \
    ENUM__CLASS(enumType, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(enumType, underlyingType, false, true, seq) \
    ENUM__VALIDATE_UNIQUE(enumType) \
    ENUM__END_TRAITS(enumType) \
    static_assert(true)

//! Defines a smart enumeration with a specific underlying type and IsStringSerializable attribute.
//! Duplicate enumeration values are allowed.
#define DEFINE_AMBIGUOUS_STRING_SERIALIZABLE_ENUM_WITH_UNDERLYING_TYPE(enumType, underlyingType, seq) \
    ENUM__CLASS(enumType, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(enumType, underlyingType, false, true, seq) \
    ENUM__END_TRAITS(enumType) \
    static_assert(true)

//! Defines a smart enumeration with the default |int| underlying type and IsStringSerializable attribute.
#define DEFINE_STRING_SERIALIZABLE_ENUM(enumType, seq) \
    DEFINE_STRING_SERIALIZABLE_ENUM_WITH_UNDERLYING_TYPE(enumType, int, seq)

////////////////////////////////////////////////////////////////////////////////

//! A statically sized vector with elements of type |T| indexed by
//! the items of enumeration type |E|.
/*!
 *  Items are value-initialized on construction.
 */
template <
    class E,
    class T,
    E Min = TEnumTraits<E>::GetMinValue(),
    E Max = TEnumTraits<E>::GetMaxValue()
>
class TEnumIndexedVector
{
public:
    using TIndex = E;
    using TValue = T;

    constexpr TEnumIndexedVector();
    constexpr TEnumIndexedVector(std::initializer_list<T> elements);

    constexpr TEnumIndexedVector(const TEnumIndexedVector&) = default;
    constexpr TEnumIndexedVector(TEnumIndexedVector&&) noexcept = default;

    constexpr TEnumIndexedVector& operator=(const TEnumIndexedVector&) = default;
    constexpr TEnumIndexedVector& operator=(TEnumIndexedVector&&) noexcept = default;

    T& operator[] (E index);
    const T& operator[] (E index) const;

    // STL interop.
    T* begin();
    const T* begin() const;
    T* end();
    const T* end() const;

    static bool IsDomainValue(E value);

private:
    using TUnderlying = std::underlying_type_t<E>;
    static constexpr int N = static_cast<TUnderlying>(Max) - static_cast<TUnderlying>(Min) + 1;
    std::array<T, N> Items_;
};

////////////////////////////////////////////////////////////////////////////////

//! Returns |true| iff the enumeration value is not bitwise zero.
template <typename E>
    requires TEnumTraits<E>::IsBitEnum
constexpr bool Any(E value) noexcept;

//! Returns |true| iff the enumeration value is bitwise zero.
template <typename E>
    requires TEnumTraits<E>::IsBitEnum
constexpr bool None(E value) noexcept;

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ENUM_INL_H_
#include "enum-inl.h"
#undef ENUM_INL_H_
