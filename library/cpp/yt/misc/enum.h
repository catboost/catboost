#pragma once

#include "preprocessor.h"

#include <util/generic/strbuf.h>

#include <stdexcept>
#include <type_traits>
#include <array>
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

// Actual overload must be provided with defines DEFINE_ENUM_XXX (see below).
template <class T>
void GetEnumTraitsImpl(T);

template <
    class T,
    bool = std::is_enum<T>::value &&
        !std::is_convertible<T, int>::value &&
        !std::is_same<void, decltype(GetEnumTraitsImpl(T()))>::value
>
struct TEnumTraits
{
    static constexpr bool IsEnum = false;
    static constexpr bool IsBitEnum = false;
    static constexpr bool IsStringSerializableEnum = false;
};

template <class T>
struct TEnumTraits<T, true>
{
    using TImpl = decltype(GetEnumTraitsImpl(T()));
    using TType = T;
    using TUnderlying = typename TImpl::TUnderlying;

    static constexpr bool IsEnum = true;
    static constexpr bool IsBitEnum = TImpl::IsBitEnum;
    static constexpr bool IsStringSerializableEnum = TImpl::IsStringSerializableEnum;

    static constexpr int DomainSize = TImpl::DomainSize;

    static TStringBuf GetTypeName();

    static const TStringBuf* FindLiteralByValue(TType value);
    static bool FindValueByLiteral(TStringBuf literal, TType* result);

    static const std::array<TStringBuf, DomainSize>& GetDomainNames();
    static const std::array<TType, DomainSize>& GetDomainValues();

    static TType FromString(TStringBuf str);
    static TString ToString(TType value);

    // For non-bit enums only.
    static constexpr TType GetMinValue();
    static constexpr TType GetMaxValue();

    // For bit enums only.
    static std::vector<TType> Decompose(TType value);

    // LLVM SmallDenseMap interop.
    // This should only be used for enums whose underlying type has big enough range
    // (see getEmptyKey and getTombstoneKey functions).
    struct TDenseMapInfo
    {
        static inline TType getEmptyKey()
        {
            return static_cast<TType>(-1);
        }

        static inline TType getTombstoneKey()
        {
            return static_cast<TType>(-2);
        }

        static unsigned getHashValue(const TType& key)
        {
            return static_cast<unsigned>(key) * 37U;
        }

        static bool isEqual(const TType& lhs, const TType& rhs)
        {
            return lhs == rhs;
        }
    };
};

////////////////////////////////////////////////////////////////////////////////

//! Defines a smart enumeration with a specific underlying type.
/*!
 * \param name Enumeration name.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_ENUM_WITH_UNDERLYING_TYPE(name, underlyingType, seq) \
    ENUM__CLASS(name, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(name, underlyingType, false, false, seq) \
    ENUM__MINMAX \
    ENUM__VALIDATE_UNIQUE(name) \
    ENUM__END_TRAITS(name)

//! Defines a smart enumeration with a specific underlying type.
//! Duplicate enumeration values are allowed.
#define DEFINE_AMBIGUOUS_ENUM_WITH_UNDERLYING_TYPE(name, underlyingType, seq) \
    ENUM__CLASS(name, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(name, underlyingType, false, false, seq) \
    ENUM__MINMAX \
    ENUM__END_TRAITS(name)

//! Defines a smart enumeration with the default |int| underlying type.
#define DEFINE_ENUM(name, seq) \
    DEFINE_ENUM_WITH_UNDERLYING_TYPE(name, int, seq)

//! Defines a smart enumeration with a specific underlying type.
/*!
 * \param name Enumeration name.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_BIT_ENUM_WITH_UNDERLYING_TYPE(name, underlyingType, seq) \
    ENUM__CLASS(name, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(name, underlyingType, true, false, seq) \
    ENUM__DECOMPOSE \
    ENUM__VALIDATE_UNIQUE(name) \
    ENUM__END_TRAITS(name) \
    ENUM__BITWISE_OPS(name)

//! Defines a smart enumeration with a specific underlying type.
//! Duplicate enumeration values are allowed.
/*!
 * \param name Enumeration name.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_AMBIGUOUS_BIT_ENUM_WITH_UNDERLYING_TYPE(name, underlyingType, seq) \
    ENUM__CLASS(name, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(name, underlyingType, true, false, seq) \
    ENUM__DECOMPOSE \
    ENUM__END_TRAITS(name) \
    ENUM__BITWISE_OPS(name)

//! Defines a smart enumeration with the default |unsigned| underlying type.
/*!
 * \param name Enumeration name.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 */
#define DEFINE_BIT_ENUM(name, seq) \
    DEFINE_BIT_ENUM_WITH_UNDERLYING_TYPE(name, unsigned, seq)

//! Defines a smart enumeration with a specific underlying type and IsStringSerializable attribute.
/*!
 * \param name Enumeration name.
 * \param seq Enumeration domain encoded as a <em>sequence</em>.
 * \param underlyingType Underlying type.
 */
#define DEFINE_STRING_SERIALIZABLE_ENUM_WITH_UNDERLYING_TYPE(name, underlyingType, seq) \
    ENUM__CLASS(name, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(name, underlyingType, false, true, seq) \
    ENUM__MINMAX \
    ENUM__VALIDATE_UNIQUE(name) \
    ENUM__END_TRAITS(name) \

//! Defines a smart enumeration with a specific underlying type and IsStringSerializable attribute.
//! Duplicate enumeration values are allowed.
#define DEFINE_AMBIGUOUS_STRING_SERIALIZABLE_ENUM_WITH_UNDERLYING_TYPE(name, underlyingType, seq) \
    ENUM__CLASS(name, underlyingType, seq) \
    ENUM__BEGIN_TRAITS(name, underlyingType, false, true, seq) \
    ENUM__MINMAX \
    ENUM__END_TRAITS(name)

//! Defines a smart enumeration with the default |int| underlying type and IsStringSerializable attribute.
#define DEFINE_STRING_SERIALIZABLE_ENUM(name, seq) \
    DEFINE_STRING_SERIALIZABLE_ENUM_WITH_UNDERLYING_TYPE(name, int, seq)

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

    TEnumIndexedVector();
    TEnumIndexedVector(std::initializer_list<T> elements);

    T& operator[] (E index);
    const T& operator[] (E index) const;

    // STL interop.
    T* begin();
    const T* begin() const;
    T* end();
    const T* end() const;

    static bool IsDomainValue(E value);

private:
    using TUnderlying = typename TEnumTraits<E>::TUnderlying;
    static constexpr int N = static_cast<TUnderlying>(Max) - static_cast<TUnderlying>(Min) + 1;
    std::array<T, N> Items_;
};

////////////////////////////////////////////////////////////////////////////////

//! Returns |true| iff the enumeration value is not bitwise zero.
template <typename E, typename = std::enable_if_t<NYT::TEnumTraits<E>::IsBitEnum, E>>
bool Any(E value);

//! Returns |true| iff the enumeration value is bitwise zero.
template <typename E, typename = std::enable_if_t<NYT::TEnumTraits<E>::IsBitEnum, E>>
bool None(E value);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ENUM_INL_H_
#include "enum-inl.h"
#undef ENUM_INL_H_
