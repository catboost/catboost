#pragma once
#ifndef ENUM_INL_H_
#error "Direct inclusion of this file is not allowed, include enum.h"
// For the sake of sane code completion.
#include "enum.h"
#endif

#include <util/string/printf.h>
#include <util/string/cast.h>

#include <algorithm>
#include <stdexcept>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

#define ENUM__CLASS(name, underlyingType, seq) \
    enum class name : underlyingType \
    { \
        PP_FOR_EACH(ENUM__DOMAIN_ITEM, seq) \
    };

#define ENUM__DOMAIN_ITEM(item) \
    PP_IF( \
        PP_IS_SEQUENCE(item), \
        ENUM__DOMAIN_ITEM_SEQ, \
        ENUM__DOMAIN_ITEM_ATOMIC \
    )(item)()

#define ENUM__DOMAIN_ITEM_ATOMIC(item) \
    item PP_COMMA

#define ENUM__DOMAIN_ITEM_SEQ(seq) \
    PP_ELEMENT(seq, 0) = PP_ELEMENT(seq, 1) PP_COMMA

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <typename TValues>
static constexpr bool AreValuesDistinct(const TValues& values)
{
    for (int i = 0; i < static_cast<int>(values.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(values.size()); ++j) {
            if (values[i] == values[j]) {
                return false;
            }
        }
    }
    return true;
}

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

#define ENUM__BEGIN_TRAITS(name, underlyingType, isBit, isStringSerializable, seq) \
    struct TEnumTraitsImpl_##name \
    { \
        using TType = name; \
        using TUnderlying = underlyingType; \
        [[maybe_unused]] static constexpr bool IsBitEnum = isBit; \
        [[maybe_unused]] static constexpr bool IsStringSerializableEnum = isStringSerializable; \
        [[maybe_unused]] static constexpr int DomainSize = PP_COUNT(seq); \
        \
        static constexpr std::array<TStringBuf, DomainSize> Names{{ \
            PP_FOR_EACH(ENUM__GET_DOMAIN_NAMES_ITEM, seq) \
        }}; \
        static constexpr std::array<TType, DomainSize> Values{{ \
            PP_FOR_EACH(ENUM__GET_DOMAIN_VALUES_ITEM, seq) \
        }}; \
        \
        static TStringBuf GetTypeName() \
        { \
            static constexpr TStringBuf typeName = PP_STRINGIZE(name); \
            return typeName; \
        } \
        \
        static const TStringBuf* FindLiteralByValue(TType value) \
        { \
            for (int i = 0; i < DomainSize; ++i) { \
                if (Values[i] == value) { \
                    return &Names[i]; \
                } \
            } \
            return nullptr; \
        } \
        \
        static bool FindValueByLiteral(TStringBuf literal, TType* result) \
        { \
            for (int i = 0; i < DomainSize; ++i) { \
                if (Names[i] == literal) { \
                    *result = Values[i]; \
                    return true; \
                } \
            } \
            return false; \
        } \
        \
        static const std::array<TStringBuf, DomainSize>& GetDomainNames() \
        { \
            return Names; \
        } \
        \
        static const std::array<TType, DomainSize>& GetDomainValues() \
        { \
            return Values; \
        } \
        \
        static TType FromString(TStringBuf str) \
        { \
            TType value; \
            if (!FindValueByLiteral(str, &value)) { \
                throw ::NYT::TSimpleException(Sprintf("Error parsing %s value %s", \
                    PP_STRINGIZE(name), \
                    TString(str).Quote().c_str()).c_str()); \
            } \
            return value; \
        }

#define ENUM__GET_DOMAIN_VALUES_ITEM(item) \
    PP_IF( \
        PP_IS_SEQUENCE(item), \
        ENUM__GET_DOMAIN_VALUES_ITEM_SEQ, \
        ENUM__GET_DOMAIN_VALUES_ITEM_ATOMIC \
    )(item)

#define ENUM__GET_DOMAIN_VALUES_ITEM_SEQ(seq) \
    ENUM__GET_DOMAIN_VALUES_ITEM_ATOMIC(PP_ELEMENT(seq, 0))

#define ENUM__GET_DOMAIN_VALUES_ITEM_ATOMIC(item) \
    TType::item,

#define ENUM__GET_DOMAIN_NAMES_ITEM(item) \
    PP_IF( \
        PP_IS_SEQUENCE(item), \
        ENUM__GET_DOMAIN_NAMES_ITEM_SEQ, \
        ENUM__GET_DOMAIN_NAMES_ITEM_ATOMIC \
    )(item)

#define ENUM__GET_DOMAIN_NAMES_ITEM_SEQ(seq) \
    ENUM__GET_DOMAIN_NAMES_ITEM_ATOMIC(PP_ELEMENT(seq, 0))

#define ENUM__GET_DOMAIN_NAMES_ITEM_ATOMIC(item) \
    TStringBuf(PP_STRINGIZE(item)),

#define ENUM__DECOMPOSE \
    static std::vector<TType> Decompose(TType value) \
    { \
        std::vector<TType> result; \
        for (int i = 0; i < DomainSize; ++i) { \
            if (static_cast<TUnderlying>(value) & static_cast<TUnderlying>(Values[i])) { \
                result.push_back(Values[i]); \
            } \
        } \
        return result; \
    }

#define ENUM__MINMAX \
    static constexpr TType GetMinValue() \
    { \
        static_assert(!Values.empty()); \
        return *std::min_element(std::begin(Values), std::end(Values)); \
    } \
    \
    static constexpr TType GetMaxValue() \
    { \
        static_assert(!Values.empty()); \
        return *std::max_element(std::begin(Values), std::end(Values)); \
    }

#define ENUM__VALIDATE_UNIQUE(name) \
    static_assert(::NYT::NDetail::AreValuesDistinct(Values), \
        "Enumeration " #name " contains duplicate values");

#define ENUM__END_TRAITS(name) \
    }; \
    \
    [[maybe_unused]] inline TEnumTraitsImpl_##name GetEnumTraitsImpl(name) \
    { \
        return TEnumTraitsImpl_##name(); \
    } \
    \
    using ::ToString; \
    [[maybe_unused]] inline TString ToString(name value) \
    { \
        return ::NYT::TEnumTraits<name>::ToString(value); \
    }

////////////////////////////////////////////////////////////////////////////////

template <class T>
std::vector<T> TEnumTraits<T, true>::Decompose(T value)
{
    return TImpl::Decompose(value);
}

template <class T>
T TEnumTraits<T, true>::FromString(TStringBuf str)
{
    return TImpl::FromString(str);
}

template <class T>
TString TEnumTraits<T, true>::ToString(TType value)
{
    TString result;
    const auto* literal = FindLiteralByValue(value);
    if (literal) {
        result = *literal;
    } else {
        result = GetTypeName();
        result += "(";
        result += ::ToString(static_cast<TUnderlying>(value));
        result += ")";
    }
    return result;
}

template <class T>
auto TEnumTraits<T, true>::GetDomainValues() -> const std::array<T, DomainSize>&
{
    return TImpl::GetDomainValues();
}

template <class T>
auto TEnumTraits<T, true>::GetDomainNames() -> const std::array<TStringBuf, DomainSize>&
{
    return TImpl::GetDomainNames();
}

template <class T>
constexpr T TEnumTraits<T, true>::GetMaxValue()
{
    return TImpl::GetMaxValue();
}

template <class T>
constexpr T TEnumTraits<T, true>::GetMinValue()
{
    return TImpl::GetMinValue();
}

template <class T>
bool TEnumTraits<T, true>::FindValueByLiteral(TStringBuf literal, TType* result)
{
    return TImpl::FindValueByLiteral(literal, result);
}

template <class T>
const TStringBuf* TEnumTraits<T, true>::FindLiteralByValue(TType value)
{
    return TImpl::FindLiteralByValue(value);
}

template <class T>
TStringBuf TEnumTraits<T, true>::GetTypeName()
{
    return TImpl::GetTypeName();
}

////////////////////////////////////////////////////////////////////////////////

template <class E, class T, E Min, E Max>
TEnumIndexedVector<E, T, Min, Max>::TEnumIndexedVector()
    : Items_{}
{ }

template <class E, class T, E Min, E Max>
TEnumIndexedVector<E, T, Min, Max>::TEnumIndexedVector(std::initializer_list<T> elements)
    : Items_{}
{
    Y_ASSERT(std::distance(elements.begin(), elements.end()) <= N);
    size_t index = 0;
    for (const auto& element : elements) {
        Items_[index++] = element;
    }
}

template <class E, class T, E Min, E Max>
T& TEnumIndexedVector<E, T, Min, Max>::operator[] (E index)
{
    Y_ASSERT(index >= Min && index <= Max);
    return Items_[static_cast<TUnderlying>(index) - static_cast<TUnderlying>(Min)];
}

template <class E, class T, E Min, E Max>
const T& TEnumIndexedVector<E, T, Min, Max>::operator[] (E index) const
{
    return const_cast<TEnumIndexedVector&>(*this)[index];
}

template <class E, class T, E Min, E Max>
T* TEnumIndexedVector<E, T, Min, Max>::begin()
{
    return Items_.data();
}

template <class E, class T, E Min, E Max>
const T* TEnumIndexedVector<E, T, Min, Max>::begin() const
{
    return Items_.data();
}

template <class E, class T, E Min, E Max>
T* TEnumIndexedVector<E, T, Min, Max>::end()
{
    return begin() + N;
}

template <class E, class T, E Min, E Max>
const T* TEnumIndexedVector<E, T, Min, Max>::end() const
{
    return begin() + N;
}

template <class E, class T, E Min, E Max>
bool TEnumIndexedVector<E, T, Min, Max>::IsDomainValue(E value)
{
    return value >= Min && value <= Max;
}

////////////////////////////////////////////////////////////////////////////////

#define ENUM__BINARY_BITWISE_OPERATOR(T, assignOp, op) \
    [[maybe_unused]] inline constexpr T operator op (T lhs, T rhs) \
    { \
        using TUnderlying = typename TEnumTraits<T>::TUnderlying; \
        return T(static_cast<TUnderlying>(lhs) op static_cast<TUnderlying>(rhs)); \
    } \
    \
    [[maybe_unused]] inline T& operator assignOp (T& lhs, T rhs) \
    { \
        using TUnderlying = typename TEnumTraits<T>::TUnderlying; \
        lhs = T(static_cast<TUnderlying>(lhs) op static_cast<TUnderlying>(rhs)); \
        return lhs; \
    }

#define ENUM__UNARY_BITWISE_OPERATOR(T, op) \
    [[maybe_unused]] inline constexpr T operator op (T value) \
    { \
        using TUnderlying = typename TEnumTraits<T>::TUnderlying; \
        return T(op static_cast<TUnderlying>(value)); \
    }

#define ENUM__BIT_SHIFT_OPERATOR(T, assignOp, op) \
    [[maybe_unused]] inline constexpr T operator op (T lhs, size_t rhs) \
    { \
        using TUnderlying = typename TEnumTraits<T>::TUnderlying; \
        return T(static_cast<TUnderlying>(lhs) op rhs); \
    } \
    \
    [[maybe_unused]] inline T& operator assignOp (T& lhs, size_t rhs) \
    { \
        using TUnderlying = typename TEnumTraits<T>::TUnderlying; \
        lhs = T(static_cast<TUnderlying>(lhs) op rhs); \
        return lhs; \
    }

#define ENUM__BITWISE_OPS(name) \
    ENUM__BINARY_BITWISE_OPERATOR(name, &=, &)  \
    ENUM__BINARY_BITWISE_OPERATOR(name, |=, | ) \
    ENUM__BINARY_BITWISE_OPERATOR(name, ^=, ^)  \
    ENUM__UNARY_BITWISE_OPERATOR(name, ~)       \
    ENUM__BIT_SHIFT_OPERATOR(name, <<=, << )    \
    ENUM__BIT_SHIFT_OPERATOR(name, >>=, >> )

////////////////////////////////////////////////////////////////////////////////

template <typename E, typename>
bool Any(E value)
{
    return static_cast<typename TEnumTraits<E>::TUnderlying>(value) != 0;
}

template <class E, typename>
bool None(E value)
{
    return static_cast<typename TEnumTraits<E>::TUnderlying>(value) == 0;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
