#pragma once
#ifndef ENUM_INL_H_
#error "Direct inclusion of this file is not allowed, include enum.h"
// For the sake of sane code completion.
#include "enum.h"
#endif

#include <util/string/printf.h>
#include <util/string/cast.h>

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

#define ENUM__BEGIN_TRAITS(name, underlyingType, isBit, isStringSerializable, seq) \
    struct TEnumTraitsImpl_##name \
    { \
        using TType = name; \
        using TUnderlying = underlyingType; \
        [[maybe_unused]] static constexpr bool IsBitEnum = isBit; \
        [[maybe_unused]] static constexpr bool IsStringSerializableEnum = isStringSerializable; \
        [[maybe_unused]] static constexpr int DomainSize = PP_COUNT(seq); \
        \
        static TStringBuf GetTypeName() \
        { \
            static constexpr TStringBuf typeName = PP_STRINGIZE(name); \
            return typeName; \
        } \
        \
        static const TStringBuf* FindLiteralByValue(TType value) \
        { \
            PP_FOR_EACH(ENUM__LITERAL_BY_VALUE_ITEM, seq) \
            return nullptr; \
        } \
        \
        static bool FindValueByLiteral(TStringBuf literal, TType* result) \
        { \
            PP_FOR_EACH(ENUM__VALUE_BY_LITERAL_ITEM, seq); \
            return false; \
        } \
        \
        static const std::array<TStringBuf, DomainSize>& GetDomainNames() \
        { \
            static const std::array<TStringBuf, PP_COUNT(seq)> Result{{ \
                PP_FOR_EACH(ENUM__GET_DOMAIN_NAMES_ITEM, seq) \
            }}; \
            return Result; \
        } \
        \
        static const std::array<TType, DomainSize>& GetDomainValues() \
        { \
            static const std::array<TType, PP_COUNT(seq)> Result{{ \
                PP_FOR_EACH(ENUM__GET_DOMAIN_VALUES_ITEM, seq) \
            }}; \
            return Result; \
        } \
        \
        static TType FromString(TStringBuf str) \
        { \
            TType value; \
            if (!FindValueByLiteral(str, &value)) { \
                throw std::runtime_error(Sprintf("Error parsing %s value %s", \
                    PP_STRINGIZE(name), \
                    TString(str).Quote().data()).data()); \
            } \
            return value; \
        }

#define ENUM__LITERAL_BY_VALUE_ITEM(item) \
    PP_IF( \
        PP_IS_SEQUENCE(item), \
        ENUM__LITERAL_BY_VALUE_ITEM_SEQ, \
        ENUM__LITERAL_BY_VALUE_ITEM_ATOMIC \
    )(item)

#define ENUM__LITERAL_BY_VALUE_ITEM_SEQ(seq) \
    ENUM__LITERAL_BY_VALUE_ITEM_ATOMIC(PP_ELEMENT(seq, 0))

#define ENUM__LITERAL_BY_VALUE_ITEM_ATOMIC(item) \
    if (static_cast<TUnderlying>(value) == static_cast<TUnderlying>(TType::item)) { \
        static constexpr TStringBuf literal = PP_STRINGIZE(item); \
        return &literal; \
    }

#define ENUM__VALUE_BY_LITERAL_ITEM(item) \
    PP_IF( \
        PP_IS_SEQUENCE(item), \
        ENUM__VALUE_BY_LITERAL_ITEM_SEQ, \
        ENUM__VALUE_BY_LITERAL_ITEM_ATOMIC \
    )(item)

#define ENUM__VALUE_BY_LITERAL_ITEM_SEQ(seq) \
    ENUM__VALUE_BY_LITERAL_ITEM_ATOMIC(PP_ELEMENT(seq, 0))

#define ENUM__VALUE_BY_LITERAL_ITEM_ATOMIC(item) \
    if (literal == PP_STRINGIZE(item)) { \
        *result = TType::item; \
        return true; \
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

#define ENUM__DECOMPOSE(name, seq) \
    static std::vector<TType> Decompose(TType value) \
    { \
        std::vector<TType> result; \
        PP_FOR_EACH(ENUM__DECOMPOSE_ITEM, seq) \
        return result; \
    }

#define ENUM__DECOMPOSE_ITEM(item) \
    ENUM__DECOMPOSE_ITEM_SEQ(PP_ELEMENT(item, 0))

#define ENUM__DECOMPOSE_ITEM_SEQ(item) \
    if (static_cast<TUnderlying>(value) & static_cast<TUnderlying>(TType::item)) { \
        result.push_back(TType::item); \
    }

#define ENUM__MINMAX(name, seq) \
    ENUM__MINMAX_IMPL(name, seq, Min, min) \
    ENUM__MINMAX_IMPL(name, seq, Max, max)

#define ENUM__MINMAX_IMPL(name, seq, Extr, extr) \
    static constexpr TType Get##Extr##Value() \
    { \
        return static_cast<TType>(std::extr({ \
            PP_FOR_EACH(ENUM__MINMAX_ITEM, seq) \
            ENUM__MINMAX_ITEM_CORE(PP_HEAD(seq)) \
        })); \
    }

#define ENUM__MINMAX_ITEM(item) \
    ENUM__MINMAX_ITEM_CORE(item),

#define ENUM__MINMAX_ITEM_CORE(item) \
    PP_IF( \
        PP_IS_SEQUENCE(item), \
        ENUM__MINMAX_ITEM_CORE_SEQ, \
        ENUM__MINMAX_ITEM_CORE_ATOMIC \
    )(item)

#define ENUM__MINMAX_ITEM_CORE_SEQ(seq) \
    ENUM__MINMAX_ITEM_CORE_ATOMIC(PP_ELEMENT(seq, 0))

#define ENUM__MINMAX_ITEM_CORE_ATOMIC(item) \
    static_cast<TUnderlying>(TType::item)

#define ENUM__VALIDATE_UNIQUE(seq) \
    static constexpr bool CheckDistinctFromOthers() \
    { \
        return true; \
    } \
    \
    static constexpr bool CheckDistinctFromOthers(TType) \
    { \
        return true; \
    } \
    \
    template<class ... TOthers> \
    static constexpr bool CheckDistinctFromOthers(TType first, TType second, TOthers... others) \
    { \
        return static_cast<TUnderlying>(first) != static_cast<TUnderlying>(second) && \
            CheckDistinctFromOthers(first, others...); \
    } \
    \
    static constexpr bool CheckAllDistinct(std::nullptr_t /*trailing comma helper*/) { \
        return true; \
    } \
    \
    template<class ... TOthers> \
    static constexpr bool CheckAllDistinct(std::nullptr_t, TType first, TOthers... others) \
    { \
        return CheckDistinctFromOthers(first, others...) && CheckAllDistinct(nullptr, others...); \
    } \
    \
    static constexpr bool DoCheckAllDistinct() { \
        return CheckAllDistinct(0 PP_FOR_EACH(ENUM__VALIDATE_UNIQUE_ITEM, seq)); \
    }

#define ENUM__VALIDATE_UNIQUE_ITEM(item) \
    PP_IF( \
        PP_IS_SEQUENCE(item), \
        ENUM__VALIDATE_UNIQUE_ITEM_SEQ, \
        ENUM__VALIDATE_UNIQUE_ITEM_ATOMIC \
    )(item)

#define ENUM__VALIDATE_UNIQUE_ITEM_SEQ(seq) \
    ENUM__VALIDATE_UNIQUE_ITEM_ATOMIC(PP_ELEMENT(seq, 0))

#define ENUM__VALIDATE_UNIQUE_ITEM_ATOMIC(item) \
    , TType::item

#define ENUM__END_TRAITS(name) \
    }; \
    \
    [[maybe_unused]] inline TEnumTraitsImpl_##name GetEnumTraitsImpl(name) \
    { \
        return TEnumTraitsImpl_##name(); \
    } \
    using ::ToString; \
    [[maybe_unused]] inline TString ToString(name value) \
    { \
        return ::NYT::TEnumTraits<name>::ToString(value); \
    }

#define ENUM__VALIDATE_UNIQUE_EPILOGUE(name) \
    static_assert(TEnumTraitsImpl_##name::DoCheckAllDistinct(), \
        "Enumeration " #name " contains duplicate values");

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
TEnumIndexedVector<E, T, Min, Max>::TEnumIndexedVector(const TEnumIndexedVector& other)
{
    *this = other;
}

template <class E, class T, E Min, E Max>
TEnumIndexedVector<E, T, Min, Max>::TEnumIndexedVector(TEnumIndexedVector&& other)
{
    *this = std::move(other);
}

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
TEnumIndexedVector<E, T, Min, Max>& TEnumIndexedVector<E, T, Min, Max>::operator=(const TEnumIndexedVector& other)
{
    std::copy(other.Items_.begin(), other.Items_.end(), Items_.begin());
    return *this;
}

template <class E, class T, E Min, E Max>
TEnumIndexedVector<E, T, Min, Max>& TEnumIndexedVector<E, T, Min, Max>::operator=(TEnumIndexedVector&& other)
{
    std::move(other.Items_.begin(), other.Items_.end(), Items_.begin());
    return *this;
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

template <class E>
typename std::enable_if<TEnumTraits<E>::IsBitEnum, bool>::type
Any(E value)
{
    return static_cast<typename TEnumTraits<E>::TUnderlying>(value) != 0;
}

template <class E>
typename std::enable_if<TEnumTraits<E>::IsBitEnum, bool>::type
None(E value)
{
    return static_cast<typename TEnumTraits<E>::TUnderlying>(value) == 0;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

