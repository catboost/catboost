#pragma once

#include <library/cpp/yt/misc/enum.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

//! A statically sized vector with elements of type |T| indexed by
//! the items of enumeration type |E|.
/*!
 *  By default, valid indexes are in range from the minimum declared value of |E|
 *  to the maximum declared value of |E|.
 *
 *  Items are value-initialized on construction.
 */
template <
    class E,
    class T,
    E Min = TEnumTraits<E>::GetMinValue(),
    E Max = TEnumTraits<E>::GetMaxValue()
>
class TEnumIndexedArray
{
public:
    static_assert(Min <= Max);

    using TIndex = E;
    using TValue = T;

    constexpr TEnumIndexedArray() = default;
    TEnumIndexedArray(std::initializer_list<std::pair<E, T>> elements);

    constexpr TEnumIndexedArray(const TEnumIndexedArray&) = default;
    constexpr TEnumIndexedArray(TEnumIndexedArray&&) = default;

    constexpr TEnumIndexedArray& operator=(const TEnumIndexedArray&) = default;
    constexpr TEnumIndexedArray& operator=(TEnumIndexedArray&&) = default;

    T& operator[] (E index);
    const T& operator[] (E index) const;

    // STL interop.
    T* begin();
    const T* begin() const;
    T* end();
    const T* end() const;

    constexpr size_t size() const;

    static bool IsValidIndex(E index);

private:
    static constexpr size_t Size = static_cast<size_t>(Max) - static_cast<size_t>(Min) + 1;
    std::array<T, Size> Items_{};
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define ENUM_INDEXED_ARRAY_INL_H_
#include "enum_indexed_array-inl.h"
#undef ENUM_INDEXED_ARRAY_INL_H_
