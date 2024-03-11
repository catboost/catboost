#pragma once
#ifndef ENUM_INDEXED_ARRAY_INL_H_
#error "Direct inclusion of this file is not allowed, include enum.h"
// For the sake of sane code completion.
#include "enum_indexed_array.h"
#endif

#include <library/cpp/yt/assert/assert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class E, class T, E Min, E Max>
TEnumIndexedArray<E, T, Min, Max>::TEnumIndexedArray(std::initializer_list<std::pair<E, T>> elements)
{
    for (const auto& [index, value] : elements) {
        (*this)[index] = value;
    }
}

template <class E, class T, E Min, E Max>
T& TEnumIndexedArray<E, T, Min, Max>::operator[] (E index)
{
    YT_ASSERT(IsValidIndex(index));
    return Items_[ToUnderlying(index) - ToUnderlying(Min)];
}

template <class E, class T, E Min, E Max>
const T& TEnumIndexedArray<E, T, Min, Max>::operator[] (E index) const
{
    return const_cast<TEnumIndexedArray&>(*this)[index];
}

template <class E, class T, E Min, E Max>
T* TEnumIndexedArray<E, T, Min, Max>::begin()
{
    return Items_.data();
}

template <class E, class T, E Min, E Max>
const T* TEnumIndexedArray<E, T, Min, Max>::begin() const
{
    return Items_.data();
}

template <class E, class T, E Min, E Max>
T* TEnumIndexedArray<E, T, Min, Max>::end()
{
    return begin() + Size;
}

template <class E, class T, E Min, E Max>
const T* TEnumIndexedArray<E, T, Min, Max>::end() const
{
    return begin() + Size;
}

template <class E, class T, E Min, E Max>
constexpr size_t TEnumIndexedArray<E, T, Min, Max>::size() const
{
    return Size;
}

template <class E, class T, E Min, E Max>
bool TEnumIndexedArray<E, T, Min, Max>::IsValidIndex(E index)
{
    return index >= Min && index <= Max;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
