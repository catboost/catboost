//===- llvm/ADT/SmallSet.h - 'Normally small' sets --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TCompactSet class.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "compact_vector.h"

#include <util/system/yassert.h>

#include <cstddef>
#include <iterator>
#include <set>
#include <type_traits>

namespace NYT {

/// TCompactSet - This maintains a set of unique values, optimizing for the case
/// when the set is small (less than N).  In this case, the set can be
/// maintained with no mallocs.  If the set gets large, we expand to using an
/// std::set to maintain reasonable lookup times.
///
/// Note that any modification of the set may invalidate *all* iterators.
template <typename T, unsigned N, typename C = std::less<T>>
class TCompactSet
{
private:
    /// Use a CompactVector to hold the elements here (even though it will never
    /// reach its 'large' stage) to avoid calling the default ctors of elements
    /// we will never use.
    TCompactVector<T, N> Vector;
    std::set<T, C> Set;

    using TSetConstIterator = typename std::set<T, C>::const_iterator;
    using TVectorConstIterator = typename TCompactVector<T, N>::const_iterator;

public:
    class const_iterator;
    using size_type = std::size_t;

    TCompactSet() {}

    [[nodiscard]] bool empty() const;

    size_type size() const;

    const T& front() const;

    /// count - Return true if the element is in the set.
    size_type count(const T& v) const;

    /// insert - Insert an element into the set if it isn't already there.
    std::pair<const_iterator, bool> insert(const T& v);

    template <typename TIter>
    void insert(TIter i, TIter e);

    bool erase(const T& v);

    void clear();

    const_iterator begin() const;
    const_iterator cbegin() const;

    const_iterator end() const;
    const_iterator cend() const;

private:
    bool IsSmall() const;
};

} // namespace NYT

#define COMPACT_SET_INL_H_
#include "compact_set-inl.h"
#undef COMPACT_SET_INL_H_

