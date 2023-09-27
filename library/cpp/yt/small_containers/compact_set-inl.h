#ifndef COMPACT_SET_INL_H_
#error "Direct inclusion of this file is not allowed, include compact_set.h"
// For the sake of sane code completion.
#include "compact_set.h"
#endif

#include <library/cpp/yt/assert/assert.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t N,  typename C, typename A>
class TCompactSet<T, N, C, A>::const_iterator
{
private:
    friend class TCompactSet<T, N, C, A>;

    union
    {
        TVectorConstIterator VIter;
        TSetConstIterator SIter;
    };

    bool Small;

    const_iterator(TVectorConstIterator it)
        : VIter(it)
        , Small(true)
    { }

    const_iterator(TSetConstIterator it)
        : SIter(it)
        , Small(false)
    { }

    template <typename TOther>
    void ConstructFrom(TOther&& rhs)
    {
        YT_ASSERT(Small == rhs.Small);

        if (Small) {
            new (&VIter)TVectorConstIterator(std::forward<TOther>(rhs).VIter);
        } else {
            new (&SIter)TSetConstIterator(std::forward<TOther>(rhs).SIter);
        }
    }

    template <typename TOther>
    const_iterator& AssignFrom(TOther&& rhs)
    {
        if (this == &rhs) {
            return *this;
        }

        if (Small && rhs.Small) {
            VIter = std::forward<TOther>(rhs).VIter;
        } else if (!Small && !rhs.Small) {
            SIter = std::forward<TOther>(rhs).SIter;
        } else {
            if (Small) {
                VIter.~TVectorConstIterator();
            } else {
                SIter.~TSetConstIterator();
            }

            if (rhs.Small) {
                new (&VIter)TVectorConstIterator(std::forward<TOther>(rhs).VIter);
            } else {
                new (&SIter)TSetConstIterator(std::forward<TOther>(rhs).SIter);
            }
        }

        Small = rhs.Small;

        return *this;
    }

public:
    static_assert(std::is_same_v<
        typename std::iterator_traits<TVectorConstIterator>::difference_type,
        typename std::iterator_traits<TSetConstIterator>::difference_type>);
    static_assert(std::is_same_v<
        typename std::iterator_traits<TVectorConstIterator>::value_type,
        typename std::iterator_traits<TSetConstIterator>::value_type>);
    static_assert(std::is_same_v<
        typename std::iterator_traits<TVectorConstIterator>::pointer,
        typename std::iterator_traits<TSetConstIterator>::pointer>);
    static_assert(std::is_same_v<
        typename std::iterator_traits<TVectorConstIterator>::reference,
        typename std::iterator_traits<TSetConstIterator>::reference>);

    using difference_type = typename std::iterator_traits<TVectorConstIterator>::difference_type;
    using value_type = typename std::iterator_traits<TVectorConstIterator>::value_type;
    using pointer = typename std::iterator_traits<TVectorConstIterator>::pointer;
    using reference = typename std::iterator_traits<TVectorConstIterator>::reference;
    using iterator_category = std::bidirectional_iterator_tag;

    const_iterator(const const_iterator& rhs)
        : Small(rhs.Small)
    {
        ConstructFrom(rhs);
    }

    const_iterator(const_iterator&& rhs)
        : Small(rhs.Small)
    {
        ConstructFrom(std::move(rhs));
    }

    ~const_iterator()
    {
        if (Small) {
            VIter.~TVectorConstIterator();
        } else {
            SIter.~TSetConstIterator();
        }
    }

    const_iterator& operator=(const const_iterator& rhs)
    {
        return AssignFrom(rhs);
    }

    const_iterator& operator=(const_iterator&& rhs)
    {
        return AssignFrom(std::move(rhs));
    }

    const_iterator& operator++()
    {
        if (Small) {
            ++VIter;
        } else {
            ++SIter;
        }

        return *this;
    }

    const_iterator operator++(int)
    {
        auto result = *this;

        if (Small) {
            ++VIter;
        } else {
            ++SIter;
        }

        return result;
    }

    const_iterator& operator--()
    {
        if (Small) {
            --VIter;
        } else {
            --SIter;
        }

        return *this;
    }

    const_iterator operator--(int)
    {
        auto result = *this;

        if (Small) {
            --VIter;
        } else {
            --SIter;
        }

        return result;
    }

    bool operator==(const const_iterator& rhs) const
    {
        if (Small != rhs.Small) {
            return false;
        }

        return Small ? (VIter == rhs.VIter) : (SIter == rhs.SIter);
    }

    bool operator!=(const const_iterator& rhs) const
    {
        return !(*this == rhs);
    }

    const T& operator*() const
    {
        return Small ? *VIter : *SIter;
    }

    const T* operator->() const
    {
        return &operator*();
    }
};

////////////////////////////////////////////////////////////////////////////////

template <typename T, size_t N,  typename C, typename A>
TCompactSet<T, N, C, A>::TCompactSet(const A& allocator)
    : Set_(allocator)
{ }

template <typename T, size_t N,  typename C, typename A>
bool TCompactSet<T, N, C, A>::empty() const
{
    return Vector_.empty() && Set_.empty();
}

template <typename T, size_t N,  typename C, typename A>
typename TCompactSet<T, N, C, A>::size_type TCompactSet<T, N, C, A>::size() const
{
    return is_small() ? Vector_.size() : Set_.size();
}

template <typename T, size_t N,  typename C, typename A>
const T& TCompactSet<T, N, C, A>::front() const
{
    return is_small() ? Vector_.front() : *Set_.begin();
}

template <typename T, size_t N,  typename C, typename A>
typename TCompactSet<T, N, C, A>::size_type TCompactSet<T, N, C, A>::count(const T& v) const
{
    if (is_small()) {
        return std::binary_search(Vector_.begin(), Vector_.end(), v, C()) ? 1 : 0;
    } else {
        return Set_.count(v);
    }
}

template <typename T, size_t N,  typename C, typename A>
bool TCompactSet<T, N, C, A>::contains(const T& v) const
{
    return count(v) == 1;
}

template <typename T, size_t N,  typename C, typename A>
std::pair<typename TCompactSet<T, N, C, A>::const_iterator, bool> TCompactSet<T, N, C, A>::insert(const T& v)
{
    if (!is_small()) {
        auto [it, inserted] = Set_.insert(v);
        return {const_iterator(std::move(it)), inserted};
    }

    auto it = std::lower_bound(Vector_.begin(), Vector_.end(), v, C());
    if (it != Vector_.end() && !C()(v, *it)) {
        return {const_iterator(std::move(it)), false}; // Don't reinsert if it already exists.
    }

    if (Vector_.size() < N) {
        auto newIt = Vector_.insert(it, v);
        return {const_iterator(std::move(newIt)), true};
    }

    Set_.insert(Vector_.begin(), Vector_.end());
    Vector_.clear();

    auto [newIt, inserted] = Set_.insert(v);
    YT_ASSERT(inserted);
    return {const_iterator(std::move(newIt)), true};
}

template <typename T, size_t N,  typename C, typename A>
template <typename TIter>
void TCompactSet<T, N, C, A>::insert(TIter i, TIter e)
{
    for (; i != e; ++i) {
        insert(*i);
    }
}

template <typename T, size_t N,  typename C, typename A>
bool TCompactSet<T, N, C, A>::erase(const T& v)
{
    if (!is_small()) {
        return Set_.erase(v);
    }

    auto [rangeBegin, rangeEnd] = std::equal_range(Vector_.begin(), Vector_.end(), v, C());
    if (rangeBegin != rangeEnd) {
        Vector_.erase(rangeBegin, rangeEnd);
        return true;
    } else {
        return false;
    }
}

template <typename T, size_t N,  typename C, typename A>
void TCompactSet<T, N, C, A>::clear()
{
    Vector_.clear();
    Set_.clear();
}

template <typename T, size_t N,  typename C, typename A>
typename TCompactSet<T, N, C, A>::const_iterator TCompactSet<T, N, C, A>::begin() const
{
    return is_small() ? const_iterator(Vector_.begin()) : const_iterator(Set_.begin());
}

template <typename T, size_t N,  typename C, typename A>
typename TCompactSet<T, N, C, A>::const_iterator TCompactSet<T, N, C, A>::cbegin() const
{
    return begin();
}

template <typename T, size_t N,  typename C, typename A>
typename TCompactSet<T, N, C, A>::const_iterator TCompactSet<T, N, C, A>::end() const
{
    return is_small() ? const_iterator(Vector_.end()) : const_iterator(Set_.end());
}

template <typename T, size_t N,  typename C, typename A>
typename TCompactSet<T, N, C, A>::const_iterator TCompactSet<T, N, C, A>::cend() const
{
    return end();
}

template <typename T, size_t N,  typename C, typename A>
bool TCompactSet<T, N, C, A>::is_small() const
{
    return Set_.empty();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
