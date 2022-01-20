#ifndef COMPACT_SET_INL_H_
#error "Direct inclusion of this file is not allowed, include compact_set.h"
// For the sake of sane code completion.
#include "compact_set.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <typename T, unsigned N,  typename C>
class TCompactSet<T, N, C>::const_iterator
{
private:
    friend class TCompactSet<T, N, C>;

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

    template <class TOther>
    void ConstructFrom(TOther&& rhs)
    {
        Y_VERIFY_DEBUG(Small == rhs.Small);

        if (Small) {
            new (&VIter)TVectorConstIterator(std::forward<TOther>(rhs).VIter);
        } else {
            new (&SIter)TSetConstIterator(std::forward<TOther>(rhs).SIter);
        }
    }

    template <class TOther>
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

template <typename T, unsigned N,  typename C>
bool TCompactSet<T, N, C>::empty() const
{
    return Vector.empty() && Set.empty();
}

template <typename T, unsigned N,  typename C>
typename TCompactSet<T, N, C>::size_type TCompactSet<T, N, C>::size() const
{
    return IsSmall() ? Vector.size() : Set.size();
}

template <typename T, unsigned N,  typename C>
const T& TCompactSet<T, N, C>::front() const
{
    return IsSmall() ? Vector.front() : *Set.begin();
}


template <typename T, unsigned N,  typename C>
typename TCompactSet<T, N, C>::size_type TCompactSet<T, N, C>::count(const T& v) const
{
    if (IsSmall()) {
        return std::binary_search(Vector.begin(), Vector.end(), v, C()) ? 1 : 0;
    } else {
        return Set.count(v);
    }
}

template <typename T, unsigned N,  typename C>
std::pair<typename TCompactSet<T, N, C>::const_iterator, bool> TCompactSet<T, N, C>::insert(const T& v)
{
    if (!IsSmall()) {
        auto [it, inserted] = Set.insert(v);
        return {const_iterator(std::move(it)), inserted};
    }

    auto it = std::lower_bound(Vector.begin(), Vector.end(), v, C());
    if (it != Vector.end() && !C()(v, *it)) {
        return {const_iterator(std::move(it)), false}; // Don't reinsert if it already exists.
    }

    if (Vector.size() < N) {
        auto newIt = Vector.insert(it, v);
        return {const_iterator(std::move(newIt)), true};
    }

    Set.insert(Vector.begin(), Vector.end());
    Vector.clear();

    auto [newIt, inserted] = Set.insert(v);
    Y_VERIFY_DEBUG(inserted);
    return {const_iterator(std::move(newIt)), true};
}

template <typename T, unsigned N,  typename C>
template <typename TIter>
void TCompactSet<T, N, C>::insert(TIter i, TIter e)
{
    for (; i != e; ++i) {
        insert(*i);
    }
}

template <typename T, unsigned N,  typename C>
bool TCompactSet<T, N, C>::erase(const T& v)
{
    if (!IsSmall()) {
        return Set.erase(v);
    }

    auto [rangeBegin, rangeEnd] = std::equal_range(Vector.begin(), Vector.end(), v, C());
    if (rangeBegin != rangeEnd) {
        Vector.erase(rangeBegin, rangeEnd);
        return true;
    } else {
        return false;
    }
}

template <typename T, unsigned N,  typename C>
void TCompactSet<T, N, C>::clear()
{
    Vector.clear();
    Set.clear();
}

template <typename T, unsigned N,  typename C>
typename TCompactSet<T, N, C>::const_iterator TCompactSet<T, N, C>::begin() const
{
    return IsSmall() ? const_iterator(Vector.begin()) : const_iterator(Set.begin());
}

template <typename T, unsigned N,  typename C>
typename TCompactSet<T, N, C>::const_iterator TCompactSet<T, N, C>::cbegin() const
{
    return begin();
}

template <typename T, unsigned N,  typename C>
typename TCompactSet<T, N, C>::const_iterator TCompactSet<T, N, C>::end() const
{
    return IsSmall() ? const_iterator(Vector.end()) : const_iterator(Set.end());
}

template <typename T, unsigned N,  typename C>
typename TCompactSet<T, N, C>::const_iterator TCompactSet<T, N, C>::cend() const
{
    return end();
}

template <typename T, unsigned N,  typename C>
bool TCompactSet<T, N, C>::IsSmall() const
{
    return Set.empty();
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
