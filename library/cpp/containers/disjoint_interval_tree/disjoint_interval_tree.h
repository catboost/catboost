#pragma once

#include <util/generic/map.h>
#include <util/system/yassert.h>

#include <type_traits>

template <class T>
class TDisjointIntervalTree {
private:
    static_assert(std::is_integral<T>::value, "expect std::is_integral<T>::value");

    using TTree = TMap<T, T>; // [key, value)
    using TIterator = typename TTree::iterator;
    using TConstIterator = typename TTree::const_iterator;
    using TReverseIterator = typename TTree::reverse_iterator;
    using TThis = TDisjointIntervalTree<T>;

    TTree Tree;
    size_t NumElements;

public:
    TDisjointIntervalTree()
        : NumElements()
    {
    }

    void Insert(const T t) {
        InsertInterval(t, t + 1);
    }

    // we assume that none of elements from [begin, end) belong to tree.
    void InsertInterval(const T begin, const T end) {
        InsertIntervalImpl(begin, end);
        NumElements += (size_t)(end - begin);
    }

    bool Has(const T t) const {
        return const_cast<TThis*>(this)->FindContaining(t) != Tree.end();
    }

    bool Intersects(const T begin, const T end) {
        if (Empty()) {
            return false;
        }

        TIterator l = Tree.lower_bound(begin);
        if (l != Tree.end()) {
            if (l->first < end) {
                return true;
            } else if (l != Tree.begin()) {
                --l;
                return l->second > begin;
            } else {
                return false;
            }
        } else {
            auto last = Tree.rbegin();
            return begin < last->second;
        }
    }

    TConstIterator FindContaining(const T t) const {
        return const_cast<TThis*>(this)->FindContaining(t);
    }

    // Erase element. Returns true when element has been deleted, otherwise false.
    bool Erase(const T t) {
        TIterator n = FindContaining(t);
        if (n == Tree.end()) {
            return false;
        }

        --NumElements;

        T& begin = const_cast<T&>(n->first);
        T& end = const_cast<T&>(n->second);

        // Optimization hack.
        if (t == begin) {
            if (++begin == end) { // OK to change key since intervals do not intersect.
                Tree.erase(n);
                return true;
            }

        } else if (t == end - 1) {
            --end;

        } else {
            const T e = end;
            end = t;
            InsertIntervalImpl(t + 1, e);
        }

        Y_ASSERT(begin < end);
        return true;
    }

    // Erase interval. Returns number of elements removed from set.
    size_t EraseInterval(const T begin, const T end) {
        Y_ASSERT(begin < end);

        if (Empty()) {
            return 0;
        }

        size_t elementsRemoved = 0;

        TIterator completelyRemoveBegin = Tree.lower_bound(begin);
        if ((completelyRemoveBegin != Tree.end() && completelyRemoveBegin->first > begin && completelyRemoveBegin != Tree.begin())
            || completelyRemoveBegin == Tree.end()) {
            // Look at the interval. It could contain [begin, end).
            TIterator containingBegin = completelyRemoveBegin;
            --containingBegin;
            if (containingBegin->first < begin && begin < containingBegin->second) { // Contains begin.
                if (containingBegin->second > end) { // Contains end.
                    const T prevEnd = containingBegin->second;
                    Y_ASSERT(containingBegin->second - begin <= NumElements);

                    Y_ASSERT(containingBegin->second - containingBegin->first > end - begin);
                    containingBegin->second = begin;
                    InsertIntervalImpl(end, prevEnd);

                    elementsRemoved = end - begin;
                    NumElements -= elementsRemoved;
                    return elementsRemoved;
                } else {
                    elementsRemoved += containingBegin->second - begin;
                    containingBegin->second = begin;
                }
            }
        }

        TIterator completelyRemoveEnd = completelyRemoveBegin != Tree.end() ? Tree.lower_bound(end) : Tree.end();
        if (completelyRemoveEnd != Tree.begin() && (completelyRemoveEnd == Tree.end() || completelyRemoveEnd->first != end)) {
            TIterator containingEnd = completelyRemoveEnd;
            --containingEnd;
            if (containingEnd->second > end) {
                T& leftBorder = const_cast<T&>(containingEnd->first);

                Y_ASSERT(leftBorder < end);

                --completelyRemoveEnd; // Don't remove the whole interval.

                // Optimization hack.
                elementsRemoved += end - leftBorder;
                leftBorder = end; // OK to change key since intervals do not intersect.
            }
        }

        for (TIterator i = completelyRemoveBegin; i != completelyRemoveEnd; ++i) {
            elementsRemoved += i->second - i->first;
        }

        Tree.erase(completelyRemoveBegin, completelyRemoveEnd);

        Y_ASSERT(elementsRemoved <= NumElements);
        NumElements -= elementsRemoved;

        return elementsRemoved;
    }

    void Swap(TDisjointIntervalTree& rhv) {
        Tree.swap(rhv.Tree);
        std::swap(NumElements, rhv.NumElements);
    }

    void Clear() {
        Tree.clear();
        NumElements = 0;
    }

    bool Empty() const {
        return Tree.empty();
    }

    size_t GetNumElements() const {
        return NumElements;
    }

    size_t GetNumIntervals() const {
        return Tree.size();
    }

    T Min() const {
        Y_ASSERT(!Empty());
        return Tree.begin()->first;
    }

    T Max() const {
        Y_ASSERT(!Empty());
        return Tree.rbegin()->second;
    }

    TConstIterator begin() const {
        return Tree.begin();
    }

    TConstIterator end() const {
        return Tree.end();
    }

private:
    void InsertIntervalImpl(const T begin, const T end) {
        Y_ASSERT(begin < end);
        Y_ASSERT(!Intersects(begin, end));

        TIterator l = Tree.lower_bound(begin);
        TIterator p = Tree.end();
        if (l != Tree.begin()) {
            p = l;
            --p;
        }

#ifndef NDEBUG
        TIterator u = Tree.upper_bound(begin);
        Y_DEBUG_ABORT_UNLESS(u == Tree.end() || u->first >= end, "Trying to add [%" PRIu64 ", %" PRIu64 ") which intersects with existing [%" PRIu64 ", %" PRIu64 ")", begin, end, u->first, u->second);
        Y_DEBUG_ABORT_UNLESS(l == Tree.end() || l == u, "Trying to add [%" PRIu64 ", %" PRIu64 ") which intersects with existing [%" PRIu64 ", %" PRIu64 ")", begin, end, l->first, l->second);
        Y_DEBUG_ABORT_UNLESS(p == Tree.end() || p->second <= begin, "Trying to add [%" PRIu64 ", %" PRIu64 ") which intersects with existing [%" PRIu64 ", %" PRIu64 ")", begin, end, p->first, p->second);
#endif

        // try to extend interval
        if (p != Tree.end() && p->second == begin) {
            p->second = end;
            //Try to merge 2 intervals - p and next one if possible
            auto next = p;
            // Next is not Tree.end() here.
            ++next;
            if (next != Tree.end() && next->first == end) {
                p->second = next->second;
                Tree.erase(next);
            }
        // Maybe new interval extends right interval
        } else if (l != Tree.end() && end == l->first) {
            T& leftBorder = const_cast<T&>(l->first);
            // Optimization hack.
            leftBorder = begin; // OK to change key since intervals do not intersect.
        } else {
            Tree.insert(std::make_pair(begin, end));
        }
    }

    TIterator FindContaining(const T t) {
        TIterator l = Tree.lower_bound(t);
        if (l != Tree.end()) {
            if (l->first == t) {
                return l;
            }
            Y_ASSERT(l->first > t);

            if (l == Tree.begin()) {
                return Tree.end();
            }

            --l;
            Y_ASSERT(l->first != t);

            if (l->first < t && t < l->second) {
                return l;
            }

        } else if (!Tree.empty()) { // l is larger than Begin of any interval, but maybe it belongs to last interval?
            TReverseIterator last = Tree.rbegin();
            Y_ASSERT(last->first != t);

            if (last->first < t && t < last->second) {
                return (++last).base();
            }
        }
        return Tree.end();
    }
};
