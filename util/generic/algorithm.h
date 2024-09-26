#pragma once

#include "is_in.h"
#include "utility.h"

#include <util/system/defaults.h>
#include <util/generic/fwd.h>

#include <numeric>
#include <algorithm>
#include <iterator>
#include <utility>

namespace NPrivate {
    template <class I, class F, class P>
    constexpr I ExtremeElementBy(I begin, I end, F&& func, P&& pred) {
        if (begin == end) {
            return end;
        }

        auto bestValue = func(*begin);
        auto bestPos = begin;

        for (++begin; begin != end; ++begin) {
            auto curValue = func(*begin);
            if (pred(curValue, bestValue)) {
                bestValue = std::move(curValue);
                bestPos = begin;
            }
        }

        return bestPos;
    }
} // namespace NPrivate

template <class T>
constexpr void Sort(T f, T l) {
    std::sort(f, l);
}

template <class T, class C>
constexpr void Sort(T f, T l, C c) {
    std::sort(f, l, c);
}

template <class TContainer>
constexpr void Sort(TContainer& container) {
    Sort(container.begin(), container.end());
}

template <class TContainer, typename TCompare>
constexpr void Sort(TContainer& container, TCompare compare) {
    Sort(container.begin(), container.end(), compare);
}

template <class TIterator, typename TGetKey>
constexpr void SortBy(TIterator begin, TIterator end, const TGetKey& getKey) {
    Sort(begin, end, [&](auto&& left, auto&& right) { return getKey(left) < getKey(right); });
}

template <class TContainer, typename TGetKey>
constexpr void SortBy(TContainer& container, const TGetKey& getKey) {
    SortBy(container.begin(), container.end(), getKey);
}

template <class T>
static inline void StableSort(T f, T l) {
    std::stable_sort(f, l);
}

template <class T, class C>
static inline void StableSort(T f, T l, C c) {
    std::stable_sort(f, l, c);
}

template <class TContainer>
static inline void StableSort(TContainer& container) {
    StableSort(container.begin(), container.end());
}

template <class TContainer, typename TCompare>
static inline void StableSort(TContainer& container, TCompare compare) {
    StableSort(container.begin(), container.end(), compare);
}

template <class TIterator, typename TGetKey>
static inline void StableSortBy(TIterator begin, TIterator end, const TGetKey& getKey) {
    StableSort(begin, end, [&](auto&& left, auto&& right) { return getKey(left) < getKey(right); });
}

template <class TContainer, typename TGetKey>
static inline void StableSortBy(TContainer& container, const TGetKey& getKey) {
    StableSortBy(container.begin(), container.end(), getKey);
}

template <class T>
constexpr void PartialSort(T f, T m, T l) {
    std::partial_sort(f, m, l);
}

template <class T, class C>
constexpr void PartialSort(T f, T m, T l, C c) {
    std::partial_sort(f, m, l, c);
}

template <class T, class R>
constexpr R PartialSortCopy(T f, T l, R of, R ol) {
    return std::partial_sort_copy(f, l, of, ol);
}

template <class T, class R, class C>
constexpr R PartialSortCopy(T f, T l, R of, R ol, C c) {
    return std::partial_sort_copy(f, l, of, ol, c);
}

template <class I, class T>
constexpr I Find(I f, I l, const T& v) {
    return std::find(f, l, v);
}

template <class C, class T>
constexpr auto Find(C&& c, const T& v) {
    using std::begin;
    using std::end;

    return std::find(begin(c), end(c), v);
}

// FindPtr - return NULL if not found. Works for arrays, containers, iterators
template <class I, class T>
constexpr auto FindPtr(I f, I l, const T& v) -> decltype(&*f) {
    I found = Find(f, l, v);
    return (found != l) ? &*found : nullptr;
}

template <class C, class T>
constexpr auto FindPtr(C&& c, const T& v) {
    using std::begin;
    using std::end;
    return FindPtr(begin(c), end(c), v);
}

template <class I, class P>
constexpr I FindIf(I f, I l, P p) {
    return std::find_if(f, l, p);
}

template <class C, class P>
constexpr auto FindIf(C&& c, P p) {
    using std::begin;
    using std::end;

    return FindIf(begin(c), end(c), p);
}

template <class I, class P>
constexpr bool AllOf(I f, I l, P pred) {
    return std::all_of(f, l, pred);
}

template <class C, class P>
constexpr bool AllOf(const C& c, P pred) {
    using std::begin;
    using std::end;
    return AllOf(begin(c), end(c), pred);
}

template <class I, class P>
constexpr bool AnyOf(I f, I l, P pred) {
    return std::any_of(f, l, pred);
}

template <class C, class P>
constexpr bool AnyOf(const C& c, P pred) {
    using std::begin;
    using std::end;
    return AnyOf(begin(c), end(c), pred);
}

// FindIfPtr - return NULL if not found. Works for arrays, containers, iterators
template <class I, class P>
constexpr auto FindIfPtr(I f, I l, P pred) -> decltype(&*f) {
    I found = FindIf(f, l, pred);
    return (found != l) ? &*found : nullptr;
}

template <class C, class P>
constexpr auto FindIfPtr(C&& c, P pred) {
    using std::begin;
    using std::end;
    return FindIfPtr(begin(c), end(c), pred);
}

template <class C, class T>
constexpr size_t FindIndex(C&& c, const T& x) {
    using std::begin;
    using std::end;
    auto it = Find(begin(c), end(c), x);
    return it == end(c) ? NPOS : (it - begin(c));
}

template <class C, class P>
constexpr size_t FindIndexIf(C&& c, P p) {
    using std::begin;
    using std::end;
    auto it = FindIf(begin(c), end(c), p);
    return it == end(c) ? NPOS : (it - begin(c));
}

// EqualToOneOf(x, "apple", "orange") means (x == "apple" || x == "orange")
template <typename T, typename... Other>
constexpr bool EqualToOneOf(const T& x, const Other&... values) {
    return (... || (x == values));
}

template <typename T, typename... Other>
constexpr size_t CountOf(const T& x, const Other&... values) {
    return (0 + ... + static_cast<size_t>(x == values));
}

template <class I>
constexpr void PushHeap(I f, I l) {
    std::push_heap(f, l);
}

template <class I, class C>
constexpr void PushHeap(I f, I l, C c) {
    std::push_heap(f, l, c);
}

template <class I>
constexpr void PopHeap(I f, I l) {
    std::pop_heap(f, l);
}

template <class I, class C>
constexpr void PopHeap(I f, I l, C c) {
    std::pop_heap(f, l, c);
}

template <class I>
constexpr void MakeHeap(I f, I l) {
    std::make_heap(f, l);
}

template <class I, class C>
constexpr void MakeHeap(I f, I l, C c) {
    std::make_heap(f, l, c);
}

template <class I>
constexpr void SortHeap(I f, I l) {
    std::sort_heap(f, l);
}

template <class I, class C>
constexpr void SortHeap(I f, I l, C c) {
    std::sort_heap(f, l, c);
}

template <class I, class T>
constexpr I LowerBound(I f, I l, const T& v) {
    return std::lower_bound(f, l, v);
}

template <class I, class T, class C>
constexpr I LowerBound(I f, I l, const T& v, C c) {
    return std::lower_bound(f, l, v, c);
}

template <class I, class T, class TGetKey>
constexpr I LowerBoundBy(I f, I l, const T& v, const TGetKey& getKey) {
    return std::lower_bound(f, l, v, [&](auto&& left, auto&& right) { return getKey(left) < right; });
}

template <class I, class T>
constexpr I UpperBound(I f, I l, const T& v) {
    return std::upper_bound(f, l, v);
}

template <class I, class T, class C>
constexpr I UpperBound(I f, I l, const T& v, C c) {
    return std::upper_bound(f, l, v, c);
}

template <class I, class T, class TGetKey>
constexpr I UpperBoundBy(I f, I l, const T& v, const TGetKey& getKey) {
    return std::upper_bound(f, l, v, [&](auto&& left, auto&& right) { return left < getKey(right); });
}

template <class T>
constexpr T Unique(T f, T l) {
    return std::unique(f, l);
}

template <class T, class P>
constexpr T Unique(T f, T l, P p) {
    return std::unique(f, l, p);
}

template <class T, class TGetKey>
constexpr T UniqueBy(T f, T l, const TGetKey& getKey) {
    return Unique(f, l, [&](auto&& left, auto&& right) { return getKey(left) == getKey(right); });
}

template <class C>
void SortUnique(C& c) {
    Sort(c.begin(), c.end());
    c.erase(Unique(c.begin(), c.end()), c.end());
}

template <class C, class Cmp>
void SortUnique(C& c, Cmp cmp) {
    Sort(c.begin(), c.end(), cmp);
    c.erase(Unique(c.begin(), c.end()), c.end());
}

template <class C, class TGetKey>
void SortUniqueBy(C& c, const TGetKey& getKey) {
    SortBy(c, getKey);
    c.erase(UniqueBy(c.begin(), c.end(), getKey), c.end());
}

template <class C, class TGetKey>
void StableSortUniqueBy(C& c, const TGetKey& getKey) {
    StableSortBy(c, getKey);
    c.erase(UniqueBy(c.begin(), c.end(), getKey), c.end());
}

template <class C, class TValue>
void Erase(C& c, const TValue& value) {
    c.erase(std::remove(c.begin(), c.end(), value), c.end());
}

template <class C, class P>
void EraseIf(C& c, P p) {
    c.erase(std::remove_if(c.begin(), c.end(), p), c.end());
}

template <class C, class P>
void EraseNodesIf(C& c, P p) {
    for (auto iter = c.begin(), last = c.end(); iter != last;) {
        if (p(*iter)) {
            c.erase(iter++);
        } else {
            ++iter;
        }
    }
}

template <class T1, class T2>
constexpr bool Equal(T1 f1, T1 l1, T2 f2) {
    return std::equal(f1, l1, f2);
}

template <class T1, class T2, class P>
constexpr bool Equal(T1 f1, T1 l1, T2 f2, P p) {
    return std::equal(f1, l1, f2, p);
}

template <class TI, class TO>
constexpr TO Copy(TI f, TI l, TO t) {
    return std::copy(f, l, t);
}

template <class TI, class TO>
constexpr TO UniqueCopy(TI f, TI l, TO t) {
    return std::unique_copy(f, l, t);
}

template <class TI, class TO, class TP>
constexpr TO UniqueCopy(TI f, TI l, TO t, TP p) {
    return std::unique_copy(f, l, t, p);
}

template <class TI, class TO, class TP>
constexpr TO RemoveCopyIf(TI f, TI l, TO t, TP p) {
    return std::remove_copy_if(f, l, t, p);
}

template <class TI, class TO>
constexpr TO ReverseCopy(TI f, TI l, TO t) {
    return std::reverse_copy(f, l, t);
}

template <class TI1, class TI2, class TO>
constexpr TO SetUnion(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p) {
    return std::set_union(f1, l1, f2, l2, p);
}

template <class TI1, class TI2, class TO, class TC>
constexpr TO SetUnion(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p, TC c) {
    return std::set_union(f1, l1, f2, l2, p, c);
}

template <class TI1, class TI2, class TO>
constexpr TO SetDifference(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p) {
    return std::set_difference(f1, l1, f2, l2, p);
}

template <class TI1, class TI2, class TO, class TC>
constexpr TO SetDifference(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p, TC c) {
    return std::set_difference(f1, l1, f2, l2, p, c);
}

template <class TI1, class TI2, class TO>
constexpr TO SetSymmetricDifference(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p) {
    return std::set_symmetric_difference(f1, l1, f2, l2, p);
}

template <class TI1, class TI2, class TO, class TC>
constexpr TO SetSymmetricDifference(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p, TC c) {
    return std::set_symmetric_difference(f1, l1, f2, l2, p, c);
}

template <class TI1, class TI2, class TO>
constexpr TO SetIntersection(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p) {
    return std::set_intersection(f1, l1, f2, l2, p);
}

template <class TI1, class TI2, class TO, class TC>
constexpr TO SetIntersection(TI1 f1, TI1 l1, TI2 f2, TI2 l2, TO p, TC c) {
    return std::set_intersection(f1, l1, f2, l2, p, c);
}

template <class I, class T>
constexpr void Fill(I f, I l, const T& v) {
    std::fill(f, l, v);
}

template <typename I, typename S, typename T>
constexpr I FillN(I f, S n, const T& v) {
    return std::fill_n(f, n, v);
}

template <class T>
constexpr void Reverse(T f, T l) {
    std::reverse(f, l);
}

template <class T>
constexpr void Rotate(T f, T m, T l) {
    std::rotate(f, m, l);
}

template <typename It, typename Val>
constexpr Val Accumulate(It begin, It end, Val val) {
    // std::move since C++20
    return std::accumulate(begin, end, std::move(val));
}

template <typename It, typename Val, typename BinOp>
constexpr Val Accumulate(It begin, It end, Val val, BinOp binOp) {
    // std::move since C++20
    return std::accumulate(begin, end, std::move(val), binOp);
}

template <typename C, typename Val>
constexpr Val Accumulate(const C& c, Val val) {
    // std::move since C++20
    return Accumulate(std::begin(c), std::end(c), std::move(val));
}

template <typename C, typename Val, typename BinOp>
constexpr Val Accumulate(const C& c, Val val, BinOp binOp) {
    // std::move since C++20
    return Accumulate(std::begin(c), std::end(c), std::move(val), binOp);
}

template <typename It1, typename It2, typename Val>
constexpr Val InnerProduct(It1 begin1, It1 end1, It2 begin2, Val val) {
    return std::inner_product(begin1, end1, begin2, val);
}

template <typename It1, typename It2, typename Val, typename BinOp1, typename BinOp2>
constexpr Val InnerProduct(It1 begin1, It1 end1, It2 begin2, Val val, BinOp1 binOp1, BinOp2 binOp2) {
    return std::inner_product(begin1, end1, begin2, val, binOp1, binOp2);
}

template <typename TVectorType>
constexpr typename TVectorType::value_type InnerProduct(const TVectorType& lhs, const TVectorType& rhs, typename TVectorType::value_type val = typename TVectorType::value_type()) {
    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), val);
}

template <typename TVectorType, typename BinOp1, typename BinOp2>
constexpr typename TVectorType::value_type InnerProduct(const TVectorType& lhs, const TVectorType& rhs, typename TVectorType::value_type val, BinOp1 binOp1, BinOp2 binOp2) {
    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), val, binOp1, binOp2);
}

template <class T>
constexpr T MinElement(T begin, T end) {
    return std::min_element(begin, end);
}

template <class T, class C>
constexpr T MinElement(T begin, T end, C comp) {
    return std::min_element(begin, end, comp);
}

template <class T>
constexpr T MaxElement(T begin, T end) {
    return std::max_element(begin, end);
}

template <class T, class C>
constexpr T MaxElement(T begin, T end, C comp) {
    return std::max_element(begin, end, comp);
}

template <class I, class F>
constexpr I MaxElementBy(I begin, I end, F&& func) {
    using TValue = decltype(func(*begin));
    return ::NPrivate::ExtremeElementBy(begin, end, std::forward<F>(func), TGreater<TValue>());
}

template <class C, class F>
constexpr auto MaxElementBy(C& c, F&& func) {
    return MaxElementBy(std::begin(c), std::end(c), std::forward<F>(func));
}

template <class C, class F>
constexpr auto MaxElementBy(const C& c, F&& func) {
    return MaxElementBy(std::begin(c), std::end(c), std::forward<F>(func));
}

template <class I, class F>
constexpr I MinElementBy(I begin, I end, F&& func) {
    using TValue = decltype(func(*begin));
    return ::NPrivate::ExtremeElementBy(begin, end, std::forward<F>(func), TLess<TValue>());
}

template <class C, class F>
constexpr auto MinElementBy(C& c, F&& func) {
    return MinElementBy(std::begin(c), std::end(c), std::forward<F>(func));
}

template <class C, class F>
constexpr auto MinElementBy(const C& c, F&& func) {
    return MinElementBy(std::begin(c), std::end(c), std::forward<F>(func));
}

template <class TOp, class... TArgs>
void ApplyToMany(TOp op, TArgs&&... args) {
    int dummy[] = {((void)op(std::forward<TArgs>(args)), 0)...};
    Y_UNUSED(dummy);
}

template <class TI, class TOp>
constexpr void ForEach(TI f, TI l, TOp op) {
    std::for_each(f, l, op);
}

namespace NPrivate {
    template <class T, class TOp, size_t... Is>
    constexpr bool AllOfImpl(T&& t, TOp&& op, std::index_sequence<Is...>) {
#if _LIBCPP_STD_VER >= 17
        return (true && ... && op(std::get<Is>(std::forward<T>(t))));
#else
        bool result = true;
        auto wrapper = [&result, &op](auto&& x) { result = result && op(std::forward<decltype(x)>(x)); };
        int dummy[] = {(wrapper(std::get<Is>(std::forward<T>(t))), 0)...};
        Y_UNUSED(dummy);
        return result;
#endif
    }

    template <class T, class TOp, size_t... Is>
    constexpr bool AnyOfImpl(T&& t, TOp&& op, std::index_sequence<Is...>) {
#if _LIBCPP_STD_VER >= 17
        return (false || ... || op(std::get<Is>(std::forward<T>(t))));
#else
        bool result = false;
        auto wrapper = [&result, &op](auto&& x) { result = result || op(std::forward<decltype(x)>(x)); };
        int dummy[] = {(wrapper(std::get<Is>(std::forward<T>(t))), 0)...};
        Y_UNUSED(dummy);
        return result;
#endif
    }

    template <class T, class TOp, size_t... Is>
    constexpr void ForEachImpl(T&& t, TOp&& op, std::index_sequence<Is...>) {
#if _LIBCPP_STD_VER >= 17
        (..., op(std::get<Is>(std::forward<T>(t))));
#else
        ::ApplyToMany(std::forward<TOp>(op), std::get<Is>(std::forward<T>(t))...);
#endif
    }
} // namespace NPrivate

// check that TOp return true for all of element from tuple T
template <class T, class TOp>
constexpr ::TEnableIfTuple<T, bool> AllOf(T&& t, TOp&& op) {
    return ::NPrivate::AllOfImpl(
        std::forward<T>(t),
        std::forward<TOp>(op),
        std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>{});
}

// check that TOp return true for at least one element from tuple T
template <class T, class TOp>
constexpr ::TEnableIfTuple<T, bool> AnyOf(T&& t, TOp&& op) {
    return ::NPrivate::AnyOfImpl(
        std::forward<T>(t),
        std::forward<TOp>(op),
        std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>{});
}

template <class T, class TOp>
constexpr ::TEnableIfTuple<T> ForEach(T&& t, TOp&& op) {
    ::NPrivate::ForEachImpl(
        std::forward<T>(t),
        std::forward<TOp>(op),
        std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>{});
}

template <class T1, class T2, class O>
constexpr void Transform(T1 b, T1 e, T2 o, O f) {
    std::transform(b, e, o, f);
}

template <class T1, class T2, class T3, class O>
constexpr void Transform(T1 b1, T1 e1, T2 b2, T3 o, O f) {
    std::transform(b1, e1, b2, o, f);
}

template <class T, class V>
constexpr typename std::iterator_traits<T>::difference_type Count(T first, T last, const V& value) {
    return std::count(first, last, value);
}

template <class TContainer, class TValue>
constexpr auto Count(const TContainer& container, const TValue& value) {
    return Count(std::cbegin(container), std::cend(container), value);
}

template <class It, class P>
constexpr auto CountIf(It first, It last, P p) {
    return std::count_if(first, last, p);
}

template <class C, class P>
constexpr auto CountIf(const C& c, P pred) {
    using std::begin;
    using std::end;
    return CountIf(begin(c), end(c), pred);
}

template <class I1, class I2>
constexpr std::pair<I1, I2> Mismatch(I1 b1, I1 e1, I2 b2) {
    return std::mismatch(b1, e1, b2);
}

template <class I1, class I2, class P>
constexpr std::pair<I1, I2> Mismatch(I1 b1, I1 e1, I2 b2, P p) {
    return std::mismatch(b1, e1, b2, p);
}

template <class RandomIterator>
constexpr void NthElement(RandomIterator begin, RandomIterator nth, RandomIterator end) {
    std::nth_element(begin, nth, end);
}

template <class RandomIterator, class Compare>
constexpr void NthElement(RandomIterator begin, RandomIterator nth, RandomIterator end, Compare compare) {
    std::nth_element(begin, nth, end, compare);
}

// no standard implementation until C++14
template <class I1, class I2>
constexpr std::pair<I1, I2> Mismatch(I1 b1, I1 e1, I2 b2, I2 e2) {
    while (b1 != e1 && b2 != e2 && *b1 == *b2) {
        ++b1;
        ++b2;
    }
    return std::make_pair(b1, b2);
}

template <class I1, class I2, class P>
constexpr std::pair<I1, I2> Mismatch(I1 b1, I1 e1, I2 b2, I2 e2, P p) {
    while (b1 != e1 && b2 != e2 && p(*b1, *b2)) {
        ++b1;
        ++b2;
    }
    return std::make_pair(b1, b2);
}

template <class It, class Val>
constexpr bool BinarySearch(It begin, It end, const Val& val) {
    return std::binary_search(begin, end, val);
}

template <class It, class Val, class Comp>
constexpr bool BinarySearch(It begin, It end, const Val& val, Comp comp) {
    return std::binary_search(begin, end, val, comp);
}

template <class It, class Val>
constexpr std::pair<It, It> EqualRange(It begin, It end, const Val& val) {
    return std::equal_range(begin, end, val);
}

template <class It, class Val, class Comp>
constexpr std::pair<It, It> EqualRange(It begin, It end, const Val& val, Comp comp) {
    return std::equal_range(begin, end, val, comp);
}

template <class TContainer>
constexpr auto AdjacentFind(TContainer&& c) {
    using std::begin;
    using std::end;
    return std::adjacent_find(begin(c), end(c));
}

template <class TContainer, class Compare>
constexpr auto AdjacentFind(TContainer&& c, Compare comp) {
    using std::begin;
    using std::end;
    return std::adjacent_find(begin(c), end(c), comp);
}

namespace NPrivate {
    template <class TForwardIterator, class TGetKey>
    constexpr TForwardIterator AdjacentFindBy(TForwardIterator begin, TForwardIterator end, const TGetKey& getKey) {
        return std::adjacent_find(begin, end, [&](auto&& left, auto&& right) { return getKey(left) == getKey(right); });
    }
} // namespace NPrivate

template <class TContainer, class TGetKey>
constexpr auto AdjacentFindBy(TContainer&& c, const TGetKey& getKey) {
    using std::begin;
    using std::end;
    return ::NPrivate::AdjacentFindBy(begin(c), end(c), getKey);
}

template <class ForwardIt>
constexpr bool IsSorted(ForwardIt begin, ForwardIt end) {
    return std::is_sorted(begin, end);
}

template <class ForwardIt, class Compare>
constexpr bool IsSorted(ForwardIt begin, ForwardIt end, Compare comp) {
    return std::is_sorted(begin, end, comp);
}

template <class TIterator, typename TGetKey>
constexpr bool IsSortedBy(TIterator begin, TIterator end, const TGetKey& getKey) {
    return IsSorted(begin, end, [&](auto&& left, auto&& right) { return getKey(left) < getKey(right); });
}

template <class TContainer, typename TGetKey>
constexpr bool IsSortedBy(const TContainer& c, const TGetKey& getKey) {
    using std::begin;
    using std::end;
    return IsSortedBy(begin(c), end(c), getKey);
}

template <class It, class Val>
constexpr void Iota(It begin, It end, Val val) {
    std::iota(begin, end, val);
}

template <class TI, class TO, class S>
constexpr TO CopyN(TI from, S s, TO to) {
    return std::copy_n(from, s, to);
}

template <class TI, class TO, class P>
constexpr TO CopyIf(TI begin, TI end, TO to, P pred) {
    return std::copy_if(begin, end, to, pred);
}

template <class T>
constexpr std::pair<const T&, const T&> MinMax(const T& first Y_LIFETIME_BOUND, const T& second Y_LIFETIME_BOUND) {
    return std::minmax(first, second);
}

template <class It>
constexpr std::pair<It, It> MinMaxElement(It first, It last) {
    return std::minmax_element(first, last);
}

template <class TIterator, class TGenerator>
constexpr void Generate(TIterator first, TIterator last, TGenerator generator) {
    std::generate(first, last, generator);
}

template <class TIterator, class TSize, class TGenerator>
constexpr void GenerateN(TIterator first, TSize count, TGenerator generator) {
    std::generate_n(first, count, generator);
}
