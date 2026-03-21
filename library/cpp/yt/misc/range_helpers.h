#pragma once

#include <library/cpp/iterator/zip.h>

#include <ranges>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

namespace NDetail {

template <class TContainer>
struct TRangeToTag
{ };

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

//! An equivalent of Python's `zip()`, but resulting range consists of tuples
//! of pointers and has length equal to the length of the shortest container.
//! Implementation with mutable references depends on "lifetime extension in
//! range-based for loops" from C++23.
template <std::ranges::range... TRanges>
auto ZipMutable(TRanges&&... ranges);

//! Converts the provided range to the specified container.
//! This is a simplified equivalent of std::ranges::to from ranges-v3.
template <class TContainer, std::ranges::input_range TRange>
auto RangeTo(TRange&& range);

//! Range to for monadic operations
template <class TContainer>
constexpr auto RangeTo();

//! Monadic operations to use RangeTo. Example:
//! auto filteredHashSet = vec | std::views::filter(pred) | RangeTo<THashSet<int>>();
template<std::ranges::input_range TRange, class TContainer>
auto operator|(TRange&& range, NDetail::TRangeToTag<TContainer>);

//! Converts a parameter pack into the specified container.
//! Useful for constructing containers of move-only types.
//! Note that `std::vector<TMoveOnly>{std::move(a), std::move(b)}`
//! will not compile since std::initializer_list has only const iterators.
//! However, `StaticRangeTo<std::vector<TMoveOnly>>(std::move(a), std::move(b))` will work.
template <class TContainer, class... TValues>
    requires (std::constructible_from<typename TContainer::value_type, TValues> && ...)
TContainer StaticRangeTo(TValues... values);

//! A tuple wrapper with implicit casts to containers via `StaticRangeTo`.
//! Useful for container list-initialization e.g. `std::vector<TMoveOnly> foo = TStaticRange{std::move(bar)};`.
template <class... TValues>
struct TStaticRange;

//! Shortcut for `RangeTo(std::ranges::views::transform)`.
template <class TContainer, std::ranges::input_range TRange, class TTransformFunction>
auto TransformRangeTo(TRange&& range, TTransformFunction&& function);

//! An equivalent of std::ranges::fold_left from ranges-v3.
template <std::ranges::range TRange, class TOperation, class TProjection = std::identity>
auto FoldRange(TRange&& range, TOperation operation, TProjection projection = {});

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define RANGE_HELPERS_INL_H_
#include "range_helpers-inl.h"
#undef RANGE_HELPERS_INL_H_
