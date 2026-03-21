#ifndef RANGE_HELPERS_INL_H_
#error "Direct inclusion of this file is not allowed, include range_helpers.h"
// For the sake of sane code completion.
#include "range_helpers.h"
#endif

namespace NYT {
namespace NDetail {

////////////////////////////////////////////////////////////////////////////////

template <class TContainer>
struct TAppendTo
{ };

template <class TContainer>
    requires requires (TContainer container, typename TContainer::value_type value) {
        container.push_back(value);
    }
struct TAppendTo<TContainer>
{
    template <class TValue>
    static void Append(TContainer& container, TValue&& value)
    {
        container.push_back(std::forward<TValue>(value));
    }
};

template <class TContainer>
    requires requires (TContainer container, typename TContainer::value_type value) {
        container.insert(value);
    }
struct TAppendTo<TContainer>
{
    template <class TValue>
    static void Append(TContainer& container, TValue&& value)
    {
        container.insert(std::forward<TValue>(value));
    }
};

////////////////////////////////////////////////////////////////////////////////

template <class TContainer>
struct TRangeTo
{ };

template <class TContainer>
    requires requires (TContainer container, typename TContainer::value_type value) {
        TAppendTo<TContainer>::Append(container, value);
    }
struct TRangeTo<TContainer>
{
    template <std::ranges::input_range TRange>
    static auto ToContainer(TRange&& range)
    {
        TContainer container;
        if constexpr (requires { std::ranges::size(range); } &&
            requires { container.reserve(std::declval<size_t>()); })
        {
            container.reserve(std::ranges::size(range));
        }

        for (auto&& element : range) {
            TAppendTo<TContainer>::Append(container, std::forward<decltype(element)>(element));
        }

        return container;
    }

    template <class... TValues>
    static auto StaticRangeToContainer(TValues... values)
    {
        TContainer container;
        if constexpr (requires { container.reserve(std::declval<size_t>()); })
        {
            container.reserve(sizeof...(TValues));
        }

        (TAppendTo<TContainer>::Append(container, std::forward<TValues>(values)), ...);
        return container;
    }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace NDetail

////////////////////////////////////////////////////////////////////////////////

template <std::ranges::range... TContainers>
auto ZipMutable(TContainers&&... containers) {
    return Zip(std::ranges::views::transform(containers, [] <class T> (T&& t) {
        return &t;
    })...);
}

template <class TContainer, std::ranges::input_range TRange>
auto RangeTo(TRange&& range)
{
    return NDetail::TRangeTo<TContainer>::template ToContainer<TRange>(std::forward<TRange>(range));
}

template <class TContainer>
constexpr auto RangeTo()
{
    return NDetail::TRangeToTag<TContainer>();
}

template<std::ranges::input_range TRange, class TContainer>
auto operator|(TRange&& range, NDetail::TRangeToTag<TContainer>)
{
    return RangeTo<TContainer>(std::forward<TRange>(range));
}

template <class TContainer, std::ranges::input_range TRange, class TTransformFunction>
auto TransformRangeTo(TRange&& range, TTransformFunction&& function)
{
    return RangeTo<TContainer>(std::ranges::views::transform(
        std::forward<TRange>(range),
        std::forward<TTransformFunction>(function)));
}

template <class TContainer, class... TValues>
    requires (std::constructible_from<typename TContainer::value_type, TValues> && ...)
TContainer StaticRangeTo(TValues... values)
{
    return NDetail::TRangeTo<TContainer>::template StaticRangeToContainer<TValues...>(std::forward<TValues>(values)...);
}

template <class... TValues>
struct TStaticRange
{
public:
    explicit TStaticRange(TValues... values)
        : Tuple_(std::forward<TValues>(values)...)
    { }

    template <class TContainer>
    operator TContainer() &&
    {
        return std::apply(&StaticRangeTo<TContainer, TValues...>, std::move(Tuple_));
    }

private:
    std::tuple<TValues...> Tuple_;
};

template <std::ranges::range TRange, class TOperation, class TProjection>
auto FoldRange(TRange&& range, TOperation operation, TProjection projection)
{
    auto iter = range.begin();
    if (iter == range.end()) {
        return std::remove_cvref_t<decltype(std::invoke(projection, *iter))>{};
    }
    auto accumulator = std::invoke(projection, *iter);
    for (++iter; iter != range.end(); ++iter) {
        accumulator = std::invoke(operation, accumulator, std::invoke(projection, *iter));
    }
    return accumulator;
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
