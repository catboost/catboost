#pragma once

#include <util/system/yassert.h>

#include <iterator>
#include <utility>

template <typename TBegin, typename TEnd = TBegin>
struct TIteratorRange {
    using TElement = std::remove_reference_t<decltype(*std::declval<TBegin>())>;

    TIteratorRange(TBegin begin, TEnd end)
        : Begin_(begin)
        , End_(end)
    {
    }

    TIteratorRange()
        : TIteratorRange(TBegin{}, TEnd{})
    {
    }

    TBegin begin() const {
        return Begin_;
    }

    TEnd end() const {
        return End_;
    }

    bool empty() const {
        // because range based for requires exactly '!='
        return !(Begin_ != End_);
    }

private:
    TBegin Begin_;
    TEnd End_;
};

template <class TIterator>
class TIteratorRange<TIterator, TIterator> {
public:
    using iterator = TIterator;
    using const_iterator = TIterator;
    using value_type = typename std::iterator_traits<iterator>::value_type;
    using reference = typename std::iterator_traits<iterator>::reference;
    using const_reference = typename std::iterator_traits<const_iterator>::reference;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = std::size_t;

    TIteratorRange()
        : Begin_()
        , End_()
    {
    }

    TIteratorRange(TIterator begin, TIterator end)
        : Begin_(begin)
        , End_(end)
    {
    }

    TIterator begin() const {
        return Begin_;
    }

    TIterator end() const {
        return End_;
    }

    Y_PURE_FUNCTION bool empty() const {
        return Begin_ == End_;
    }

    size_type size() const {
        return End_ - Begin_;
    }

    reference operator[](size_t at) const {
        Y_ASSERT(at < size());

        return *(Begin_ + at);
    }

private:
    TIterator Begin_;
    TIterator End_;
};

template <class TIterator>
TIteratorRange<TIterator> MakeIteratorRange(TIterator begin, TIterator end) {
    return TIteratorRange<TIterator>(begin, end);
}

template <class TIterator>
TIteratorRange<TIterator> MakeIteratorRange(const std::pair<TIterator, TIterator>& range) {
    return TIteratorRange<TIterator>(range.first, range.second);
}

template <class TBegin, class TEnd>
TIteratorRange<TBegin, TEnd> MakeIteratorRange(TBegin begin, TEnd end) {
    return {begin, end};
}
