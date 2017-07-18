#pragma once

#include <iterator>
#include <utility>

template <class Iterator>
class TIteratorRange {
public:
    using iterator = Iterator;
    using const_iterator = Iterator;
    using size_type = std::size_t;

    TIteratorRange()
        : Begin_()
        , End_()
    {
    }

    TIteratorRange(Iterator begin, Iterator end)
        : Begin_(begin)
        , End_(end)
    {
    }

    Iterator begin() const {
        return Begin_;
    }

    Iterator end() const {
        return End_;
    }

    bool empty() const {
        return Begin_ == End_;
    }

    size_type size() const {
        return std::distance(Begin_, End_);
    }

private:
    Iterator Begin_;
    Iterator End_;
};

template <class Iterator>
TIteratorRange<Iterator> MakeIteratorRange(Iterator begin, Iterator end) {
    return TIteratorRange<Iterator>(begin, end);
}

template <class Iterator>
TIteratorRange<Iterator> MakeIteratorRange(const std::pair<Iterator, Iterator>& range) {
    return TIteratorRange<Iterator>(range.first, range.second);
}
