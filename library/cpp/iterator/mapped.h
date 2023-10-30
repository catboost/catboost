#pragma once

#include <util/generic/iterator_range.h>
#include <util/generic/store_policy.h>

#include <iterator>


namespace NIteratorPrivate {
    template <class TIterator>
    constexpr bool HasRandomAccess() {
        return std::is_same_v<typename std::iterator_traits<TIterator>::iterator_category,
                              std::random_access_iterator_tag>;
    }
}


template <class TIterator, class TMapper>
class TMappedIterator {
protected:
    using TSelf = TMappedIterator<TIterator, TMapper>;
    using TSrcPointerType = typename std::iterator_traits<TIterator>::reference;
    using TInvokeResult = std::invoke_result_t<TMapper, TSrcPointerType>;
    using TValue = std::remove_reference_t<TInvokeResult>;
public:
    using difference_type = std::ptrdiff_t;
    using value_type = TValue;
    using reference = TValue&;
    using const_reference = const TValue&;
    using pointer = std::remove_reference_t<TValue>*;
    using iterator_category = std::conditional_t<NIteratorPrivate::HasRandomAccess<TIterator>(),
        std::random_access_iterator_tag, std::input_iterator_tag>;

    TMappedIterator(TIterator it, TMapper mapper)
        : Iter(it)
        , Mapper(std::move(mapper))
    {
    }

    TSelf& operator++() {
        ++Iter;
        return *this;
    }
    TSelf& operator--() {
        --Iter;
        return *this;
    }
    TInvokeResult operator*() {
        return Mapper((*Iter));
    }
    TInvokeResult operator*() const {
        return Mapper((*Iter));
    }

    pointer operator->() const {
        return &(Mapper((*Iter)));
    }

    TInvokeResult operator[](difference_type n) const {
        return Mapper(*(Iter + n));
    }
    TSelf& operator+=(difference_type n) {
        Iter += n;
        return *this;
    }
    TSelf& operator-=(difference_type n) {
        Iter -= n;
        return *this;
    }
    TSelf operator+(difference_type n) const {
        return TSelf(Iter + n, Mapper);
    }
    TSelf operator-(difference_type n) const {
        return TSelf(Iter - n, Mapper);
    }
    difference_type operator-(const TSelf& other) const {
        return Iter - other.Iter;
    }
    bool operator==(const TSelf& other) const {
        return Iter == other.Iter;
    }
    bool operator!=(const TSelf& other) const {
        return Iter != other.Iter;
    }
    bool operator>(const TSelf& other) const {
        return Iter > other.Iter;
    }
    bool operator<(const TSelf& other) const {
        return Iter < other.Iter;
    }

private:
    TIterator Iter;
    TMapper Mapper;
};


template <class TContainer, class TMapper>
class TInputMappedRange {
protected:
    using TContainerStorage = TAutoEmbedOrPtrPolicy<TContainer>;
    using TMapperStorage = TAutoEmbedOrPtrPolicy<TMapper>;
    using TMapperWrapper = std::reference_wrapper<std::remove_reference_t<TMapper>>;
    using TInternalIterator = decltype(std::begin(std::declval<TContainer&>()));
    using TIterator = TMappedIterator<TInternalIterator, TMapperWrapper>;
public:
    using iterator = TIterator;
    using const_iterator = TIterator;
    using value_type = typename TIterator::value_type;
    using reference = typename TIterator::reference;
    using const_reference = typename TIterator::const_reference;

    TInputMappedRange(TContainer&& container, TMapper&& mapper)
        : Container(std::forward<TContainer>(container))
        , Mapper(std::forward<TMapper>(mapper))
    {
    }

    TIterator begin() const {
        return {std::begin(*Container.Ptr()), {*Mapper.Ptr()}};
    }

    TIterator end() const {
        return {std::end(*Container.Ptr()), {*Mapper.Ptr()}};
    }

    bool empty() const {
        return std::begin(*Container.Ptr()) == std::end(*Container.Ptr());
    }

protected:
    mutable TContainerStorage Container;
    mutable TMapperStorage Mapper;
};


template <class TContainer, class TMapper>
class TRandomAccessMappedRange : public TInputMappedRange<TContainer, TMapper> {
    using TBase = TInputMappedRange<TContainer, TMapper>;
    using TInternalIterator = typename TBase::TInternalIterator;
    using TIterator = typename TBase::TIterator;
public:
    using iterator = typename TBase::iterator;
    using const_iterator = typename TBase::const_iterator;
    using value_type = typename TBase::value_type;
    using reference = typename TBase::reference;
    using const_reference = typename TBase::const_reference;

    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = std::size_t;

    TRandomAccessMappedRange(TContainer&& container, TMapper&& mapper)
        : TBase(std::forward<TContainer>(container), std::forward<TMapper>(mapper))
    {
    }

    using TBase::begin;
    using TBase::end;
    using TBase::empty;

    size_type size() const {
        return std::end(*this->Container.Ptr()) - std::begin(*this->Container.Ptr());
    }

    const_reference operator[](size_t at) const {
        Y_ASSERT(at < this->size());

        return *(this->begin() + at);
    }

    reference operator[](size_t at) {
        Y_ASSERT(at < this->size());

        return *(this->begin() + at);
    }
};

template <class TIterator, class TMapper>
TMappedIterator<TIterator, TMapper> MakeMappedIterator(TIterator iter, TMapper mapper) {
    return {iter, mapper};
}

template <class TIterator, class TMapper>
auto MakeMappedRange(TIterator begin, TIterator end, TMapper mapper) {
    return MakeIteratorRange(MakeMappedIterator(begin, mapper), MakeMappedIterator(end, mapper));
}

template <class TContainer, class TMapper>
auto MakeMappedRange(TContainer&& container, TMapper&& mapper) {
    if constexpr (NIteratorPrivate::HasRandomAccess<decltype(std::begin(container))>()) {
        return TRandomAccessMappedRange<TContainer, TMapper>(std::forward<TContainer>(container), std::forward<TMapper>(mapper));
    } else {
        return TInputMappedRange<TContainer, TMapper>(std::forward<TContainer>(container), std::forward<TMapper>(mapper));
    }
}
