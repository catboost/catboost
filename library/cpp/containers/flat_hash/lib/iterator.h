#pragma once

#include "concepts/container.h"

#include <util/system/yassert.h>

#include <iterator>

namespace NFlatHash {

template <class Container, class T>
class TIterator {
private:
    static_assert(NConcepts::ContainerV<std::decay_t<Container>>);

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = typename std::add_pointer<T>::type;
    using reference = typename std::add_lvalue_reference<T>::type;

private:
    using size_type = typename Container::size_type;

public:
    TIterator(Container* cont)
        : Cont_(cont)
        , Idx_(0)
    {
        if (!cont->IsTaken(Idx_)) {
            Next();
        }
    }

    TIterator(Container* cont, size_type idx)
        : Cont_(cont)
        , Idx_(idx) {}

    template <class C, class U, class = std::enable_if_t<std::is_convertible<C*, Container*>::value &&
                                                         std::is_convertible<U, T>::value>>
    TIterator(const TIterator<C, U>& rhs)
        : Cont_(rhs.Cont_)
        , Idx_(rhs.Idx_) {}

    TIterator(const TIterator&) = default;

    TIterator& operator=(const TIterator&) = default;

    TIterator& operator++() {
        Next();
        return *this;
    }
    TIterator operator++(int) {
        auto idx = Idx_;
        Next();
        return { Cont_, idx };
    }

    reference operator*() {
        return Cont_->Node(Idx_);
    }

    pointer operator->() {
        return &Cont_->Node(Idx_);
    }

    const pointer operator->() const {
        return &Cont_->Node(Idx_);
    }

    bool operator==(const TIterator& rhs) const noexcept {
        Y_ASSERT(Cont_ == rhs.Cont_);
        return Idx_ == rhs.Idx_;
    }

    bool operator!=(const TIterator& rhs) const noexcept {
        return !operator==(rhs);
    }

private:
    void Next() {
        // Container provider ensures that it's not empty.
        do {
            ++Idx_;
        } while (Idx_ != Cont_->Size() && !Cont_->IsTaken(Idx_));
    }

private:
    template <class C, class U>
    friend class TIterator;

    Container* Cont_ = nullptr;

protected:
    size_type Idx_ = 0;
};

}  // namespace NFlatHash
