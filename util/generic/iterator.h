#pragma once

#include <iterator>
#include <utility>

namespace NStlIterator {
    template <class T>
    struct TRefFromPtr;

    template <class T>
    struct TRefFromPtr<T*> {
        using TResult = T&;
    };

    template <class T>
    struct TTraits {
        using TPtr = typename T::TPtr;
        using TRef = typename TRefFromPtr<TPtr>::TResult;

        static inline TPtr Ptr(const T& p) noexcept {
            return T::Ptr(p);
        }
    };

    template <class T>
    struct TTraits<T*> {
        using TPtr = T*;
        using TRef = T&;

        static inline TPtr Ptr(TPtr p) noexcept {
            return p;
        }
    };
}

template <class TSlave>
class TStlIterator {
public:
    class TIterator {
    public:
        static constexpr bool IsNoexceptNext = noexcept(std::declval<TSlave>().Next());

        using difference_type = std::ptrdiff_t;
        using value_type = typename TSlave::TRetVal;
        using TValueTraits = NStlIterator::TTraits<value_type>; // TODO: DROP!
        using pointer = typename TValueTraits::TPtr;
        using reference = typename TValueTraits::TRef;
        using iterator_category = std::forward_iterator_tag;

        inline TIterator() noexcept
            : Slave_(nullptr)
            , Cur_()
        {
        }

        inline TIterator(TSlave* slave) noexcept(IsNoexceptNext)
            : Slave_(slave)
            , Cur_(Slave_->Next())
        {
        }

        const value_type& Value() const noexcept {
            return Cur_;
        }

        inline bool operator==(const TIterator& it) const noexcept {
            return Cur_ == it.Cur_;
        }

        inline bool operator!=(const TIterator& it) const noexcept {
            return !(*this == it);
        }

        inline pointer operator->() const noexcept {
            return TValueTraits::Ptr(Cur_);
        }

        inline reference operator*() const noexcept {
            return *TValueTraits::Ptr(Cur_);
        }

        inline TIterator& operator++() noexcept(IsNoexceptNext) {
            Cur_ = Slave_->Next();

            return *this;
        }

    private:
        TSlave* Slave_;
        value_type Cur_;
    };

public:
    inline TIterator begin() const noexcept(TIterator::IsNoexceptNext) {
        return TIterator(const_cast<TSlave*>(static_cast<const TSlave*>(this)));
    }

    inline TIterator end() const noexcept {
        return TIterator();
    }
};

/**
 * Transform given reverse iterator into forward iterator pointing to the same element.
 *
 * @see http://stackoverflow.com/a/1830240
 */
template <class TIterator>
auto ToForwardIterator(TIterator iter) {
    return std::next(iter).base();
}
