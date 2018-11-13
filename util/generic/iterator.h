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

/**
 * Range adaptor that turns a derived class with a Java-style iteration
 * interface into an STL range.
 *
 * Derived class is expected to define:
 * \code
 * using TRetVal = <pointer>;
 * TRetVal Next();
 * \endcode
 *
 * `TRetVal` is expected to be a pointer-like type. `Next()` returning nullptr
 * signals end of range.
 *
 * Since iteration state is stored inside the derived class, the resulting range
 * is an input range (works for single pass algorithms only). Technically speaking,
 * if `TRetVal` is a non-const pointer, it can also work as an output range.
 *
 * Example usage:
 * \code
 * class TSquaresGenerator: public TInputRangeAdaptor<TSquaresGenerator> {
 * public:
 *     using TRetVal = const double*;
 *     TRetVal Next() {
 *         Current_ = State_ * State_;
 *         State_ += 1.0;
 *         // Never return nullptr => we have infinite range!
 *         return &Current_;
 *     }
 *
 * private:
 *     double State_ = 0.0;
 *     double Current_ = 0.0;
 * }
 * \endcode
 */
template <class TSlave>
class TInputRangeAdaptor {
public:
    class TIterator {
    public:
        static constexpr bool IsNoexceptNext = noexcept(std::declval<TSlave>().Next());

        using difference_type = std::ptrdiff_t;
        using TNextType = decltype(std::declval<TSlave>().Next());
        using TValueTraits = NStlIterator::TTraits<TNextType>; // TODO: DROP!
        using pointer = typename TValueTraits::TPtr;
        using reference = typename TValueTraits::TRef;
        using value_type = std::remove_cv_t<std::remove_reference_t<reference>>;
        using iterator_category = std::input_iterator_tag;

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
        TNextType Cur_;
    };

public:
    using const_iterator = TIterator;
    using iterator = const_iterator;

    inline iterator begin() const noexcept(TIterator::IsNoexceptNext) {
        return TIterator(const_cast<TSlave*>(static_cast<const TSlave*>(this)));
    }

    inline iterator end() const noexcept {
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
