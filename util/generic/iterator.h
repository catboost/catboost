#pragma once

#include <iterator>
#include <utility>

namespace NStlIterator {
    template <class T>
    class TProxy {
    public:
        TProxy() = default;
        TProxy(T&& value)
            : Value_(std::move(value))
        {
        }

        const T* operator->() const noexcept {
            return &Value_;
        }

        const T& operator*() const noexcept {
            return Value_;
        }

        bool operator==(const TProxy& rhs) const {
            return Value_ == rhs.Value_;
        }

    private:
        T Value_;
    };
} // namespace NStlIterator

/**
 * Range adaptor that turns a derived class with a Java-style iteration
 * interface into an STL range.
 *
 * Derived class is expected to define:
 * \code
 * TSomething* Next();
 * \endcode
 *
 * `Next()` returning `nullptr` signals end of range. Note that you can also use
 * pointer-like types instead of actual pointers (e.g. `TAtomicSharedPtr`).
 *
 * Since iteration state is stored inside the derived class, the resulting range
 * is an input range (works for single pass algorithms only). Technically speaking,
 * if you're returning a non-const pointer from `Next`, it can also work as an output range.
 *
 * Example usage:
 * \code
 * class TSquaresGenerator: public TInputRangeAdaptor<TSquaresGenerator> {
 * public:
 *     const double* Next() {
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
public: // TODO: private
    class TIterator {
    public:
        static constexpr bool IsNoexceptNext = noexcept(std::declval<TSlave>().Next());

        using difference_type = std::ptrdiff_t;
        using pointer = decltype(std::declval<TSlave>().Next());
        using reference = decltype(*std::declval<TSlave>().Next());
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
            return Cur_;
        }

        inline reference operator*() const noexcept {
            return *Cur_;
        }

        inline TIterator& operator++() noexcept(IsNoexceptNext) {
            Cur_ = Slave_->Next();

            return *this;
        }

    private:
        TSlave* Slave_;
        pointer Cur_;
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
