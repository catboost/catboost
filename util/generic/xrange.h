#pragma once

#include "typetraits.h"
#include "utility.h"
#include <util/system/yassert.h>
#include <iterator>

/** @file
 * Some similar for python xrange(): https://docs.python.org/2/library/functions.html#xrange
 * Discussion: https://clubs.at.yandex-team.ru/arcadia/6124
 *
 * Example usage:
 *  for (auto i: xrange(MyVector.size())) { // instead for (size_t i = 0; i < MyVector.size(); ++i)
 *      DoSomething(i, MyVector[i]);
 *  }
 *
 * TVector<int> arithmeticSeq = xrange(10); // instead: TVector<int> arithmeticSeq; for(size_t i = 0; i < 10; ++i) { arithmeticSeq.push_back(i); }
 *
 */

namespace NPrivate {
    template <typename T>
    class TSimpleXRange {
        using TDiff = decltype(T() - T());

    public:
        constexpr TSimpleXRange(T start, T finish) noexcept
            : Start(start)
            , Finish(Max(start, finish))
        {
        }

        class TIterator {
        public:
            using value_type = T;
            using difference_type = TDiff;
            using pointer = const T*;
            using reference = const T&;
            using iterator_category = std::random_access_iterator_tag;

            constexpr TIterator(T value) noexcept
                : Value(value)
            {
            }

            constexpr T operator*() const noexcept {
                return Value;
            }

            constexpr bool operator!=(const TIterator& other) const noexcept {
                return Value != other.Value;
            }

            constexpr bool operator==(const TIterator& other) const noexcept {
                return Value == other.Value;
            }

            TIterator& operator++() noexcept {
                ++Value;
                return *this;
            }

            TIterator& operator--() noexcept {
                --Value;
                return *this;
            }

            constexpr TDiff operator-(const TIterator& b) const noexcept {
                return Value - b.Value;
            }

            template <typename IntType>
            constexpr TIterator operator+(const IntType& b) const noexcept {
                return TIterator(Value + b);
            }

            template <typename IntType>
            TIterator& operator+=(const IntType& b) noexcept {
                Value += b;
                return *this;
            }

            template <typename IntType>
            constexpr TIterator operator-(const IntType& b) const noexcept {
                return TIterator(Value - b);
            }

            template <typename IntType>
            TIterator& operator-=(const IntType& b) noexcept {
                Value -= b;
                return *this;
            }

            constexpr bool operator<(const TIterator& b) const noexcept {
                return Value < b.Value;
            }

        private:
            T Value;
        };

        using value_type = T;
        using iterator = TIterator;
        using const_iterator = TIterator;

        constexpr TIterator begin() const noexcept {
            return TIterator(Start);
        }

        constexpr TIterator end() const noexcept {
            return TIterator(Finish);
        }

        constexpr T size() const noexcept {
            return Finish - Start;
        }

        template <class Container>
        operator Container() const {
            return Container(begin(), end());
        }

    private:
        T Start;
        T Finish;
    };

    template <typename T>
    class TSteppedXRange {
        using TDiff = decltype(T() - T());

    public:
        constexpr TSteppedXRange(T start, T finish, TDiff step) noexcept
            : Start_(start)
            , Step_(step)
            , Finish_(CalcRealFinish(Start_, finish, Step_))
        {
            static_assert(std::is_integral<T>::value || std::is_pointer<T>::value, "T should be integral type or pointer");
        }

        class TIterator {
        public:
            using value_type = T;
            using difference_type = TDiff;
            using pointer = const T*;
            using reference = const T&;
            using iterator_category = std::random_access_iterator_tag;

            constexpr TIterator(T value, const TSteppedXRange& parent) noexcept
                : Value_(value)
                , Parent_(&parent)
            {
            }

            constexpr T operator*() const noexcept {
                return Value_;
            }

            constexpr bool operator!=(const TIterator& other) const noexcept {
                return Value_ != other.Value_;
            }

            constexpr bool operator==(const TIterator& other) const noexcept {
                return Value_ == other.Value_;
            }

            TIterator& operator++() noexcept {
                Value_ += Parent_->Step_;
                return *this;
            }

            TIterator& operator--() noexcept {
                Value_ -= Parent_->Step_;
                return *this;
            }

            constexpr TDiff operator-(const TIterator& b) const noexcept {
                return (Value_ - b.Value_) / Parent_->Step_;
            }

            template <typename IntType>
            constexpr TIterator operator+(const IntType& b) const noexcept {
                return TIterator(*this) += b;
            }

            template <typename IntType>
            TIterator& operator+=(const IntType& b) noexcept {
                Value_ += b * Parent_->Step_;
                return *this;
            }

            template <typename IntType>
            constexpr TIterator operator-(const IntType& b) const noexcept {
                return TIterator(*this) -= b;
            }

            template <typename IntType>
            TIterator& operator-=(const IntType& b) noexcept {
                Value_ -= b * Parent_->Step_;
                return *this;
            }

        private:
            T Value_;
            const TSteppedXRange* Parent_;
        };

        using value_type = T;
        using iterator = TIterator;
        using const_iterator = TIterator;

        constexpr TIterator begin() const noexcept {
            return TIterator(Start_, *this);
        }

        constexpr TIterator end() const noexcept {
            return TIterator(Finish_, *this);
        }

        static T CalcRealFinish(T start, T expFinish, TDiff step) {
            Y_ASSERT(step != 0);
            if (step > 0) {
                if (expFinish > start) {
                    return start + step * ((expFinish - 1 - start) / step + 1);
                }
                return start;
            }
            return start - TSteppedXRange<TDiff>::CalcRealFinish(0, start - expFinish, -step);
        }

        constexpr T size() const noexcept {
            return (Finish_ - Start_) / Step_;
        }

        template <class Container>
        operator Container() const {
            return Container(begin(), end());
        }

    private:
        const T Start_;
        const TDiff Step_;
        const T Finish_;
    };

} // namespace NPrivate

/**
 * generate arithmetic progression that starts at start with certain step and stop at finish (not including)
 *
 * @param step must be non-zero
 */
template <typename T>
constexpr ::NPrivate::TSteppedXRange<T> xrange(T start, T finish, decltype(T() - T()) step) noexcept {
    return {start, finish, step};
}

/// generate sequence [start; finish)
template <typename T>
constexpr ::NPrivate::TSimpleXRange<T> xrange(T start, T finish) noexcept {
    return {start, finish};
}

/// generate sequence [0; finish)
template <typename T>
constexpr auto xrange(T finish) noexcept -> decltype(xrange(T(), finish)) {
    return xrange(T(), finish);
}
