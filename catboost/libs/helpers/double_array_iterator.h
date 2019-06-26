#pragma once

#include <iterator>
#include <utility>


// needed to sort 2 arrays in parallel using std::sort/Sort.

namespace std {

    template <class T1, class T2>
    inline void swap(std::pair<T1&, T2&> lhs, std::pair<T1&, T2&> rhs) {
        std::swap(lhs.first, rhs.first);
        std::swap(lhs.second, rhs.second);
    }

}

namespace NCB {
    template <class T1, class T2>
    struct TDoubleArrayIterator
    {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = std::pair<T1, T2>;
        using difference_type = std::ptrdiff_t;
        using pointer = std::pair<T1&, T2&>*;
        using reference = std::pair<T1&, T2&>;

    public:
        T1* FirstArrayIter;
        T2* SecondArrayIter;

    public:
        reference operator*() const {
            return reference{*FirstArrayIter, *SecondArrayIter};
        }

        TDoubleArrayIterator& operator++() {
            ++FirstArrayIter;
            ++SecondArrayIter;
            return *this;
        }

        TDoubleArrayIterator operator++(int) {
            TDoubleArrayIterator result(*this);
            ++(*this);
            return result;
        }

        TDoubleArrayIterator& operator--() {
            --FirstArrayIter;
            --SecondArrayIter;
            return *this;
        }

        TDoubleArrayIterator operator--(int) {
            TDoubleArrayIterator result(*this);
            --(*this);
            return result;
        }

        TDoubleArrayIterator operator+(difference_type i) const {
            return TDoubleArrayIterator{FirstArrayIter + i, SecondArrayIter + i};
        }

        TDoubleArrayIterator& operator+=(difference_type i) {
            FirstArrayIter += i;
            SecondArrayIter += i;
            return *this;
        }

        difference_type operator-(const TDoubleArrayIterator& rhs) const {
            return FirstArrayIter - rhs.FirstArrayIter;
        }

        TDoubleArrayIterator operator-(difference_type i) const {
            return TDoubleArrayIterator{FirstArrayIter - i, SecondArrayIter - i};
        }

        TDoubleArrayIterator& operator-=(difference_type i) {
            FirstArrayIter -= i;
            SecondArrayIter -= i;
            return *this;
        }

        bool operator==(const TDoubleArrayIterator& rhs) const {
            return FirstArrayIter == rhs.FirstArrayIter;
        }

        bool operator!=(const TDoubleArrayIterator& rhs) const {
            return FirstArrayIter != rhs.FirstArrayIter;
        }

        bool operator<(const TDoubleArrayIterator& rhs) const {
            return FirstArrayIter < rhs.FirstArrayIter;
        }

        bool operator<=(const TDoubleArrayIterator& rhs) const {
            return FirstArrayIter <= rhs.FirstArrayIter;
        }

        bool operator>(const TDoubleArrayIterator& rhs) const {
            return FirstArrayIter > rhs.FirstArrayIter;
        }

        bool operator>=(const TDoubleArrayIterator& rhs) const {
            return FirstArrayIter >= rhs.FirstArrayIter;
        }

    };

}
