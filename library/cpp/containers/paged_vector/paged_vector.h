#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

#include <algorithm>
#include <iterator>

namespace NPagedVector {
    template <class T, ui32 PageSize = 1u << 20u>
    class TPagedVector;

    namespace NPrivate {
        template <class T, class TT, ui32 PageSize>
        struct TPagedVectorIterator {
        private:
            friend class TPagedVector<TT, PageSize>;
            using TVec = TPagedVector<TT, PageSize>;
            using TSelf = TPagedVectorIterator<T, TT, PageSize>;
            size_t Offset_;
            TVec* Vector_;

            template <class T1, class TT1, ui32 PageSize1>
            friend struct TPagedVectorIterator;

        public:
            TPagedVectorIterator()
                : Offset_()
                , Vector_()
            {
            }

            TPagedVectorIterator(TVec* vector, size_t offset)
                : Offset_(offset)
                , Vector_(vector)
            {
            }

            template <class T1, class TT1, ui32 PageSize1>
            TPagedVectorIterator(const TPagedVectorIterator<T1, TT1, PageSize1>& it)
                : Offset_(it.Offset_)
                , Vector_(it.Vector_)
            {
            }

            T& operator*() const {
                return (*Vector_)[Offset_];
            }

            T* operator->() const {
                return &(**this);
            }

            template <class T1, class TT1, ui32 PageSize1>
            bool operator==(const TPagedVectorIterator<T1, TT1, PageSize1>& it) const {
                return Offset_ == it.Offset_;
            }

            template <class T1, class TT1, ui32 PageSize1>
            bool operator!=(const TPagedVectorIterator<T1, TT1, PageSize1>& it) const {
                return !(*this == it);
            }

            template <class T1, class TT1, ui32 PageSize1>
            bool operator<(const TPagedVectorIterator<T1, TT1, PageSize1>& it) const {
                return Offset_ < it.Offset_;
            }

            template <class T1, class TT1, ui32 PageSize1>
            bool operator<=(const TPagedVectorIterator<T1, TT1, PageSize1>& it) const {
                return Offset_ <= it.Offset_;
            }

            template <class T1, class TT1, ui32 PageSize1>
            bool operator>(const TPagedVectorIterator<T1, TT1, PageSize1>& it) const {
                return !(*this <= it);
            }

            template <class T1, class TT1, ui32 PageSize1>
            bool operator>=(const TPagedVectorIterator<T1, TT1, PageSize1>& it) const {
                return !(*this < it);
            }

            template <class T1, class TT1, ui32 PageSize1>
            ptrdiff_t operator-(const TPagedVectorIterator<T1, TT1, PageSize1>& it) const {
                return Offset_ - it.Offset_;
            }

            TSelf& operator+=(ptrdiff_t off) {
                Offset_ += off;
                return *this;
            }

            TSelf& operator-=(ptrdiff_t off) {
                return this->operator+=(-off);
            }

            TSelf& operator++() {
                return this->operator+=(1);
            }

            TSelf& operator--() {
                return this->operator+=(-1);
            }

            TSelf operator++(int) {
                TSelf it = *this;
                this->operator+=(1);
                return it;
            }

            TSelf operator--(int) {
                TSelf it = *this;
                this->operator+=(-1);
                return it;
            }

            TSelf operator+(ptrdiff_t off) const {
                TSelf res = *this;
                res += off;
                return res;
            }

            TSelf operator-(ptrdiff_t off) const {
                return this->operator+(-off);
            }

            size_t GetOffset() const {
                return Offset_;
            }
        };
    } // namespace NPrivate
} // namespace NPagedVector

namespace std {
    template <class T, class TT, ui32 PageSize>
    struct iterator_traits<NPagedVector::NPrivate::TPagedVectorIterator<T, TT, PageSize>> {
        using difference_type = ptrdiff_t;
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using iterator_category = random_access_iterator_tag;
    };

} // namespace std

namespace NPagedVector {
    // 2-level radix tree
    template <class T, ui32 PageSize>
    class TPagedVector {
        static_assert(PageSize, "expect PageSize");

        class alignas(T) TPage {
            alignas(T) std::array<char, PageSize * sizeof(T)> Data_;

        public:
            T* data() {
                return reinterpret_cast<T*>(Data_.data());
            }

            const T* data() const {
                return reinterpret_cast<const T*>(Data_.data());
            }

            T& operator[](size_t idx) {
                return *(data() + idx);
            }

            const T& operator[](size_t idx) const {
                return *(data() + idx);
            }
        };

        using TPages = TVector<THolder<TPage>>;
        using TSelf = TPagedVector<T, PageSize>;

        TPages Pages_;
        size_t CurrentPageSize_ = 0;

    public:
        using iterator = NPrivate::TPagedVectorIterator<T, T, PageSize>;
        using const_iterator = NPrivate::TPagedVectorIterator<const T, T, PageSize>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;
        using value_type = T;
        using reference = value_type&;
        using const_reference = const value_type&;

        TPagedVector() = default;
        TPagedVector(TPagedVector&& other) noexcept
            : Pages_(std::move(other.Pages_))
            , CurrentPageSize_(other.CurrentPageSize_)
        {
            other.CurrentPageSize_ = 0;
        }

        TPagedVector(const TPagedVector& other) {
            Pages_.reserve(other.Pages_.size());
            try {
                for (auto& ptr : other.Pages_) {
                    auto& newPage = *Pages_.emplace_back(MakeHolder<TPage>());
                    CurrentPageSize_ = 0;
                    const size_t copyCount = Pages_.size() == other.Pages_.size()
                                                 ? other.CurrentPageSize_
                                                 : PageSize;

                    std::uninitialized_copy_n(ptr->data(), copyCount, newPage.data());
                    CurrentPageSize_ = copyCount;
                }
            } catch (...) {
                clear();
                throw;
            }
        }

        ~TPagedVector() {
            clear();
        }

        template <typename TIter>
        TPagedVector(TIter b, TIter e) {
            append(b, e);
        }

        TPagedVector& operator=(const TPagedVector& other) {
            if (this != &other) {
                TPagedVector tmp(other);
                swap(tmp);
            }
            return *this;
        }

        TPagedVector& operator=(TPagedVector&& other) noexcept {
            if (this != &other) {
                clear();
                Pages_ = std::move(other.Pages_);
                CurrentPageSize_ = other.CurrentPageSize_;
                other.CurrentPageSize_ = 0;
            }
            return *this;
        }

        iterator begin() {
            return iterator(this, 0);
        }

        const_iterator begin() const {
            return const_iterator((TSelf*)this, 0);
        }

        iterator end() {
            return iterator(this, size());
        }

        const_iterator end() const {
            return const_iterator((TSelf*)this, size());
        }

        reverse_iterator rbegin() {
            return reverse_iterator(end());
        }

        const_reverse_iterator rbegin() const {
            return const_reverse_iterator(end());
        }

        reverse_iterator rend() {
            return reverse_iterator(begin());
        }

        const_reverse_iterator rend() const {
            return const_reverse_iterator(begin());
        }

        void swap(TSelf& v) {
            Pages_.swap(v.Pages_);
            std::swap(CurrentPageSize_, v.CurrentPageSize_);
        }

    private:
        static size_t PageNumber(size_t idx) {
            return idx / PageSize;
        }

        static size_t InPageIndex(size_t idx) {
            return idx % PageSize;
        }

        TPage& PageAt(size_t pnum) const {
            return *Pages_.at(pnum);
        }

        TPage& CurrentPage() const {
            return *Pages_.back();
        }

        size_t NPages() const {
            return Pages_.size();
        }

        void AllocateNewPage() {
            Pages_.emplace_back(MakeHolder<TPage>());
            CurrentPageSize_ = 0;
        }

        void PrepareAppend() {
            if (Pages_.empty() || CurrentPageSize_ >= PageSize) {
                AllocateNewPage();
            }
        }

    public:
        size_t size() const {
            return Pages_.empty() ? 0 : (NPages() - 1) * PageSize + CurrentPageSize_;
        }

        bool empty() const {
            return Pages_.empty() || (1 == NPages() && CurrentPageSize_ == 0);
        }

        explicit operator bool() const noexcept {
            return !empty();
        }

        template <typename... Args>
        reference emplace_back(Args&&... args) {
            PrepareAppend();
            T* ptr = new (CurrentPage().data() + CurrentPageSize_) T(std::forward<Args>(args)...);
            ++CurrentPageSize_;
            return *ptr;
        }

        void push_back(const_reference t) {
            PrepareAppend();
            new (CurrentPage().data() + CurrentPageSize_) T(t);
            ++CurrentPageSize_;
        }

        void pop_back() {
            Y_ASSERT(!empty());
            if (CurrentPageSize_ == 0) {
                Pages_.pop_back();
                CurrentPageSize_ = PageSize;
            }
            --CurrentPageSize_;
            if constexpr (!std::is_trivially_destructible_v<T>) {
                CurrentPage()[CurrentPageSize_].~T();
            }
        }

        template <typename TIter>
        void append(TIter b, TIter e) {
            for (TIter it = b; it != e; ++it) {
                push_back(*it);
            }
        }

        iterator erase(iterator it) {
            if (CurrentPageSize_ == 0) {
                Pages_.pop_back();
                CurrentPageSize_ = Pages_.empty() ? 0 : PageSize;
            }

            size_t pidx = InPageIndex(it.Offset_);
            for (size_t pnum = PageNumber(it.Offset_);; ++pnum) {
                TPage& page = *Pages_[pnum];
                if (pnum + 1 == Pages_.size()) {
                    std::shift_left(page.data() + pidx, page.data() + CurrentPageSize_, 1);
                    --CurrentPageSize_;
                    if constexpr (!std::is_trivially_destructible_v<T>) {
                        page[CurrentPageSize_].~T();
                    }
                    break;
                }

                std::shift_left(page.data() + pidx, page.data() + PageSize, 1);
                TPage& nextPage = *Pages_[pnum + 1];
                page[PageSize - 1] = std::move(nextPage[0]);
                pidx = 0;
            }

            return it;
        }

        iterator erase(iterator b, iterator e) {
            // todo : suboptimal!
            while (b != e) {
                b = erase(b);
                --e;
            }

            return b;
        }

        // iterator insert(iterator it, const value_type& v) {
        //     size_t pnum = PageNumber(it.Offset_);
        //     size_t pidx = InPageIndex(it.Offset_);

        // PrepareAppend();

        // for (size_t p = NPages() - 1; p > pnum; --p) {
        //     PageAt(p).insert(PageAt(p).begin(), PageAt(p - 1).back());
        //     PageAt(p - 1).pop_back();
        // }

        // PageAt(pnum).insert(PageAt(pnum).begin() + pidx, v);
        // return it;
        // }

        // template <typename TIter>
        // void insert(iterator it, TIter b, TIter e) {
        //     // todo : suboptimal!
        //     for (; b != e; ++b, ++it) {
        //         it = insert(it, *b);
        //     }
        // }

        reference front() {
            Y_ASSERT(CurrentPageSize_ > 0 || Pages_.size() > 1);
            return (*Pages_.front())[0];
        }

        const_reference front() const {
            Y_ASSERT(CurrentPageSize_ > 0 || Pages_.size() > 1);
            return (*Pages_.front())[0];
        }

        reference back() {
            if (CurrentPageSize_ > 0) {
                return CurrentPage()[CurrentPageSize_ - 1];
            } else {
                Y_ASSERT(Pages_.size() >= 2);
                return (**(Pages_.end() - 2))[PageSize - 1];
            }
        }

        const_reference back() const {
            if (CurrentPageSize_ > 0) {
                return CurrentPage()[CurrentPageSize_ - 1];
            } else {
                Y_ASSERT(Pages_.size() >= 2);
                return (**(Pages_.end() - 2))[PageSize - 1];
            }
        }

        void clear() {
            if constexpr (std::is_trivially_destructible_v<T>) {
                Pages_.clear();
                CurrentPageSize_ = 0;
            } else {
                while (!Pages_.empty()) {
                    TPage& page = CurrentPage();
                    while (CurrentPageSize_ > 0) {
                        --CurrentPageSize_;
                        page[CurrentPageSize_].~T();
                    }
                    Pages_.pop_back();
                    CurrentPageSize_ = Pages_.empty() ? 0 : PageSize;
                }
            }
        }

        void resize(size_t sz) {
            size_t curSize = size();
            if (sz == curSize) {
                return;
            }

            if (sz < curSize) {
                while (sz < curSize) {
                    pop_back();
                    --curSize;
                }
            } else {
                while (sz > curSize) {
                    emplace_back();
                    ++curSize;
                }
            }
        }

        reference at(size_t idx) {
            if (idx >= size()) {
                throw std::out_of_range("TPagedVector::at() - index out of range");
            }
            const size_t pnum = PageNumber(idx);
            const size_t inPageIdx = InPageIndex(idx);

            return (*Pages_[pnum])[inPageIdx];
        }

        const_reference at(size_t idx) const {
            if (idx >= size()) {
                throw std::out_of_range("TPagedVector::at() - index out of range");
            }
            const size_t pnum = PageNumber(idx);
            const size_t inPageIdx = InPageIndex(idx);

            return (*Pages_[pnum])[inPageIdx];
        }

        reference operator[](size_t idx) {
            return Pages_.operator[](PageNumber(idx))->operator[](InPageIndex(idx));
        }

        const_reference operator[](size_t idx) const {
            return Pages_.operator[](PageNumber(idx))->operator[](InPageIndex(idx));
        }

        friend bool operator==(const TSelf& a, const TSelf& b) {
            return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
        }

        friend bool operator<(const TSelf& a, const TSelf& b) {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        }
    };

    namespace NPrivate {
        using TIteratorCheck = std::is_same<std::random_access_iterator_tag, std::iterator_traits<
                                                                                 TPagedVector<ui32>::iterator>::iterator_category>;
        static_assert(TIteratorCheck::value, "expect TIteratorCheck::Result");
    } // namespace NPrivate

} // namespace NPagedVector
