#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

#include <iterator>

namespace NPagedVector {
    template <class T, ui32 PageSize = 1u << 20u, class A = std::allocator<T>>
    class TPagedVector;

    namespace NPrivate {
        template <class T, class TT, ui32 PageSize, class A>
        struct TPagedVectorIterator {
        private:
            friend class TPagedVector<TT, PageSize, A>;
            typedef TPagedVector<TT, PageSize, A> TVec;
            typedef TPagedVectorIterator<T, TT, PageSize, A> TSelf;
            size_t Offset;
            TVec* Vector;

            template <class T1, class TT1, ui32 PageSize1, class A1>
            friend struct TPagedVectorIterator;

        public:
            TPagedVectorIterator()
                : Offset()
                , Vector()
            {
            }

            TPagedVectorIterator(TVec* vector, size_t offset)
                : Offset(offset)
                , Vector(vector)
            {
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            TPagedVectorIterator(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it)
                : Offset(it.Offset)
                , Vector(it.Vector)
            {
            }

            T& operator*() const {
                return (*Vector)[Offset];
            }

            T* operator->() const {
                return &(**this);
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            bool operator==(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it) const {
                return Offset == it.Offset;
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            bool operator!=(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it) const {
                return !(*this == it);
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            bool operator<(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it) const {
                return Offset < it.Offset;
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            bool operator<=(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it) const {
                return Offset <= it.Offset;
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            bool operator>(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it) const {
                return !(*this <= it);
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            bool operator>=(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it) const {
                return !(*this < it);
            }

            template <class T1, class TT1, ui32 PageSize1, class A1>
            ptrdiff_t operator-(const TPagedVectorIterator<T1, TT1, PageSize1, A1>& it) const {
                return Offset - it.Offset;
            }

            TSelf& operator+=(ptrdiff_t off) {
                Offset += off;
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
                return Offset;
            }
        };
    }
}

namespace std {
    template <class T, class TT, ui32 PageSize, class A>
    struct iterator_traits<NPagedVector::NPrivate::TPagedVectorIterator<T, TT, PageSize, A>> {
        typedef ptrdiff_t difference_type;
        typedef T value_type;
        typedef T* pointer;
        typedef T& reference;
        typedef random_access_iterator_tag iterator_category;
    };

}

namespace NPagedVector {
    //2-level radix tree
    template <class T, ui32 PageSize, class A>
    class TPagedVector: private TVector<TSimpleSharedPtr<TVector<T, A>>, A> {
        static_assert(PageSize, "expect PageSize");

        typedef TVector<T, A> TPage;
        typedef TVector<TSimpleSharedPtr<TPage>, A> TPages;
        typedef TPagedVector<T, PageSize, A> TSelf;

    public:
        typedef NPrivate::TPagedVectorIterator<T, T, PageSize, A> iterator;
        typedef NPrivate::TPagedVectorIterator<const T, T, PageSize, A> const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
        typedef T value_type;
        typedef value_type& reference;
        typedef const value_type& const_reference;

        TPagedVector() = default;

        template <typename TIter>
        TPagedVector(TIter b, TIter e) {
            append(b, e);
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
            TPages::swap((TPages&)v);
        }

    private:
        static size_t PageNumber(size_t idx) {
            return idx / PageSize;
        }

        static size_t InPageIndex(size_t idx) {
            return idx % PageSize;
        }

        static size_t Index(size_t pnum, size_t poff) {
            return pnum * PageSize + poff;
        }

        TPage& PageAt(size_t pnum) const {
            return *TPages::at(pnum);
        }

        TPage& CurrentPage() const {
            return *TPages::back();
        }

        size_t CurrentPageSize() const {
            return TPages::empty() ? 0 : CurrentPage().size();
        }

        size_t NPages() const {
            return TPages::size();
        }

        void AllocateNewPage() {
            TPages::push_back(new TPage());
            CurrentPage().reserve(PageSize);
        }

        void MakeNewPage() {
            AllocateNewPage();
            CurrentPage().resize(PageSize);
        }

        void PrepareAppend() {
            if (TPages::empty() || CurrentPage().size() + 1 > PageSize)
                AllocateNewPage();
        }

    public:
        size_t size() const {
            return empty() ? 0 : (NPages() - 1) * PageSize + CurrentPage().size();
        }

        bool empty() const {
            return TPages::empty() || (1 == NPages() && CurrentPage().empty());
        }

        explicit operator bool() const noexcept {
            return !empty();
        }

        template<typename... Args>
        reference emplace_back(Args&&... args) {
            PrepareAppend();
            return CurrentPage().emplace_back(std::forward<Args>(args)...);
        }

        void push_back(const_reference t) {
            PrepareAppend();
            CurrentPage().push_back(t);
        }

        void pop_back() {
            if (CurrentPage().empty())
                TPages::pop_back();
            CurrentPage().pop_back();
        }

        template <typename TIter>
        void append(TIter b, TIter e) {
            size_t sz = e - b;
            size_t sz1 = Min<size_t>(sz, PageSize - CurrentPageSize());
            size_t sz2 = (sz - sz1) / PageSize;
            size_t sz3 = (sz - sz1) % PageSize;

            if (sz1) {
                PrepareAppend();
                TPage& p = CurrentPage();
                p.insert(p.end(), b, b + sz1);
            }

            for (size_t i = 0; i < sz2; ++i) {
                AllocateNewPage();
                TPage& p = CurrentPage();
                p.insert(p.end(), b + sz1 + i * PageSize, b + sz1 + (i + 1) * PageSize);
            }

            if (sz3) {
                AllocateNewPage();
                TPage& p = CurrentPage();
                p.insert(p.end(), b + sz1 + sz2 * PageSize, e);
            }
        }

        iterator erase(iterator it) {
            size_t pnum = PageNumber(it.Offset);
            size_t pidx = InPageIndex(it.Offset);

            if (CurrentPage().empty())
                TPages::pop_back();

            for (size_t p = NPages() - 1; p > pnum; --p) {
                PageAt(p - 1).push_back(PageAt(p).front());
                PageAt(p).erase(PageAt(p).begin());
            }

            PageAt(pnum).erase(PageAt(pnum).begin() + pidx);
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

        iterator insert(iterator it, const value_type& v) {
            size_t pnum = PageNumber(it.Offset);
            size_t pidx = InPageIndex(it.Offset);

            PrepareAppend();

            for (size_t p = NPages() - 1; p > pnum; --p) {
                PageAt(p).insert(PageAt(p).begin(), PageAt(p - 1).back());
                PageAt(p - 1).pop_back();
            }

            PageAt(pnum).insert(PageAt(pnum).begin() + pidx, v);
            return it;
        }

        template <typename TIter>
        void insert(iterator it, TIter b, TIter e) {
            // todo : suboptimal!
            for (; b != e; ++b, ++it)
                it = insert(it, *b);
        }

        reference front() {
            return TPages::front()->front();
        }

        const_reference front() const {
            return TPages::front()->front();
        }

        reference back() {
            return CurrentPage().back();
        }

        const_reference back() const {
            return CurrentPage().back();
        }

        void clear() {
            TPages::clear();
        }

        void resize(size_t sz) {
            if (sz == size())
                return;

            const size_t npages = NPages();
            const size_t newwholepages = sz / PageSize;
            const size_t pagepart = sz % PageSize;
            const size_t newpages = newwholepages + bool(pagepart);

            if (npages && newwholepages >= npages)
                CurrentPage().resize(PageSize);

            if (newpages < npages)
                TPages::resize(newpages);
            else
                for (size_t i = npages; i < newpages; ++i)
                    MakeNewPage();

            if (pagepart)
                CurrentPage().resize(pagepart);

            Y_ABORT_UNLESS(sz == size(), "%" PRIu64 " %" PRIu64, (ui64)sz, (ui64)size());
        }

        reference at(size_t idx) {
            return TPages::at(PageNumber(idx))->at(InPageIndex(idx));
        }

        const_reference at(size_t idx) const {
            return TPages::at(PageNumber(idx))->at(InPageIndex(idx));
        }

        reference operator[](size_t idx) {
            return TPages::operator[](PageNumber(idx))->operator[](InPageIndex(idx));
        }

        const_reference operator[](size_t idx) const {
            return TPages::operator[](PageNumber(idx))->operator[](InPageIndex(idx));
        }

        friend bool operator==(const TSelf& a, const TSelf& b) {
            return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
        }

        friend bool operator<(const TSelf& a, const TSelf& b) {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        }
    };

    namespace NPrivate {
        typedef std::is_same<std::random_access_iterator_tag, std::iterator_traits<
                                                                  TPagedVector<ui32>::iterator>::iterator_category>
            TIteratorCheck;
        static_assert(TIteratorCheck::value, "expect TIteratorCheck::Result");
    }

}
