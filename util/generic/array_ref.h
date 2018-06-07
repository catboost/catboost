#pragma once

#include "vector_ops.h"

#include <util/generic/fwd.h>
#include <util/generic/utility.h>

#include <algorithm>
#include <initializer_list>

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// array_ref.h -- TArrayRef<T>: like TStringBuf, but for arbitrary types
//
// Class, that points to an array: it stores array start pointer and number of elements.
//
// - Can be used instead of vector<T> or (T* ptr, int size) pair in function arguments.
//   TArrayRef validates 'index' and 'count' arguments and sets appropriate pointers if count == -1;
// - Can be automatically constructed from any container with data() and size() methods -- no
//   caller code modification needed!
//
// EXAMPLE
//     TUtf16String Join(char separator, const TArrayRef<TUtf16String> words)
//
//     // Return all pairs of adjacent strings: ["p[0] p[1]", ..., "p[N-1] p[N]"]
//     //
//     TVector<TUtf16String> Pairs(const TVector<TString>& phrase) {
//         TVector<TUtf16String> result;
//         for (size_t i = 0; i + 1 < phrase.size(); ++i)
//              auto pair = Join(" ", TArrayRef<const TUtf16String>(phrase, i, 1));
//              result.push_back(std::move(pair));
//         }
//         return result;
//     }
//
// CONST-CORRECTNESS
//     TArrayRef<T>::operator[] has same const-semantics as operator[] for C++ pointers: pointed-to
//     objects *can* be modified via pointer, that is *const* (pointer 'T* const' is const
//     *itself*, meaning it cannot be re-pointed to another object), So if you want to pass
//     (const T*, int size) analog, use 'const TArrayRef<const T>'.
//
template <class T>
class TArrayRef: public NVectorOps::TVectorOps<T, TArrayRef<T>> {
public:
    inline TArrayRef() noexcept
        : T_(nullptr)
        , S_(0)
    {
    }

    inline TArrayRef(T* data, size_t len) noexcept
        : T_(data)
        , S_(len)
    {
    }

    inline TArrayRef(T* begin, T* end) noexcept
        : T_(begin)
        , S_(end - begin)
    {
    }

    inline TArrayRef(std::initializer_list<T> list) noexcept
        : T_(list.begin())
        , S_(list.size())
    {
    }

    template <class Container>
    inline TArrayRef(Container&& container, decltype(std::declval<T*&>() = container.data(), nullptr) = nullptr) noexcept
        : T_(container.data())
        , S_(container.size())
    {
    }

    template <size_t N>
    inline TArrayRef(T (&array)[N]) noexcept
        : T_(array)
        , S_(N)
    {
    }

    template <class TT, typename = std::enable_if_t<std::is_same<std::remove_const_t<T>, std::remove_const_t<TT>>::value>>
    bool operator==(const TArrayRef<TT>& other) const noexcept {
        return Size() == other.Size() && std::equal(this->Begin(), this->End(), other.Begin());
    }

    inline ~TArrayRef() = default;

    inline T* Data() const noexcept {
        return T_;
    }

    inline size_t Size() const noexcept {
        return S_;
    }

    inline void Swap(TArrayRef& a) noexcept {
        ::DoSwap(T_, a.T_);
        ::DoSwap(S_, a.S_);
    }

    /* STL compatibility. */

    inline T* data() const noexcept {
        return Data();
    }

    inline size_t size() const noexcept {
        return Size();
    }

    inline void swap(TArrayRef& a) noexcept {
        Swap(a);
    }

    TArrayRef<T> Slice(size_t offset) const {
        Y_ASSERT(offset <= size());
        return TArrayRef<T>(data() + offset, size() - offset);
    }

    TArrayRef<T> Slice(size_t offset, size_t size) const {
        Y_ASSERT(offset + size <= this->size());

        return TArrayRef<T>(data() + offset, data() + offset + size);
    }

private:
    T* T_;
    size_t S_;
};
// Functions (a-la std::make_pair) that allow more compact initialization of TArrayRef:
//
//     void Foo(const TArrayRef<TTypeWithANameWayLongerThanNeeded> things);
//
//     int main() {
//         TTypeWithANameWayLongerThanNeeded* thingsBegin;
//         TTypeWithANameWayLongerThanNeeded* thingsEnd;
//
//         Foo(MakeArrayRef(thingsBegin, thingsEnd));
//         //
//         // instead of
//         //
//         Foo(TArrayRef<const TTypeWithANameWayLongerThanNeeded>(thingsBegin, thingsEnd))
//     }
//
template <class Range>
TArrayRef<const typename Range::value_type> MakeArrayRef(const Range& range) {
    return TArrayRef<const typename Range::value_type>(range);
}

template <class Range>
TArrayRef<typename Range::value_type> MakeArrayRef(Range& range) {
    return TArrayRef<typename Range::value_type>(range);
}

template <class T>
TArrayRef<T> MakeArrayRef(T* data, size_t size) {
    return TArrayRef<T>(data, size);
}

template <class T>
TArrayRef<T> MakeArrayRef(T* begin, T* end) {
    return TArrayRef<T>(begin, end);
}
