#pragma once

#include <util/generic/yexception.h>
#include <util/generic/utility.h>
#include <util/memory/alloc.h>
#include <util/stream/output.h>
#include <util/system/yassert.h>

#include <cstdlib>

// vector that is 8 bytes when empty (TVector is 24 bytes)

template <typename T>
class TCompactVector {
private:
    typedef TCompactVector<T> TThis;

    // XXX: make header independent on T and introduce nullptr
    struct THeader {
        size_t Size;
        size_t Capacity;
    };

    T* Ptr;

    THeader* Header() {
        return ((THeader*)Ptr) - 1;
    }

    const THeader* Header() const {
        return ((THeader*)Ptr) - 1;
    }

public:
    typedef T* TIterator;
    typedef const T* TConstIterator;

    typedef TIterator iterator;
    typedef TConstIterator const_iterator;

    TCompactVector()
        : Ptr(nullptr)
    {
    }

    TCompactVector(const TThis& that)
        : Ptr(nullptr)
    {
        Reserve(that.Size());
        for (TConstIterator i = that.Begin(); i != that.End(); ++i) {
            PushBack(*i);
        }
    }

    ~TCompactVector() {
        for (size_t i = 0; i < Size(); ++i) {
            try {
                (*this)[i].~T();
            } catch (...) {
            }
        }
        if (Ptr)
            free(Header());
    }

    TIterator Begin() {
        return Ptr;
    }

    TIterator End() {
        return Ptr + Size();
    }

    TConstIterator Begin() const {
        return Ptr;
    }

    TConstIterator End() const {
        return Ptr + Size();
    }

    iterator begin() {
        return Begin();
    }

    const_iterator begin() const {
        return Begin();
    }

    iterator end() {
        return End();
    }

    const_iterator end() const {
        return End();
    }

    void Swap(TThis& that) {
        DoSwap(Ptr, that.Ptr);
    }

    void Reserve(size_t newCapacity) {
        if (newCapacity <= Capacity()) {
        } else if (Ptr == nullptr) {
            void* mem = ::malloc(sizeof(THeader) + newCapacity * sizeof(T));
            if (mem == nullptr)
                ythrow yexception() << "out of memory";
            Ptr = (T*)(((THeader*)mem) + 1);
            Header()->Size = 0;
            Header()->Capacity = newCapacity;
        } else {
            TThis copy;
            size_t realNewCapacity = Max(Capacity() * 2, newCapacity);
            copy.Reserve(realNewCapacity);
            for (TConstIterator it = Begin(); it != End(); ++it) {
                copy.PushBack(*it);
            }
            Swap(copy);
        }
    }

    size_t Size() const {
        return Ptr ? Header()->Size : 0;
    }

    size_t size() const {
        return Size();
    }

    bool Empty() const {
        return Size() == 0;
    }

    bool empty() const {
        return Empty();
    }

    size_t Capacity() const {
        return Ptr ? Header()->Capacity : 0;
    }

    void PushBack(const T& elem) {
        Reserve(Size() + 1);
        new (Ptr + Size()) T(elem);
        ++(Header()->Size);
    }

    T& Back() {
        return *(End() - 1);
    }

    const T& Back() const {
        return *(End() - 1);
    }

    T& back() {
        return Back();
    }

    const T& back() const {
        return Back();
    }

    TIterator Insert(TIterator pos, const T& elem) {
        Y_ASSERT(pos >= Begin());
        Y_ASSERT(pos <= End());

        size_t posn = pos - Begin();
        if (pos == End()) {
            PushBack(elem);
        } else {
            Y_ASSERT(Size() > 0);

            Reserve(Size() + 1);

            PushBack(*(End() - 1));

            for (size_t i = Size() - 2; i + 1 > posn; --i) {
                (*this)[i + 1] = (*this)[i];
            }

            (*this)[posn] = elem;
        }
        return Begin() + posn;
    }

    iterator insert(iterator pos, const T& elem) {
        return Insert(pos, elem);
    }

    void Clear() {
        TThis clean;
        Swap(clean);
    }

    void clear() {
        Clear();
    }

    T& operator[](size_t index) {
        Y_ASSERT(index < Size());
        return Ptr[index];
    }

    const T& operator[](size_t index) const {
        Y_ASSERT(index < Size());
        return Ptr[index];
    }
};
