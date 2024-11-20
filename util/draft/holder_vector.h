#pragma once

#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/noncopyable.h>

template <class T, class D = TDelete>
class THolderVector: public TVector<T*>, public TNonCopyable {
    using TBase = TVector<T*>;

public:
    explicit THolderVector(size_t n = 0)
        : TBase(n)
    {
    }

    ~THolderVector() {
        Clear();
    }

    void Clear() {
        for (typename TBase::iterator it = TBase::begin(); it != TBase::end(); ++it) {
            if (*it) {
                D::Destroy(*it);
            }
        }
        TBase::clear();
    }

    size_t Size() const {
        return TBase::size();
    }

    // TVector takes ownership of T
    void PushBack(T* t) {
        try {
            TBase::push_back(t);
        } catch (...) {
            if (t) {
                D::Destroy(t);
            }
            throw;
        }
    }

    void PushBack(std::unique_ptr<T> t) {
        PushBack(t.release());
    }

    void PushBack(THolder<T> t) {
        PushBack(t.Release());
    }

    void Reset(size_t i, THolder<T> t) {
        T* current = (*this)[i];
        if (current) {
            Y_ASSERT(current != t.Get());
            D::Destroy(current);
        }
        (*this)[i] = t.Release();
    }

    void PopBack() {
        if (size()) {
            D::Destroy(back());
            TBase::pop_back();
        }
    }

    T* Release(size_t i) {
        T* t = (*this)[i];
        (*this)[i] = nullptr;
        return t;
    }

    void Resize(size_t newSize) {
        for (size_t i = newSize; i < size(); ++i) {
            D::Destroy((*this)[i]);
        }
        TBase::resize(newSize);
    }

    void Swap(THolderVector& other) {
        TBase::swap(other);
    }

    using TBase::operator[];
    using TBase::operator bool;
    using TBase::at;
    using TBase::back;
    using TBase::begin;
    using TBase::capacity;
    using TBase::empty;
    using TBase::end;
    using TBase::front;
    using TBase::reserve;
    using TBase::size;

    using typename TBase::const_iterator;
    using typename TBase::const_reverse_iterator;
    using typename TBase::iterator;
    using typename TBase::reverse_iterator;
    using typename TBase::value_type;
};
