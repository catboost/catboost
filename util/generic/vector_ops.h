#pragma once

#include <util/system/yassert.h>
#include <util/generic/yexception.h>

namespace NVectorOps {
    template <class T, class TVec>
    class TVectorOpsBase {
        inline const TVec& Vec() const noexcept {
            return *static_cast<const TVec*>(this);
        }

    public:
        using TConstIterator = const T*;
        using TConstReference = const T&;

        inline const T* Data() const noexcept {
            return Vec().Data();
        }

        inline size_t Size() const noexcept {
            return Vec().Size();
        }

        inline bool Empty() const noexcept {
            return !Size();
        }

        inline TConstIterator Begin() const noexcept {
            return Data();
        }

        inline TConstIterator End() const noexcept {
            return Data() + Size();
        }

        inline TConstReference Front() const noexcept {
            return (*this)[0];
        }

        inline TConstReference Back() const noexcept {
            Y_ASSERT(!Empty());

            return *(End() - 1);
        }

        inline TConstReference At(size_t n) const {
            if (n >= Size()) {
                ThrowRangeError("array ref range error");
            }

            return (*this)[n];
        }

        inline const T* operator~() const noexcept {
            return Data();
        }

        inline size_t operator+() const noexcept {
            return Size();
        }

        inline explicit operator bool() const noexcept {
            return !Empty();
        }

        inline const T& operator[](size_t n) const noexcept {
            Y_ASSERT(n < Size());

            return *(Begin() + n);
        }

        //compat, do not use
        using const_iterator = TConstIterator;
        using const_reference = TConstReference;
        using value_type = T;

        inline const_iterator begin() const noexcept {
            return Begin();
        }

        inline const_iterator end() const noexcept {
            return End();
        }

        inline size_t size() const noexcept {
            return Size();
        }

        inline bool empty() const noexcept {
            return Empty();
        }

        inline const_reference front() const noexcept {
            return Front();
        }

        inline const_reference back() const noexcept {
            return Back();
        }

        inline const_reference at(size_t n) const {
            return At(n);
        }
    };

    template <class T, class TVec>
    class TVectorOps: public TVectorOpsBase<T, TVec> {
        using TBase = TVectorOpsBase<T, TVec>;

        inline TVec& Vec() noexcept {
            return *static_cast<TVec*>(this);
        }

    public:
        using TIterator = T*;
        using TReference = T&;

        using TBase::Data;
        using TBase::Begin;
        using TBase::End;
        using TBase::Front;
        using TBase::Back;
        using TBase::At;
        using TBase::operator~;
        using TBase::operator[];

        inline T* Data() noexcept {
            return Vec().Data();
        }

        inline TIterator Begin() noexcept {
            return this->Data();
        }

        inline TIterator End() noexcept {
            return this->Data() + this->Size();
        }

        inline TReference Front() noexcept {
            return (*this)[0];
        }

        inline TReference Back() noexcept {
            Y_ASSERT(!this->Empty());

            return *(this->End() - 1);
        }

        inline TReference At(size_t n) {
            if (n >= this->Size()) {
                ThrowRangeError("array ref range error");
            }

            return (*this)[n];
        }

        inline T* operator~() noexcept {
            return this->Data();
        }

        inline T& operator[](size_t n) noexcept {
            Y_ASSERT(n < this->Size());

            return *(this->Begin() + n);
        }

        //compat, do not use
        using iterator = TIterator;
        using reference = TReference;

        using TBase::begin;
        using TBase::end;
        using TBase::front;
        using TBase::back;
        using TBase::at;

        inline iterator begin() noexcept {
            return this->Begin();
        }

        inline iterator end() noexcept {
            return this->End();
        }

        inline reference front() noexcept {
            return this->Front();
        }

        inline reference back() noexcept {
            return this->Back();
        }

        inline reference at(size_t n) {
            return this->At(n);
        }
    };

    template <class T, class TVec>
    class TVectorOps<const T, TVec>: public TVectorOpsBase<const T, TVec> {
    };
}
