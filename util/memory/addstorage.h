#pragma once

#include <util/system/align.h>
#include <util/system/defaults.h>

#include <new>

template <class T>
class TAdditionalStorage {
    union TInfo {
    public:
        inline TInfo(size_t length) noexcept {
            Data_.Length = length;
        }

        inline ~TInfo() = default;

        inline size_t Length() const noexcept {
            return Data_.Length;
        }

    private:
        TAligner Aligner_;

        struct {
            size_t Length;
        } Data_;
    };

public:
    inline TAdditionalStorage() noexcept = default;

    inline ~TAdditionalStorage() = default;

    inline void* operator new(size_t len1, size_t len2) {
        void* data = ::operator new(sizeof(TInfo) + len1 + len2);

        return 1 + new (data) TInfo(len2);
    }

    inline void operator delete(void* ptr) noexcept {
        DoDelete(ptr);
    }

    inline void operator delete(void* ptr, size_t) noexcept {
        DoDelete(ptr);
    }

    inline void operator delete(void* ptr, size_t, size_t) noexcept {
        /*
         * this delete operator can be called automagically by compiler
         */

        DoDelete(ptr);
    }

    inline void* AdditionalData() const noexcept {
        return 1 + (T*)this;
    }

    static inline T* ObjectFromData(void* data) noexcept {
        return ((T*)data) - 1;
    }

    inline size_t AdditionalDataLength() const noexcept {
        return (((TInfo*)((void*)(T*)this)) - 1)->Length();
    }

private:
    static inline void DoDelete(void* ptr) noexcept {
        TInfo* info = (TInfo*)ptr - 1;
        info->~TInfo();
        ::operator delete((void*)info);
    }

private:
    void* operator new(size_t);
};
