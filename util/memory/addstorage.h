#pragma once

#include <util/system/align.h>
#include <util/system/defaults.h>

#include <new>

namespace NPrivate {
    class TAdditionalStorageInfo {
    public:
        constexpr TAdditionalStorageInfo(size_t length) noexcept
            : Length_(length)
        {
        }

        constexpr size_t Length() const noexcept {
            return Length_;
        }

    private:
        size_t Length_;
    };
} // namespace NPrivate

template <class T>
class alignas(::NPrivate::TAdditionalStorageInfo) TAdditionalStorage {
    using TInfo = ::NPrivate::TAdditionalStorageInfo;

public:
    inline TAdditionalStorage() noexcept = default;

    inline ~TAdditionalStorage() = default;

    inline void* operator new(size_t len1, size_t len2) {
        static_assert(alignof(T) >= alignof(TInfo));
        Y_ASSERT(len1 == sizeof(T));
        void* data = ::operator new(CombinedSizeOfInstanceWithTInfo() + len2);
        void* info = InfoPtr(static_cast<T*>(data));
        Y_UNUSED(new (info) TInfo(len2));

        return data;
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
        return (char*)(static_cast<const T*>(this)) + CombinedSizeOfInstanceWithTInfo();
    }

    static inline T* ObjectFromData(void* data) noexcept {
        return reinterpret_cast<T*>(static_cast<char*>(data) - CombinedSizeOfInstanceWithTInfo());
    }

    inline size_t AdditionalDataLength() const noexcept {
        return InfoPtr(static_cast<const T*>(this))->Length();
    }

private:
    static inline void DoDelete(void* ptr) noexcept {
        TInfo* info = InfoPtr(static_cast<T*>(ptr));
        info->~TInfo();
        ::operator delete(ptr);
    }

    static constexpr size_t CombinedSizeOfInstanceWithTInfo() noexcept {
        return AlignUp(sizeof(T), alignof(TInfo)) + sizeof(TInfo);
    }

    static constexpr TInfo* InfoPtr(T* instance) noexcept {
        return const_cast<TInfo*>(InfoPtr(static_cast<const T*>(instance)));
    }

    static constexpr const TInfo* InfoPtr(const T* instance) noexcept {
        return reinterpret_cast<const TInfo*>(reinterpret_cast<const char*>(instance) + CombinedSizeOfInstanceWithTInfo() - sizeof(TInfo));
    }

private:
    void* operator new(size_t) = delete;
};
