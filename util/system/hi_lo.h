#pragma once

#include "unaligned_mem.h"

#include <utility>

#ifndef _little_endian_
    #error "Not implemented"
#endif

namespace NHiLoPrivate {
    template <class TRepr>
    class TConstIntRef {
    public:
        explicit TConstIntRef(const char* ptr)
            : Ptr(ptr)
        {
        }

        TRepr Get() const {
            return ReadUnaligned<TRepr>(Ptr);
        }
        operator TRepr() const {
            return Get();
        }

        const char* GetPtr() const {
            return Ptr;
        }

    protected:
        const char* Ptr;
    };

    template <class TRepr>
    class TIntRef: public TConstIntRef<TRepr> {
    public:
        explicit TIntRef(char* ptr)
            : TConstIntRef<TRepr>(ptr)
        {
        }

        TIntRef& operator=(TRepr value) {
            WriteUnaligned<TRepr>(GetPtr(), value);
            return *this;
        }

        char* GetPtr() const {
            return const_cast<char*>(this->Ptr);
        }
    };

    template <class T>
    struct TReferenceType {
        using TType = T;
    };

    template <class T>
    struct TReferenceType<TConstIntRef<T>> {
        using TType = T;
    };

    template <class T>
    struct TReferenceType<TIntRef<T>> {
        using TType = T;
    };

    template <class TRepr>
    auto MakeIntRef(const char* ptr) {
        return TConstIntRef<TRepr>(ptr);
    }

    template <class TRepr>
    auto MakeIntRef(char* ptr) {
        return TIntRef<TRepr>(ptr);
    }

    template <class T>
    const char* CharPtrOf(const T& value) {
        return reinterpret_cast<const char*>(&value);
    }

    template <class T>
    char* CharPtrOf(T& value) {
        return reinterpret_cast<char*>(&value);
    }

    template <class T>
    const char* CharPtrOf(TConstIntRef<T> value) {
        return value.GetPtr();
    }

    template <class T>
    char* CharPtrOf(TIntRef<T> value) {
        return value.GetPtr();
    }

    template <bool IsLow, class TRepr, class T>
    auto MakeIntRef(T&& value) {
        using TRef = typename TReferenceType<typename std::decay<T>::type>::TType;
        static_assert(
            std::is_scalar<TRef>::value,
            "Hi* and Lo* functions can be applied only to scalar values");
        static_assert(sizeof(TRef) >= sizeof(TRepr), "Requested bit range is not within provided value");
        constexpr size_t offset = IsLow ? 0 : sizeof(TRef) - sizeof(TRepr);

        return MakeIntRef<TRepr>(CharPtrOf(std::forward<T>(value)) + offset);
    }
} // namespace NHiLoPrivate

/**
 * Return manipulator object that allows to get and set lower or higher bits of the value.
 *
 * @param value Must be a scalar value of sufficient size or a manipulator object obtained by
 * calling any of the other Hi/Lo functions.
 *
 * @{
 */
template <class T>
auto Lo32(T&& value) {
    return NHiLoPrivate::MakeIntRef<true, ui32>(std::forward<T>(value));
}

template <class T>
auto Hi32(T&& value) {
    return NHiLoPrivate::MakeIntRef<false, ui32>(std::forward<T>(value));
}

template <class T>
auto Lo16(T&& value) {
    return NHiLoPrivate::MakeIntRef<true, ui16>(std::forward<T>(value));
}

template <class T>
auto Hi16(T&& value) {
    return NHiLoPrivate::MakeIntRef<false, ui16>(std::forward<T>(value));
}

template <class T>
auto Lo8(T&& value) {
    return NHiLoPrivate::MakeIntRef<true, ui8>(std::forward<T>(value));
}

template <class T>
auto Hi8(T&& value) {
    return NHiLoPrivate::MakeIntRef<false, ui8>(std::forward<T>(value));
}

/** @} */
