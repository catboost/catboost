#pragma once

#include <util/system/atexit.h>
#include <util/system/atomic.h>

#include <new>

template <class T>
struct TSingletonTraits {
    static constexpr size_t Priority = 65536;
};

namespace NPrivate {
    void FillWithTrash(void* ptr, size_t len);

    void LockRecursive(TAtomic& lock) noexcept;
    void UnlockRecursive(TAtomic& lock) noexcept;

    template <class T>
    void Destroyer(void* ptr) {
        ((T*)ptr)->~T();
        FillWithTrash(ptr, sizeof(T));
    }

    template <class T, size_t P>
    Y_NO_INLINE T* SingletonBase(T*& ptr) {
        alignas(T) static char buf[sizeof(T)];
        static TAtomic lock;

        LockRecursive(lock);

        auto ret = AtomicGet(ptr);

        try {
            if (!ret) {
                ret = ::new (buf) T();

                try {
                    AtExit(Destroyer<T>, ret, P);
                } catch (...) {
                    Destroyer<T>(ret);

                    throw;
                }

                AtomicSet(ptr, ret);
            }
        } catch (...) {
            UnlockRecursive(lock);

            throw;
        }

        UnlockRecursive(lock);

        return ret;
    }

    template <class T, size_t P>
    T* SingletonInt() {
        static_assert(sizeof(T) < 32000, "use HugeSingleton instead");

        static T* ptr;
        auto ret = AtomicGet(ptr);

        if (Y_UNLIKELY(!ret)) {
            ret = SingletonBase<T, P>(ptr);
        }

        return ret;
    }

    template <class T>
    class TDefault {
    public:
        inline TDefault()
            : T_()
        {
        }

        inline const T* Get() const noexcept {
            return &T_;
        }

    private:
        T T_;
    };

    template <class T>
    struct THeapStore {
        inline THeapStore()
            : D(new T())
        {
        }

        inline ~THeapStore() {
            delete D;
        }

        T* D;
    };
}

#define Y_DECLARE_SINGLETON_FRIEND()            \
    template <class T, size_t P>                \
    friend T* ::NPrivate::SingletonInt();       \
    template <class T, size_t P>                \
    friend T* ::NPrivate::SingletonBase(T*&);

template <class T>
T* Singleton() {
    return ::NPrivate::SingletonInt<T, TSingletonTraits<T>::Priority>();
}

template <class T>
T* HugeSingleton() {
    return Singleton< ::NPrivate::THeapStore<T>>()->D;
}

template <class T, size_t P>
T* SingletonWithPriority() {
    return ::NPrivate::SingletonInt<T, P>();
}

template <class T, size_t P>
T* HugeSingletonWithPriority() {
    return SingletonWithPriority< ::NPrivate::THeapStore<T>, P>()->D;
}

template <class T>
const T& Default() {
    return *(::NPrivate::SingletonInt<typename ::NPrivate::TDefault<T>, TSingletonTraits<T>::Priority>()->Get());
}
