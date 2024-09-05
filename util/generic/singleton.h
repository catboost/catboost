#pragma once

#include <util/system/atexit.h>
#include <util/system/compiler.h>

#include <atomic>
#include <new>
#include <utility>

template <class T>
struct TSingletonTraits {
    static constexpr size_t Priority = 65536;
};

namespace NPrivate {
    void FillWithTrash(void* ptr, size_t len);

    void LockRecursive(std::atomic<size_t>& lock) noexcept;
    void UnlockRecursive(std::atomic<size_t>& lock) noexcept;

    template <class T>
    void Destroyer(void* ptr) {
        ((T*)ptr)->~T();
        FillWithTrash(ptr, sizeof(T));
    }

    template <class T, size_t P, class... TArgs>
    Y_NO_INLINE T* SingletonBase(std::atomic<T*>& ptr, TArgs&&... args) {
        alignas(T) static char buf[sizeof(T)];
        static std::atomic<size_t> lock;

        LockRecursive(lock);

        auto ret = ptr.load();

        try {
            if (!ret) {
                ret = ::new (buf) T(std::forward<TArgs>(args)...);

                try {
                    AtExit(Destroyer<T>, ret, P);
                } catch (...) {
                    Destroyer<T>(ret);

                    throw;
                }

                ptr.store(ret);
            }
        } catch (...) {
            UnlockRecursive(lock);

            throw;
        }

        UnlockRecursive(lock);

        return ret;
    }

    template <class T, size_t P, class... TArgs>
    T* SingletonInt(TArgs&&... args) {
        static_assert(sizeof(T) < 32000, "use HugeSingleton instead");

        static std::atomic<T*> ptr;
        auto ret = ptr.load();

        if (Y_UNLIKELY(!ret)) {
            ret = SingletonBase<T, P>(ptr, std::forward<TArgs>(args)...);
        }

        return ret;
    }

    template <class T>
    class TDefault {
    public:
        template <class... TArgs>
        inline TDefault(TArgs&&... args)
            : T_(std::forward<TArgs>(args)...)
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
        template <class... TArgs>
        inline THeapStore(TArgs&&... args)
            : D(new T(std::forward<TArgs>(args)...))
        {
        }

        inline ~THeapStore() {
            delete D;
        }

        T* D;
    };
} // namespace NPrivate

#define Y_DECLARE_SINGLETON_FRIEND()                \
    template <class T, size_t P, class... TArgs>    \
    friend T* ::NPrivate::SingletonInt(TArgs&&...); \
    template <class T, size_t P, class... TArgs>    \
    friend T* ::NPrivate::SingletonBase(std::atomic<T*>&, TArgs&&...);

template <class T, class... TArgs>
Y_RETURNS_NONNULL T* Singleton(TArgs&&... args) {
    return ::NPrivate::SingletonInt<T, TSingletonTraits<T>::Priority>(std::forward<TArgs>(args)...);
}

template <class T, class... TArgs>
Y_RETURNS_NONNULL T* HugeSingleton(TArgs&&... args) {
    return Singleton<::NPrivate::THeapStore<T>>(std::forward<TArgs>(args)...)->D;
}

template <class T, size_t P, class... TArgs>
Y_RETURNS_NONNULL T* SingletonWithPriority(TArgs&&... args) {
    return ::NPrivate::SingletonInt<T, P>(std::forward<TArgs>(args)...);
}

template <class T, size_t P, class... TArgs>
Y_RETURNS_NONNULL T* HugeSingletonWithPriority(TArgs&&... args) {
    return SingletonWithPriority<::NPrivate::THeapStore<T>, P>(std::forward<TArgs>(args)...)->D;
}

template <class T>
const T& Default() {
    return *(::NPrivate::SingletonInt<typename ::NPrivate::TDefault<T>, TSingletonTraits<T>::Priority>()->Get());
}
