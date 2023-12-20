#pragma once

#include <util/system/tls.h>
#include <util/generic/singleton.h>
#include <util/generic/ptr.h>

namespace NPrivate {
    template <class T, size_t Priority>
    struct TFastThreadSingletonHelper {
        static inline T* GetSlow() {
            return SingletonWithPriority<NTls::TValue<T>, Priority>()->GetPtr();
        }

        static inline T* Get() {
#if defined(Y_HAVE_FAST_POD_TLS)
            Y_POD_STATIC_THREAD(T*) fast(nullptr);

            if (Y_UNLIKELY(!fast)) {
                fast = GetSlow();
            }

            return fast;
#else
            return GetSlow();
#endif
        }
    };
}

template <class T, size_t Priority>
Y_RETURNS_NONNULL static inline T* FastTlsSingletonWithPriority() {
    return ::NPrivate::TFastThreadSingletonHelper<T, Priority>::Get();
}

// NB: the singleton is the same for all modules that use
// FastTlsSingleton with the same type parameter. If unique singleton
// required, use unique types.
template <class T>
Y_RETURNS_NONNULL static inline T* FastTlsSingleton() {
    return FastTlsSingletonWithPriority<T, TSingletonTraits<T>::Priority>();
}
