#pragma once

#include <util/generic/typetraits.h>

#include <new>

namespace NRangeOps {
    template <class T, bool isTrivial>
    struct TRangeOpsBase {
        static inline void DestroyRange(T* b, T* e) noexcept {
            while (e > b) {
                (--e)->~T();
            }
        }

        static inline void InitializeRange(T* b, T* e) {
            T* c = b;

            try {
                for (; c < e; ++c) {
                    new (c) T();
                }
            } catch (...) {
                DestroyRange(b, c);

                throw;
            }
        }
    };

    template <class T>
    struct TRangeOpsBase<T, true> {
        static inline void DestroyRange(T*, T*) noexcept {
        }

        static inline void InitializeRange(T*, T*) noexcept {
        }
    };

    template <class T>
    using TRangeOps = TRangeOpsBase<T, TTypeTraits<T>::IsPod>;

    template <class T>
    static inline void DestroyRange(T* b, T* e) noexcept {
        TRangeOps<T>::DestroyRange(b, e);
    }

    template <class T>
    static inline void InitializeRange(T* b, T* e) {
        TRangeOps<T>::InitializeRange(b, e);
    }
}
