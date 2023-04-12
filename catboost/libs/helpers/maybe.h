#pragma once

#include "exception.h"

#include <util/generic/maybe.h>


namespace NCB {
    struct TPolicyUnavailableData {
        static void OnEmpty(const std::type_info&) {
            CB_ENSURE_INTERNAL(false, "Attempt to access unavailable data");
        }
    };

    template <class T>
    using TMaybeData = TMaybe<T, TPolicyUnavailableData>;

    template <class T, class TPtr>
    TMaybeData<T *> MakeMaybeData(const TPtr &ptr) {
        return ptr ? TMaybeData<T *>(ptr.Get()) : Nothing();
    }

    template <class T>
    TMaybeData<T *> MakeMaybeData(T *ptr) {
        return ptr ? TMaybeData<T *>(ptr) : Nothing();
    }


    template <class T, class Policy, class TDataEqualFunction>
    bool Equal(
        const TMaybe<T, Policy>& lhs,
        const TMaybe<T, Policy>& rhs,
        TDataEqualFunction&& dataEqualFunction
    ) {
        if (lhs.Defined()) {
            if (rhs.Defined()) {
                return dataEqualFunction(*lhs, *rhs);
            }
            return false;
        }
        return !rhs.Defined();
    }
}
