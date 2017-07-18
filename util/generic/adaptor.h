#pragma once

#include "typetraits.h"

namespace NPrivate {
    template <typename TCont>
    struct TReverseImpl {
        TCont Cont;

        TReverseImpl(TCont cont)
            : Cont(cont)
        {
        }

        auto begin() const {
            return Cont.rbegin();
        }

        auto end() const {
            return Cont.rend();
        }

        auto begin() {
            return Cont.rbegin();
        }

        auto end() {
            return Cont.rend();
        }
    };
}

/**
 * Provides a reverse view into the provided container.
 *
 * Example usage:
 * @code
 * for(auto&& value: Reversed(container)) {
 *     // use value here.
 * }
 * @endcode
 *
 * @param cont                          Container to provide a view into. Must be an lvalue.
 * @returns                             A reverse view into the provided container.
 */
template <typename TCont>
constexpr ::NPrivate::TReverseImpl<TCont> Reversed(TCont&& cont) {
    static_assert(std::is_lvalue_reference<TCont&&>::value, "cont should be lvalue reference");
    return ::NPrivate::TReverseImpl<TCont>(cont);
}
