#pragma once

template <class TTo, class TFrom>
static inline TTo UnionCast(TFrom from) {
    union Y_HIDDEN {
        TFrom from;
        TTo to;
    } a;

    a.from = from;

    return a.to;
}

template <class TTo, class TFrom>
static inline TTo ReinterpretCast(TFrom from) {
    static_assert(sizeof(TFrom) == sizeof(TTo), "sizeof(TFrom) != sizeof(TTo)");

    return ::UnionCast<TTo>(from);
}
