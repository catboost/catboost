#pragma once

#include "array_ref.h"

/**
 * These are legacy typedefs and methods. They should be removed.
 *
 * DEPRECATED. DO NOT USE.
 */
template <typename T>
using TRegion = TArrayRef<T>;
using TDataRegion = TArrayRef<const char>;
using TMemRegion = TArrayRef<char>;

/**
 * Legacy converters to region containers which follow yandex-style memory access
 *
 * DEPRECATED. DO NOT USE.
 */
template <typename TCont>
TArrayRef<const typename TCont::value_type> ToRegion(const TCont& cont) {
    return MakeArrayRef(cont);
}

template <typename TCont>
TArrayRef<typename TCont::value_type> ToRegion(TCont& cont) {
    return MakeArrayRef(cont);
}


