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

