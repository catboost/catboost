#pragma once

#include <util/system/compiler.h>

// Workaround for fiber (un)friendly TLS.
// Volatile qualifier prevents caching access to thread local variables.
#define YT_THREAD_LOCAL(...) thread_local __VA_ARGS__ volatile

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE T& GetTlsRef(volatile T& arg);

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT

#define TLS_INL_H_
#include "tls-inl.h"
#undef TLS_INL_H_
