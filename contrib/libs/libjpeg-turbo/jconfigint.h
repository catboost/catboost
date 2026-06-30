#pragma once

#if defined(__arm__) || defined(__ARM__)
#   include "jconfigint-armv7a.h"
#elif defined(__aarch64__) || defined(_M_ARM64)
#   include "jconfigint-armv8a.h"
#elif defined(__i686__) || defined(_M_IX86)
#   include "jconfigint-x86.h"
#elif defined(__wasm__) && !defined(__wasm64__)
#   include "jconfigint-wasm32.h"
#elif defined(__wasm64__)
#   include "jconfigint-wasm64.h"
#else
#   include "jconfigint-linux.h"
#endif
