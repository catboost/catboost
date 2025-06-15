#pragma once

#include "jconfigint-linux.h"

#undef SIZEOF_SIZE_T
#define SIZEOF_SIZE_T  4

#ifdef __ANDROID__
#undef WITH_SIMD
#endif
