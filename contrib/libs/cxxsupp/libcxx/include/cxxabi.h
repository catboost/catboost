#pragma once

#if defined(__IOS__) || defined(__ANDROID__)
#include_next <cxxabi.h>
#elif defined(_WIN32)
// pass
#else
#include <contrib/libs/cxxsupp/libcxxrt/cxxabi.h>
#endif
