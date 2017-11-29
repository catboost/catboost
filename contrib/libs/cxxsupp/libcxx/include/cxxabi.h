#pragma once

#if defined(__IOS__)
#include_next <cxxabi.h>
#else
#include <contrib/libs/cxxsupp/libcxxrt/cxxabi.h>
#endif