#pragma once

#if defined(__IOS__)
#define Y_HDR <../include/cxxabi.h>
#include Y_HDR
#undef Y_HDR
#else
#include <contrib/libs/cxxsupp/libcxxrt/cxxabi.h>
#endif
