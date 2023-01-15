#pragma once

#if defined(__linux__) && !defined(__ANDROID__)
#include "config-linux.h"
#endif

#if defined(_MSC_VER)
#include "config-win.h"
#endif
