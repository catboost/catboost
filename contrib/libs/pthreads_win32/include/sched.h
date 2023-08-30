#pragma once

#define PTW32_STATIC_LIB
#define PTW32_YANDEX

#if _MSC_VER >= 1900
#define HAVE_STRUCT_TIMESPEC
#endif

#include "../sched.h"
