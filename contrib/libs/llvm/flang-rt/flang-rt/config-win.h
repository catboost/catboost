#pragma once

#include "config-linux.h"

#undef HAVE_STRERROR_R
#define HAVE_DECL_STRERROR_S 1

#undef HAVE_BACKTRACE
