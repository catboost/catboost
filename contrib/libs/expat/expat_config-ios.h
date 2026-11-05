#pragma once

#include "expat_config-linux.h"

#define HAVE_ARC4RANDOM 1
#define HAVE_ARC4RANDOM_BUF 1
#undef HAVE_GETENTROPY
#undef HAVE_GETRANDOM
#undef HAVE_SYSCALL_GETRANDOM
#undef XML_DEV_URANDOM
