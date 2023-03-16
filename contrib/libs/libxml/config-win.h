#pragma once

#include "config-linux.h"

#undef HAVE_LIBPTHREAD
#undef HAVE_PTHREAD_H
#undef HAVE_UNISTD_H
#undef HAVE_RAND_R

// Under Windows, use compiler-specific TLS: it seems broken the other way round.
#define HAVE_COMPILER_TLS
