#pragma once

#include "pyconfig.freebsd.h"

#undef HAVE_IEEEFP_H
#undef HAVE_STAT_TV_NSEC
#undef HAVE_CHROOT
#undef HAVE_LIBUTIL_H
#undef HAVE_GETRESGID
#undef HAVE_SETRESGID
#undef HAVE_GETRESUID
#undef HAVE_SETRESUID
#undef HAVE_SEM_TIMEDWAIT
#undef POSIX_SEMAPHORES_NOT_ENABLED

#define HAVE_STAT_TV_NSEC2 1
#define _UNIX_
