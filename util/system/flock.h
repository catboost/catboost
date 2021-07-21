#pragma once

#include "error.h"
#include "defaults.h"
#include "file.h"

#if defined(_unix_)

    #include <sys/file.h>
    #include <fcntl.h>

static inline int Flock(int fd, int op) {
    return flock(fd, op);
}

#else // not _unix_

    #ifdef __cplusplus
extern "C" {
    #endif

    #define LOCK_SH 1 /* shared lock */
    #define LOCK_EX 2 /* exclusive lock */
    #define LOCK_NB 4 /* don't block when locking */
    #define LOCK_UN 8 /* unlock */

    int Flock(void* hndl, int operation);
    int flock(int fd, int operation);
    int fsync(int fd);

    #ifdef __cplusplus
}
    #endif

#endif // not _unix_
