#include "flock.h"

#ifndef _unix_

    #include <util/generic/utility.h>

    #include "winint.h"
    #include <io.h>
    #include <errno.h>

    #ifdef __cplusplus
extern "C" {
    #endif

    int flock(int fd, int op) {
        return Flock((HANDLE)_get_osfhandle(fd), op);
    }

    int Flock(void* hdl, int op) {
        errno = 0;

        if (hdl == INVALID_HANDLE_VALUE) {
            errno = EBADF;
            return -1;
        }

        DWORD low = 1, high = 0;
        OVERLAPPED io;

        Zero(io);

        UnlockFileEx(hdl, 0, low, high, &io);

        switch (op & ~LOCK_NB) {
            case LOCK_EX:
            case LOCK_SH: {
                auto mode = ((op & ~LOCK_NB) == LOCK_EX) ? LOCKFILE_EXCLUSIVE_LOCK : 0;
                if (op & LOCK_NB) {
                    if (LockFileEx(hdl, mode | LOCKFILE_FAIL_IMMEDIATELY, 0, low, high, &io)) {
                        return 0;
                    } else if (GetLastError() == ERROR_LOCK_VIOLATION) {
                        ClearLastSystemError();
                        errno = EWOULDBLOCK;
                        return -1;
                    }
                } else {
                    if (LockFileEx(hdl, mode, 0, low, high, &io)) {
                        return 0;
                    }
                }
                break;
            }
            case LOCK_UN:
                return 0;
                break;
            default:
                break;
        }
        errno = EINVAL;
        return -1;
    }

    int fsync(int fd) {
        return _commit(fd);
    }

    #ifdef __cplusplus
}
    #endif

#endif
