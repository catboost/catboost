#include "sysstat.h"

#ifdef _win_

    #include "winint.h"
    #include <errno.h>

int Chmod(const char* fname, int mode) {
    if (!fname) {
        errno = EINVAL;
        return -1;
    }
    ui32 fAttr = ::GetFileAttributesA(fname);
    if (fAttr == 0xffffffff)
        return -1;
    if (mode & _S_IWRITE) {
        fAttr &= ~FILE_ATTRIBUTE_READONLY;
    } else {
        fAttr |= FILE_ATTRIBUTE_READONLY;
    }
    if (!::SetFileAttributesA(fname, fAttr)) {
        return -1;
    }
    return 0;
}

int Mkdir(const char* path, int /*mode*/) {
    errno = 0;
    if (!path) {
        errno = EINVAL;
        return -1;
    }
    if (!CreateDirectoryA(path, (LPSECURITY_ATTRIBUTES) nullptr)) {
        ui32 errCode = GetLastError();
        if (errCode == ERROR_ALREADY_EXISTS) {
            errno = EEXIST;
        } else if (errCode == ERROR_PATH_NOT_FOUND) {
            errno = ENOENT;
        } else {
            errno = EINVAL;
        }
        return -1;
    }
    return 0;
}

#endif
