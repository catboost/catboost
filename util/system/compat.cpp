#include "compat.h"
#include "progname.h"

#include <util/generic/string.h>

#ifdef _win_
    #include "winint.h"
    #include <io.h>
#endif

#ifndef HAVE_NATIVE_GETPROGNAME
const char* getprogname() {
    return GetProgramName().data();
}
#endif

#ifdef _win_

void sleep(i64 len) {
    Sleep((unsigned long)len * 1000);
}

void usleep(i64 len) {
    Sleep((unsigned long)len / 1000);
}

    #include <fcntl.h>
int ftruncate(int fd, i64 length) {
    return _chsize_s(fd, length);
}
int truncate(const char* name, i64 length) {
    int fd = ::_open(name, _O_WRONLY);
    int ret = ftruncate(fd, length);
    ::close(fd);
    return ret;
}
#endif
