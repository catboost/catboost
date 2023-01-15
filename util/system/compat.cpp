#include "compat.h"
#include "defaults.h"
#include "progname.h"

#include <util/generic/string.h>

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <cstdlib>

#ifdef _win_
#include "winint.h"
#include <io.h>
#endif

#ifndef HAVE_NATIVE_GETPROGNAME
const char* getprogname() {
    return GetProgramName().data();
}
#endif

extern "C" {
#if !defined(_MSC_VER)
char* strrev(char* s) {
    char* d = s;
    char* r = strchr(s, 0);
    for (--r; d < r; ++d, --r) {
        char tmp;
        tmp = *d;
        *d = *r;
        *r = tmp;
    }
    return s;
}

char* strupr(char* s) {
    char* d;
    for (d = s; *d; ++d)
        *d = (char)toupper((int)*d);
    return s;
}

char* strlwr(char* s) {
    char* d;
    for (d = s; *d; ++d)
        *d = (char)tolower((int)*d);
    return s;
}

#endif

#if !defined(__linux__) && !defined(__FreeBSD__)
char* stpcpy(char* dst, const char* src) {
    size_t len = strlen(src);
    return ((char*)memcpy(dst, src, len + 1)) + len;
}
#endif

#if defined(_MSC_VER) || defined(_sun_)
/*
 * Get next token from string *stringp, where tokens are possibly-empty
 * strings separated by characters from delim.
 *
 * Writes NULs into the string at *stringp to end tokens.
 * delim need not remain constant from call to call.
 * On return, *stringp points past the last NUL written (if there might
 * be further tokens), or is NULL (if there are definitely no more tokens).
 *
 * If *stringp is NULL, strsep returns NULL.
 */
char* strsep(char** stringp, const char* delim) {
    char* s;
    const char* spanp;
    int c, sc;
    char* tok;

    if ((s = *stringp) == nullptr)
        return nullptr;
    for (tok = s;;) {
        c = *s++;
        spanp = delim;
        do {
            if ((sc = *spanp++) == c) {
                if (c == 0)
                    s = nullptr;
                else
                    s[-1] = 0;
                *stringp = s;
                return tok;
            }
        } while (sc != 0);
    }
    /* NOTREACHED */
}
#endif

} // extern "C"

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
