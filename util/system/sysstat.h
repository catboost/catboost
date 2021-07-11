#pragma once

#include "defaults.h"
#include <sys/stat.h>

#ifdef _win_
    #define S_IRUSR _S_IREAD
    #define S_IWUSR _S_IWRITE
    #define S_IXUSR _S_IEXEC
    #define S_IRWXU (S_IRUSR | S_IWUSR | S_IXUSR)

    #define S_IRGRP _S_IREAD
    #define S_IWGRP _S_IWRITE
    #define S_IXGRP _S_IEXEC
    #define S_IRWXG (S_IRGRP | S_IWGRP | S_IXGRP)

    #define S_IROTH _S_IREAD
    #define S_IWOTH _S_IWRITE
    #define S_IXOTH _S_IEXEC
    #define S_IRWXO (S_IROTH | S_IWOTH | S_IXOTH)
#endif

int Chmod(const char* fname, int mode);
int Umask(int mode);

static constexpr int MODE0777 = (S_IRWXU | S_IRWXG | S_IRWXO);
static constexpr int MODE0775 = (S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
static constexpr int MODE0755 = (S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

int Mkdir(const char* path, int mode);

/* uff... mkfifo(...) is not used now */

#ifdef _unix_
inline int Chmod(const char* fname, int mode) {
    return ::chmod(fname, mode);
}
inline int Umask(int mode) {
    return ::umask(mode);
}
inline int Mkdir(const char* path, int mode) {
    return ::mkdir(path, mode);
}
#endif

#ifdef _win_
inline int Umask(int /*mode*/) {
    /* The only thing this method could make is to set FILE_ATTRIBUTE_READONLY on a handle from 'int open(...)',
       but open() is deprecated. */
    return 0;
}
#endif
