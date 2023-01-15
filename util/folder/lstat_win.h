#pragma once

#include <util/system/defaults.h>
#include "fts.h"

#ifdef _win_
    #include <sys/stat.h>

    #ifdef __cplusplus
extern "C" {
    #endif

    #define _S_IFLNK 0xA000
    int lstat(const char* fileName, stat_struct* fileStat);

    #ifdef __cplusplus
}
    #endif

#endif //_win_
