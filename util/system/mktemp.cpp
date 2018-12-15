#include "mktemp.h"

#include <util/folder/dirut.h>
#include <util/generic/yexception.h>
#include <util/stream/file.h>

#include <cerrno>
#include <cstring>

#ifdef _win32_
#include "winint.h"
#include <io.h>
#else
#include <unistd.h>
#include <stdlib.h>
#endif

extern "C" int mkstemps(char* path, int slen);

#ifdef _win32_
static char* WinMkTemp(const char* wrkDir, const char* prefix) {
    char* fname = new char[MAX_PATH];

    if (GetTempFileName(wrkDir, (prefix) ? (prefix) : "yand", 0, fname))
        return fname;
    else {
        delete[] fname;
        return nullptr;
    }
}
#endif

// Create temporary file with "tmp" extension
static char* makeTempName(const char* wrkDir, const char* prefix) {
    char* buf = nullptr;

#ifndef _win32_
    int buflen = 20;

    TString sysTmp;

    if (wrkDir && *wrkDir) {
        buflen += strlen(wrkDir) + 1; // +1 -- for '/' after dir name
    } else {
        sysTmp = GetSystemTempDir();
        buflen += sysTmp.size() + 1;
    }

    if (prefix)
        buflen += strlen(prefix);
    buf = new char[buflen + 1];

    if (wrkDir && *wrkDir)
        strcpy(buf, wrkDir);
    else
        strcpy(buf, sysTmp.data());

    if (buf[strlen(buf) - 1] != '/')
        strcat(buf, "/");

    if (prefix)
        strcat(buf, prefix);

    strcat(buf, "XXXXXX.tmp");

    int fd = mkstemps(buf, 4);
    if (fd < 0) {
        delete[] buf;
        buf = nullptr;
    } else {
        close(fd);
    }
#else
    const int TmpDirSize = 1024;
    char TmpDir[TmpDirSize];
    const char* pDir = 0;

    if (wrkDir && *wrkDir)
        pDir = wrkDir;
    else if (GetTempPath(TmpDirSize, TmpDir))
        pDir = TmpDir;

    buf = WinMkTemp(pDir, prefix);
#endif
    return buf;
}

TString MakeTempName(const char* wrkDir, const char* prefix) {
    TArrayHolder<char> ret(makeTempName(wrkDir, prefix));

    if (!ret) {
        ythrow TSystemError() << "can not create temp name(" << wrkDir << ", " << prefix << ")";
    }

    return ret.Get();
}
