#include "tempfile.h"

#include <util/folder/dirut.h>
#include <util/generic/yexception.h>

#include <cstring>

#ifdef _win32_
    #include "winint.h"
    #include <io.h>
#else
    #include <unistd.h>
#endif

extern "C" int mkstemps(char* path, int slen);

TString MakeTempName(const char* wrkDir, const char* prefix, const char* extension) {
#ifndef _win32_
    TString filePath;

    if (wrkDir && *wrkDir) {
        filePath += wrkDir;
    } else {
        filePath += GetSystemTempDir();
    }

    if (filePath.back() != '/') {
        filePath += '/';
    }

    if (prefix) {
        filePath += prefix;
    }

    filePath += "XXXXXX"; // mkstemps requirement

    size_t extensionPartLength = 0;
    if (extension && *extension) {
        if (extension[0] != '.') {
            filePath += '.';
            extensionPartLength += 1;
        }
        filePath += extension;
        extensionPartLength += strlen(extension);
    }

    int fd = mkstemps(const_cast<char*>(filePath.data()), extensionPartLength);
    if (fd >= 0) {
        close(fd);
        return filePath;
    }
#else
    char tmpDir[MAX_PATH + 1]; // +1 -- for terminating null character
    char filePath[MAX_PATH];
    const char* pDir = 0;

    if (wrkDir && *wrkDir) {
        pDir = wrkDir;
    } else if (GetTempPath(MAX_PATH + 1, tmpDir)) {
        pDir = tmpDir;
    }

    // it always takes up to 3 characters, no more
    if (GetTempFileName(pDir, (prefix) ? (prefix) : "yan", 0, filePath)) {
        return filePath;
    }
#endif

    ythrow TSystemError() << "can not create temp name(" << wrkDir << ", " << prefix << ", " << extension << ")";
}
