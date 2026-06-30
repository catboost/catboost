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
    TString filePath;

    if (wrkDir && *wrkDir) {
        filePath += wrkDir;
    } else {
        filePath += GetSystemTempDir();
    }

#ifdef _win32_
    // https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-gettempfilenamea?redirectedfrom=MSDN
    const unsigned int DirPathMaxLen = 247;
    if (filePath.length() <= DirPathMaxLen) {
        // it always takes up to 3 characters, no more
        char winFilePath[MAX_PATH];
        if (GetTempFileName(filePath.c_str(), (prefix) ? (prefix) : "yan", 0,
                            winFilePath)) {
            return winFilePath;
        }
    }
#endif // _win32_

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

    ythrow TSystemError() << "can not create temp name(" << wrkDir << ", " << prefix << ", " << extension << ")";
}
