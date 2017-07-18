#include <util/stream/pipe.h>
#include <util/string/split.h>
#include <util/system/fstat.h>
#include <util/system/fs.h>

#include "file_utils.h"

TString GetDir(const TString& filePath) {
    size_t delimPos = filePath.rfind('/');
#ifdef _win32_
    if (delimPos == TString::npos) {
        // There's a possibility of Windows-style path
        delimPos = filePath.rfind('\\');
    }
#endif
    return (delimPos == TString::npos) ? TString(".") + LOCSLASH_C : filePath.substr(0, delimPos + 1);
}

TString GetFilename(const TString& filePath) {
    size_t delimPos = filePath.rfind('/');
#ifdef _win32_
    if (delimPos == TString::npos) {
        // There's a possibility of Windows-style path
        delimPos = filePath.rfind('\\');
    }
#endif
    return (delimPos == TString::npos) ? filePath : filePath.substr(delimPos + 1);
}

i64 GetFileSize(const TString& srcLocation) {
    size_t delimPos = srcLocation.find(':');
    // heuristics that says that 1-letters are probably Win drives
    bool isRemote = (delimPos != TString::npos) && (delimPos != 1);
    if (!isRemote) {
        return GetFileLength(srcLocation);
    }

#ifdef _unix_
    bool useRsync;
    TString command;
    if (srcLocation.StartsWith("rsync://")) {
        useRsync = true;
        command = "rsync " + srcLocation;
    } else {
        useRsync = false;
        TString host = srcLocation.substr(0, delimPos);
        TString path = srcLocation.substr(delimPos + 1);
        command = "rsh " + host + " \"ls -ln " + path + "\"";
    }

    TPipeInput in(command);

    TString s;
    if (!in.ReadLine(s))
        return -1;

    yvector<TString> el;
    Split(~s, " ", el);

    i64 size;
    try {
        size = FromString<i64>(el[useRsync ? 1 : 4]);
    } catch (...) {
        return -1;
    }
    return size;
#else
    ythrow yexception() << "rsync is not available on non-Unix platforms";
#endif
}

TString ResolveLocation(TString fileLocation, const TString& homeDir) {
    size_t delimPos = fileLocation.find(':');
    if ((delimPos == TString::npos) || (delimPos == 1)) {
        if (!resolvepath(fileLocation, homeDir))
            ythrow yexception() << "Failed to resolve fileLocation " << ~fileLocation << " with homeDir " << ~homeDir;
    }
    return fileLocation;
}
