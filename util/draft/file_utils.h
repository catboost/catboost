#pragma once

#include <util/system/compat.h>
#include <util/system/error.h>
#include <util/system/fs.h>
#include <util/folder/dirut.h>
#include <util/generic/noncopyable.h>
#include <util/generic/string.h>
#include <util/stream/file.h>
#include <util/string/cast.h>
#include <util/generic/yexception.h>

TString GetDir(const TString& filePath);
TString GetFilename(const TString& filePath);

// returns -1 if not found
i64 GetFileSize(const TString& srcLocation);

TString ResolveLocation(TString fileLocation, const TString& homeDir);

/////////////

class WithTmpDir: public TNonCopyable {
private:
    TString TmpDirName;

public:
    WithTmpDir(const TString& tmpDirName)
        : TmpDirName(tmpDirName)
    {
        MakeDirIfNotExist(TmpDirName.c_str());
    }

    const TString& Name() const {
        return TmpDirName;
    }

    ~WithTmpDir() {
        try {
            RemoveDirWithContents(TmpDirName.c_str());
        } catch (std::exception& e) {
            Cerr << "Exception: " << e.what() << Endl;
        } catch (...) {
            Cerr << "Unknown exception while trying to remove tmp dir " << TmpDirName << Endl;
        }
    }
};

class WithUniqTmpDir: public WithTmpDir {
public:
    WithUniqTmpDir(const TString& parentDir = GetSystemTempDir())
        : WithTmpDir(parentDir + LOCSLASH_C + getprogname() + '.' + ToString(getpid()) + '.' + ToString(rand()))
    {
    }
};

class WithTmpFile: public TNonCopyable {
private:
    TString TmpFileName;

public:
    WithTmpFile(const TString& tmpFileName) //  TODO - maybe require file to exist - use MakeTempName
        : TmpFileName(tmpFileName)
    {
    }

    const TString& Name() const {
        return TmpFileName;
    }

    ~WithTmpFile() {
        if (!NFs::Remove(TmpFileName))
            Cerr << "Failed to remove file " << TmpFileName << ", errno = " << errno << Endl;
    }
};
