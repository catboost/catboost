#pragma once

#include "fs.h"
#include "file.h"

#include <util/folder/path.h>
#include <util/generic/maybe.h>
#include <util/generic/noncopyable.h>
#include <util/generic/string.h>

class TTempFile: TMoveOnly {
public:
    inline TTempFile(const TString& fname)
        : Name_(fname)
    {
    }

    inline TTempFile(TTempFile&& rhs) noexcept
        : Name_(std::move(rhs.Name_))
    {
        rhs.Name_.Clear();
    }

    inline ~TTempFile() {
        if (Name_.Defined()) {
            NFs::Remove(*Name_);
        }
    }

    TTempFile& operator=(TTempFile&& rhs);

    inline const TString& Name() const {
        return *Name_;
    }

private:
    TMaybe<TString> Name_;
};

class TTempFileHandle: public TTempFile, public TFile {
public:
    TTempFileHandle();
    TTempFileHandle(const TString& fname);

    static TTempFileHandle InCurrentDir(const TString& filePrefix = "yandex", const TString& extension = "tmp");
    static TTempFileHandle InDir(const TFsPath& dirPath, const TString& filePrefix = "yandex", const TString& extension = "tmp");

private:
    TFile CreateFile() const;
};

/*
 * Creates a unique temporary filename in specified directory.
 * If specified directory is NULL or empty, then system temporary directory is used.
 *
 * Note, that the function is not race-free, the file is guaranteed to exist at the time the function returns, but not at the time the returned name is first used.
 * Throws TSystemError on error.
 *
 * Returned filepath has such format: dir/prefixXXXXXX.extension or dir/prefixXXXXXX
 * But win32: dir/preXXXX.tmp (prefix is up to 3 characters, extension is always tmp).
 */
TString MakeTempName(const char* wrkDir = nullptr, const char* prefix = "yandex", const char* extension = "tmp");
