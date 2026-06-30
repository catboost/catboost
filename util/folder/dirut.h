#pragma once

#include <util/system/defaults.h>
#include <util/system/sysstat.h>
#include <util/system/fs.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

#include <sys/types.h>

#include <cerrno>
#include <cstdlib>

#ifdef _win32_
    #include <util/system/winint.h>
    #include <direct.h>
    #include <malloc.h>
    #include <time.h>
    #include <io.h>
    #include "dirent_win.h"

// these live in mktemp_system.cpp
extern "C" int mkstemps(char* path, int slen);
char* mkdtemp(char* path);

#else
    #ifdef _sun_
        #include <alloca.h>

char* mkdtemp(char* path);
    #endif
    #include <unistd.h>
    #include <pwd.h>
    #include <dirent.h>
    #ifndef DT_DIR
        #include <sys/stat.h>
    #endif
#endif

bool IsDir(const TString& path);

int mkpath(char* path, int mode = 0777);

TString GetHomeDir();

void MakeDirIfNotExist(const char* path, int mode = 0777);

inline void MakeDirIfNotExist(const TString& path, int mode = 0777) {
    MakeDirIfNotExist(path.data(), mode);
}

/// Create path making parent directories as needed
void MakePathIfNotExist(const char* path, int mode = 0777);

void SlashFolderLocal(TString& folder);
bool correctpath(TString& filename);
bool resolvepath(TString& folder, const TString& home);

char GetDirectorySeparator();
const char* GetDirectorySeparatorS();

void RemoveDirWithContents(TString dirName);

const char* GetFileNameComponent(const char* f);

inline TString GetFileNameComponent(const TString& f) {
    return GetFileNameComponent(f.data());
}

/// RealPath doesn't guarantee trailing separator to be stripped or left in place for directories.
TString RealPath(const TString& path);     // throws
TString RealLocation(const TString& path); /// throws; last file name component doesn't need to exist

TString GetSystemTempDir();

int MakeTempDir(char path[/*FILENAME_MAX*/], const char* prefix);

int ResolvePath(const char* rel, const char* abs, char res[/*FILENAME_MAX*/], bool isdir = false);
TString ResolvePath(const char* rel, const char* abs, bool isdir = false);
TString ResolvePath(const char* path, bool isDir = false);

TString ResolveDir(const char* path);

bool SafeResolveDir(const char* path, TString& result);

TString GetDirName(const TString& path);

TString GetBaseName(const TString& path);

TString StripFileComponent(const TString& fileName);

class TExistenceChecker {
public:
    TExistenceChecker(bool strict = false)
        : Strict(strict)
    {
    }

    void SetStrict(bool strict) {
        Strict = strict;
    }

    bool IsStrict() const {
        return Strict;
    }

    const char* Check(const char* fname) const {
        if (!fname || !*fname) {
            return nullptr;
        }
        if (Strict) {
            NFs::EnsureExists(fname);
        } else if (!NFs::Exists(fname)) {
            fname = nullptr;
        }
        return fname;
    }

private:
    bool Strict;
};
