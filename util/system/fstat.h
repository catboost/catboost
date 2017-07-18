#pragma once

#include "fhandle.h"

class TFile;
class TFsPath;
class TString;

struct TFileStat {
    ui32 Mode = 0; /* protection */
    ui32 Uid = 0;  /* user ID of owner */
    ui32 Gid = 0;  /* group ID of owner */

    ui64 NLinks = 0; /* number of hard links */
    ui64 Size = 0;   /* total size, in bytes */

    time_t ATime = 0; /* time of last access */
    time_t MTime = 0; /* time of last modification */
    time_t CTime = 0; /* time of last status change */

public:
    TFileStat();

    bool IsNull() const;

    bool IsFile() const;
    bool IsDir() const;
    bool IsSymlink() const;

    explicit TFileStat(const TFile& f);
    explicit TFileStat(FHANDLE f);
    TFileStat(const TFsPath& fileName, bool nofollow = false);
    TFileStat(const TString& fileName, bool nofollow = false);
    TFileStat(const char* fileName, bool nofollow = false);

    friend bool operator==(const TFileStat& l, const TFileStat& r);
    friend bool operator!=(const TFileStat& l, const TFileStat& r);

private:
    void MakeFromFileName(const char* fileName, bool nofollow);
};

i64 GetFileLength(FHANDLE fd);
i64 GetFileLength(const char* name);
i64 GetFileLength(const TString& name);
