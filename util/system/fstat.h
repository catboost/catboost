#pragma once

#include <util/generic/fwd.h>
#include <util/system/fhandle.h>

class TFile;
class TFsPath;
struct stat;

struct TFileStat {
    ui32 Mode = 0; /* protection */
    ui32 Uid = 0;  /* user ID of owner */
    ui32 Gid = 0;  /* group ID of owner */

    ui64 NLinks = 0;         /* number of hard links */
    ui64 Size = 0;           /* total size, in bytes */
    ui64 INode = 0;          /* inode number */
    ui64 AllocationSize = 0; /* number of bytes allocated on the disk */

    time_t ATime = 0;   /* time of last access */
    long ATimeNSec = 0; /* nsec of last access */
    time_t MTime = 0;   /* time of last modification */
    long MTimeNSec = 0; /* nsec of last modification */
    time_t CTime = 0;   /* time of last status change */
    long CTimeNSec = 0; /* nsec of last status change */

    TFileStat();

    bool IsNull() const noexcept;

    bool IsFile() const noexcept;
    bool IsDir() const noexcept;
    bool IsSymlink() const noexcept;

#if defined(_unix_)
    explicit TFileStat(const struct stat& fs);
#endif
    explicit TFileStat(const TFile& f);
    explicit TFileStat(FHANDLE f);
    TFileStat(const TFsPath& fileName, bool nofollow = false);
    TFileStat(const TString& fileName, bool nofollow = false);
    TFileStat(const char* fileName, bool nofollow = false);

    bool operator==(const TFileStat& other) const noexcept;

private:
    void MakeFromFileName(const char* fileName, bool nofollow);
};

i64 GetFileLength(FHANDLE fd);
i64 GetFileLength(const char* name);
i64 GetFileLength(const TString& name);
