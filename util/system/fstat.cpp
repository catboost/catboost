#include "fstat.h"
#include "file.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <util/datetime/systime.h>
#include <util/generic/string.h>
#include <util/folder/path.h>

#include <errno.h>

#if defined(_win_)
#include "fs_win.h"

#ifdef _S_IFLNK
#undef _S_IFLNK
#endif
#define _S_IFLNK 0x80000000

ui32 GetFileMode(DWORD fileAttributes) {
    ui32 mode = 0;
    if (fileAttributes == 0xFFFFFFFF)
        return mode;
    if (fileAttributes & FILE_ATTRIBUTE_DEVICE)
        mode |= _S_IFCHR;
    if (fileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)
        mode |= _S_IFLNK; // todo: was undefined by the moment of writing this code
    if (fileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        mode |= _S_IFDIR;
    if (fileAttributes & (FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_ARCHIVE))
        mode |= _S_IFREG;
    return mode;
}

#define S_ISDIR(st_mode) (st_mode & _S_IFDIR)
#define S_ISREG(st_mode) (st_mode & _S_IFREG)
#define S_ISLNK(st_mode) (st_mode & _S_IFLNK)

using TSystemFStat = BY_HANDLE_FILE_INFORMATION;

#else

using TSystemFStat = struct stat;

#endif

static void MakeStat(TFileStat& st, const TSystemFStat& fs) {
#ifdef _unix_
    st.Mode = fs.st_mode;
    st.NLinks = fs.st_nlink;
    st.Uid = fs.st_uid;
    st.Gid = fs.st_gid;
    st.Size = fs.st_size;
    st.ATime = fs.st_atime;
    st.MTime = fs.st_mtime;
    st.CTime = fs.st_ctime;
    st.INode = fs.st_ino;
#else
    timeval tv;
    FileTimeToTimeval(&fs.ftCreationTime, &tv);
    st.CTime = tv.tv_sec;
    FileTimeToTimeval(&fs.ftLastAccessTime, &tv);
    st.ATime = tv.tv_sec;
    FileTimeToTimeval(&fs.ftLastWriteTime, &tv);
    st.MTime = tv.tv_sec;
    st.NLinks = fs.nNumberOfLinks;
    st.Mode = GetFileMode(fs.dwFileAttributes);
    st.Uid = 0;
    st.Gid = 0;
    st.Size = ((ui64)fs.nFileSizeHigh << 32) | fs.nFileSizeLow;
    st.INode = ((ui64)fs.nFileIndexHigh << 32) | fs.nFileIndexLow;
#endif
}

static bool GetStatByHandle(TSystemFStat& fs, FHANDLE f) {
#ifdef _win_
    return GetFileInformationByHandle(f, &fs);
#else
    return !fstat(f, &fs);
#endif
}

static bool GetStatByName(TSystemFStat& fs, const char* fileName, bool nofollow) {
#ifdef _win_
    TFileHandle h = NFsPrivate::CreateFileWithUtf8Name(fileName, FILE_READ_ATTRIBUTES | FILE_READ_EA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                                       OPEN_EXISTING,
                                                       (nofollow ? FILE_FLAG_OPEN_REPARSE_POINT : 0) | FILE_FLAG_BACKUP_SEMANTICS, true);
    return GetStatByHandle(fs, h);
#else
    return !(nofollow ? lstat : stat)(fileName, &fs);
#endif
}

TFileStat::TFileStat() {
}

TFileStat::TFileStat(const TFile& f) {
    *this = TFileStat(f.GetHandle());
}

TFileStat::TFileStat(FHANDLE f) {
    TSystemFStat st;
    if (GetStatByHandle(st, f)) {
        MakeStat(*this, st);
    } else {
        *this = TFileStat();
    }
}

void TFileStat::MakeFromFileName(const char* fileName, bool nofollow) {
    TSystemFStat st;
    if (GetStatByName(st, fileName, nofollow)) {
        MakeStat(*this, st);
    } else {
        *this = TFileStat();
    }
}

TFileStat::TFileStat(const TFsPath& fileName, bool nofollow) {
    MakeFromFileName(fileName.GetPath().data(), nofollow);
}

TFileStat::TFileStat(const TString& fileName, bool nofollow) {
    MakeFromFileName(fileName.data(), nofollow);
}

TFileStat::TFileStat(const char* fileName, bool nofollow) {
    MakeFromFileName(fileName, nofollow);
}

bool TFileStat::IsNull() const noexcept {
    return *this == TFileStat();
}

bool TFileStat::IsFile() const noexcept {
    return S_ISREG(Mode);
}

bool TFileStat::IsDir() const noexcept {
    return S_ISDIR(Mode);
}

bool TFileStat::IsSymlink() const noexcept {
    return S_ISLNK(Mode);
}

bool operator==(const TFileStat& l, const TFileStat& r) noexcept {
    return l.Mode == r.Mode &&
           l.Uid == r.Uid &&
           l.Gid == r.Gid &&
           l.NLinks == r.NLinks &&
           l.Size == r.Size &&
           l.ATime == r.ATime &&
           l.MTime == r.MTime &&
           l.CTime == r.CTime;
}

bool operator!=(const TFileStat& l, const TFileStat& r) noexcept {
    return !(l == r);
}

i64 GetFileLength(FHANDLE fd) {
#if defined(_win_)
    LARGE_INTEGER pos;
    if (!::GetFileSizeEx(fd, &pos))
        return -1L;
    return pos.QuadPart;
#elif defined(_unix_)
    struct stat statbuf;
    if (::fstat(fd, &statbuf) != 0)
        return -1L;
    if (!(statbuf.st_mode & (S_IFREG | S_IFBLK | S_IFCHR))) {
        // st_size only makes sense for regular files or devices
        errno = EINVAL;
        return -1L;
    }
    return statbuf.st_size;
#else
#error unsupported platform
#endif
}

i64 GetFileLength(const char* name) {
#if defined(_win_)
    WIN32_FIND_DATA fData;
    HANDLE h = FindFirstFileA(name, &fData);
    if (h == INVALID_HANDLE_VALUE)
        return -1;
    FindClose(h);
    return (((i64)fData.nFileSizeHigh) * (i64(MAXDWORD) + 1)) + (i64)fData.nFileSizeLow;
#elif defined(_unix_)
    struct stat buf;
    int r = ::stat(name, &buf);
    if (r == -1)
        return -1;
    if (!(buf.st_mode & (S_IFREG | S_IFBLK | S_IFCHR))) {
        // st_size only makes sense for regular files or devices
        errno = EINVAL;
        return -1;
    }
    return (i64)buf.st_size;
#else
#error unsupported platform
#endif
}

i64 GetFileLength(const TString& name) {
    return GetFileLength(name.data());
}
