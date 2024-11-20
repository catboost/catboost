#include "fstat.h"
#include "file.h"

#include <sys/stat.h>

#include <util/folder/path.h>

#include <cerrno>

#if defined(_win_)
    #include "fs_win.h"

    #ifdef _S_IFLNK
        #undef _S_IFLNK
    #endif
    #define _S_IFLNK 0x80000000

// See https://docs.microsoft.com/en-us/windows/win32/fileio/file-attribute-constants
// for possible flag values
static ui32 GetWinFileType(DWORD fileAttributes, ULONG reparseTag) {
    // I'm not really sure, why it is done like this. MSDN tells that
    // FILE_ATTRIBUTE_DEVICE is reserved for system use. Some more info is
    // available at https://stackoverflow.com/questions/3419527/setting-file-attribute-device-in-visual-studio
    // We should probably replace this with GetFileType call and check for
    // FILE_TYPE_CHAR and FILE_TYPE_PIPE return values.
    if (fileAttributes & FILE_ATTRIBUTE_DEVICE) {
        return _S_IFCHR;
    }

    if (fileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
        // We consider IO_REPARSE_TAG_SYMLINK and IO_REPARSE_TAG_MOUNT_POINT
        // both to be symlinks to align with current WinReadLink behaviour.
        if (reparseTag == IO_REPARSE_TAG_SYMLINK || reparseTag == IO_REPARSE_TAG_MOUNT_POINT) {
            return _S_IFLNK;
        }
    }

    if (fileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
        return _S_IFDIR;
    }

    return _S_IFREG;
}

static ui32 GetFileMode(DWORD fileAttributes, ULONG reparseTag) {
    ui32 mode = 0;
    if (fileAttributes == 0xFFFFFFFF) {
        return mode;
    }

    mode |= GetWinFileType(fileAttributes, reparseTag);

    if ((fileAttributes & FILE_ATTRIBUTE_READONLY) == 0) {
        mode |= _S_IWRITE;
    }
    return mode;
}

    #define S_ISDIR(st_mode) (st_mode & _S_IFDIR)
    #define S_ISREG(st_mode) (st_mode & _S_IFREG)
    #define S_ISLNK(st_mode) (st_mode & _S_IFLNK)

struct TSystemFStat: public BY_HANDLE_FILE_INFORMATION {
    ULONG ReparseTag = 0;
};

#elif defined(_unix_)
using TSystemFStat = struct stat;
#else
    #error unsupported platform
#endif

#if defined(_unix_)
static void MakeStatFromStructStat(TFileStat& st, const struct stat& fs) {
    st.Mode = fs.st_mode;
    st.NLinks = fs.st_nlink;
    st.Uid = fs.st_uid;
    st.Gid = fs.st_gid;
    st.Size = fs.st_size;
    st.AllocationSize = fs.st_blocks * 512;

    #if defined(_linux_)
    st.ATime = fs.st_atim.tv_sec;
    st.ATimeNSec = fs.st_atim.tv_nsec;

    st.MTime = fs.st_mtim.tv_sec;
    st.MTimeNSec = fs.st_mtim.tv_nsec;

    st.CTime = fs.st_ctim.tv_sec;
    st.CTimeNSec = fs.st_ctim.tv_nsec;
    #elif defined(_darwin_)
    st.ATime = fs.st_atimespec.tv_sec;
    st.ATimeNSec = fs.st_atimespec.tv_nsec;

    st.MTime = fs.st_mtimespec.tv_sec;
    st.MTimeNSec = fs.st_mtimespec.tv_nsec;

    st.CTime = fs.st_birthtimespec.tv_sec;
    st.CTimeNSec = fs.st_birthtimespec.tv_nsec;
    #else
    // Fallback.
    st.ATime = fs.st_atime;
    st.MTime = fs.st_mtime;
    st.CTime = fs.st_ctime;
    #endif

    st.INode = fs.st_ino;
}
#endif

static void MakeStat(TFileStat& st, const TSystemFStat& fs) {
#ifdef _unix_
    MakeStatFromStructStat(st, fs);
#else
    timespec timeSpec;
    FileTimeToTimespec(fs.ftCreationTime, &timeSpec);
    st.CTime = timeSpec.tv_sec;
    st.CTimeNSec = timeSpec.tv_nsec;

    FileTimeToTimespec(fs.ftLastAccessTime, &timeSpec);
    st.ATime = timeSpec.tv_sec;
    st.ATimeNSec = timeSpec.tv_nsec;

    FileTimeToTimespec(fs.ftLastWriteTime, &timeSpec);
    st.MTime = timeSpec.tv_sec;
    st.MTimeNSec = timeSpec.tv_nsec;

    st.NLinks = fs.nNumberOfLinks;
    st.Mode = GetFileMode(fs.dwFileAttributes, fs.ReparseTag);
    st.Uid = 0;
    st.Gid = 0;
    st.Size = ((ui64)fs.nFileSizeHigh << 32) | fs.nFileSizeLow;
    st.AllocationSize = st.Size; // FIXME
    st.INode = ((ui64)fs.nFileIndexHigh << 32) | fs.nFileIndexLow;
#endif
}

static bool GetStatByHandle(TSystemFStat& fs, FHANDLE f) {
#ifdef _win_
    if (!GetFileInformationByHandle(f, &fs)) {
        return false;
    }
    if (fs.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
        fs.ReparseTag = NFsPrivate::WinReadReparseTag(f);
    }
    return true;
#else
    return !fstat(f, &fs);
#endif
}

static bool GetStatByName(TSystemFStat& fs, const char* fileName, bool nofollow) {
#ifdef _win_
    TFileHandle h = NFsPrivate::CreateFileWithUtf8Name(
        fileName,
        FILE_READ_ATTRIBUTES | FILE_READ_EA,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        OPEN_EXISTING,
        (nofollow ? FILE_FLAG_OPEN_REPARSE_POINT : 0) | FILE_FLAG_BACKUP_SEMANTICS,
        true);
    if (!h.IsOpen()) {
        return false;
    }
    return GetStatByHandle(fs, h);
#else
    return !(nofollow ? lstat : stat)(fileName, &fs);
#endif
}

TFileStat::TFileStat() = default;

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

#if defined(_unix_)
TFileStat::TFileStat(const struct stat& st) {
    MakeStatFromStructStat(*this, st);
}
#endif

bool TFileStat::operator==(const TFileStat& other) const noexcept = default;

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

i64 GetFileLength(FHANDLE fd) {
#if defined(_win_)
    LARGE_INTEGER pos;
    if (!::GetFileSizeEx(fd, &pos)) {
        return -1L;
    }
    return pos.QuadPart;
#elif defined(_unix_)
    struct stat statbuf;
    if (::fstat(fd, &statbuf) != 0) {
        return -1L;
    }
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
    if (h == INVALID_HANDLE_VALUE) {
        return -1;
    }
    FindClose(h);
    return (((i64)fData.nFileSizeHigh) * (i64(MAXDWORD) + 1)) + (i64)fData.nFileSizeLow;
#elif defined(_unix_)
    struct stat buf;
    int r = ::stat(name, &buf);
    if (r == -1) {
        return -1;
    }
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
