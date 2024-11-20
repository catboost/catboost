#include "fs_win.h"
#include "defaults.h"
#include "maxlen.h"

#include <util/folder/dirut.h>
#include <util/charset/wide.h>
#include "file.h"

#include <winioctl.h>

namespace NFsPrivate {
    static LPCWSTR UTF8ToWCHAR(const TStringBuf str, TUtf16String& wstr) {
        wstr.resize(str.size());
        size_t written = 0;
        if (!UTF8ToWide(str.data(), str.size(), wstr.begin(), written)) {
            return nullptr;
        }
        wstr.erase(written);
        static_assert(sizeof(WCHAR) == sizeof(wchar16), "expect sizeof(WCHAR) == sizeof(wchar16)");
        return (const WCHAR*)wstr.data();
    }

    static TString WCHARToUTF8(const LPWSTR wstr, size_t len) {
        static_assert(sizeof(WCHAR) == sizeof(wchar16), "expect sizeof(WCHAR) == sizeof(wchar16)");

        return WideToUTF8((wchar16*)wstr, len);
    }

    HANDLE CreateFileWithUtf8Name(const TStringBuf fName, ui32 accessMode, ui32 shareMode, ui32 createMode, ui32 attributes, bool inheritHandle) {
        TUtf16String wstr;
        LPCWSTR wname = UTF8ToWCHAR(fName, wstr);
        if (!wname) {
            ::SetLastError(ERROR_INVALID_NAME);
            return INVALID_HANDLE_VALUE;
        }
        SECURITY_ATTRIBUTES secAttrs;
        secAttrs.bInheritHandle = inheritHandle ? TRUE : FALSE;
        secAttrs.lpSecurityDescriptor = nullptr;
        secAttrs.nLength = sizeof(secAttrs);
        return ::CreateFileW(wname, accessMode, shareMode, &secAttrs, createMode, attributes, nullptr);
    }

    bool WinRename(const TString& oldPath, const TString& newPath) {
        TUtf16String op, np;
        LPCWSTR opPtr = UTF8ToWCHAR(oldPath, op);
        LPCWSTR npPtr = UTF8ToWCHAR(newPath, np);
        if (!opPtr || !npPtr) {
            ::SetLastError(ERROR_INVALID_NAME);
            return false;
        }

        return MoveFileExW(opPtr, npPtr, MOVEFILE_REPLACE_EXISTING) != 0;
    }

    bool WinRemove(const TString& path) {
        TUtf16String wstr;
        LPCWSTR wname = UTF8ToWCHAR(path, wstr);
        if (!wname) {
            ::SetLastError(ERROR_INVALID_NAME);
            return false;
        }
        WIN32_FILE_ATTRIBUTE_DATA fad;
        if (::GetFileAttributesExW(wname, GetFileExInfoStandard, &fad)) {
            if (fad.dwFileAttributes & FILE_ATTRIBUTE_READONLY) {
                fad.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;
                ::SetFileAttributesW(wname, fad.dwFileAttributes);
            }
            if (fad.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                return ::RemoveDirectoryW(wname) != 0;
            }
            return ::DeleteFileW(wname) != 0;
        }

        return false;
    }

    bool WinSymLink(const TString& targetName, const TString& linkName) {
        TString tName(targetName);
        {
            size_t pos;
            while ((pos = tName.find('/')) != TString::npos) {
                tName.replace(pos, 1, LOCSLASH_S);
            }
        }
        TUtf16String tstr;
        LPCWSTR wname = UTF8ToWCHAR(tName, tstr);
        TUtf16String lstr;
        LPCWSTR lname = UTF8ToWCHAR(linkName, lstr);

        // we can't create a dangling link to a dir in this way
        ui32 attr = ::GetFileAttributesW(wname);
        if (attr == INVALID_FILE_ATTRIBUTES) {
            TTempBuf result;
            if (GetFullPathNameW(lname, result.Size(), (LPWSTR)result.Data(), 0) != 0) {
                TString fullPath = WideToUTF8(TWtringBuf((const wchar16*)result.Data()));
                TStringBuf linkDir(fullPath);
                linkDir.RNextTok('\\');

                if (linkDir) {
                    TString fullTarget(tName);
                    resolvepath(fullTarget, TString{linkDir});
                    TUtf16String fullTargetW;
                    LPCWSTR ptrFullTarget = UTF8ToWCHAR(fullTarget, fullTargetW);
                    attr = ::GetFileAttributesW(ptrFullTarget);
                }
            }
        }
        return 0 != CreateSymbolicLinkW(lname, wname, attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) ? SYMBOLIC_LINK_FLAG_DIRECTORY : 0);
    }

    bool WinHardLink(const TString& existingPath, const TString& newPath) {
        TUtf16String ep, np;
        LPCWSTR epPtr = UTF8ToWCHAR(existingPath, ep);
        LPCWSTR npPtr = UTF8ToWCHAR(newPath, np);
        if (!epPtr || !npPtr) {
            ::SetLastError(ERROR_INVALID_NAME);
            return false;
        }

        return (CreateHardLinkW(npPtr, epPtr, nullptr) != 0);
    }

    bool WinExists(const TString& path) {
        TUtf16String buf;
        LPCWSTR ptr = UTF8ToWCHAR(path, buf);
        return ::GetFileAttributesW(ptr) != INVALID_FILE_ATTRIBUTES;
    }

    TString WinCurrentWorkingDirectory() {
        TTempBuf result;
        LPWSTR buf = reinterpret_cast<LPWSTR>(result.Data());
        int r = GetCurrentDirectoryW(result.Size() / sizeof(WCHAR), buf);
        if (r == 0) {
            throw TIoSystemError() << "failed to GetCurrentDirectory";
        }
        return WCHARToUTF8(buf, r);
    }

    bool WinSetCurrentWorkingDirectory(const TString& path) {
        TUtf16String wstr;
        LPCWSTR wname = UTF8ToWCHAR(path, wstr);
        if (!wname) {
            ::SetLastError(ERROR_INVALID_NAME);
            return false;
        }
        return SetCurrentDirectoryW(wname);
    }

    bool WinMakeDirectory(const TString& path) {
        TUtf16String buf;
        LPCWSTR ptr = UTF8ToWCHAR(path, buf);
        return CreateDirectoryW(ptr, (LPSECURITY_ATTRIBUTES) nullptr);
    }
    // edited part of <Ntifs.h> from Windows DDK

#define SYMLINK_FLAG_RELATIVE 1

    struct TReparseBufferHeader {
        USHORT SubstituteNameOffset;
        USHORT SubstituteNameLength;
        USHORT PrintNameOffset;
        USHORT PrintNameLength;
    };

    struct TSymbolicLinkReparseBuffer: public TReparseBufferHeader {
        ULONG Flags; // 0 or SYMLINK_FLAG_RELATIVE
        wchar16 PathBuffer[1];
    };

    struct TMountPointReparseBuffer: public TReparseBufferHeader {
        wchar16 PathBuffer[1];
    };

    struct TGenericReparseBuffer {
        wchar16 DataBuffer[1];
    };

    struct REPARSE_DATA_BUFFER {
        ULONG ReparseTag;
        USHORT ReparseDataLength;
        USHORT Reserved;
        union {
            TSymbolicLinkReparseBuffer SymbolicLinkReparseBuffer;
            TMountPointReparseBuffer MountPointReparseBuffer;
            TGenericReparseBuffer GenericReparseBuffer;
        };
    };

    // the end of edited part of <Ntifs.h>

    // For more info see:
    // * https://docs.microsoft.com/en-us/windows/win32/fileio/reparse-points
    // * https://docs.microsoft.com/en-us/windows-hardware/drivers/ifs/fsctl-get-reparse-point
    // * https://docs.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_reparse_data_buffer
    REPARSE_DATA_BUFFER* ReadReparsePoint(HANDLE h, TTempBuf& buf) {
        while (true) {
            DWORD bytesReturned = 0;
            BOOL res = DeviceIoControl(h, FSCTL_GET_REPARSE_POINT, nullptr, 0, buf.Data(), buf.Size(), &bytesReturned, nullptr);
            if (res) {
                REPARSE_DATA_BUFFER* rdb = (REPARSE_DATA_BUFFER*)buf.Data();
                return rdb;
            } else {
                if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
                    buf = TTempBuf(buf.Size() * 2);
                } else {
                    return nullptr;
                }
            }
        }
    }

    TString WinReadLink(const TString& name) {
        TFileHandle h = CreateFileWithUtf8Name(name, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING,
                                               FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, true);
        if (h == INVALID_HANDLE_VALUE) {
            ythrow TIoSystemError() << "can't open file " << name;
        }
        TTempBuf buf;
        REPARSE_DATA_BUFFER* rdb = ReadReparsePoint(h, buf);
        if (rdb == nullptr) {
            ythrow TIoSystemError() << "can't read reparse point " << name;
        }
        if (rdb->ReparseTag == IO_REPARSE_TAG_SYMLINK) {
            wchar16* str = (wchar16*)&rdb->SymbolicLinkReparseBuffer.PathBuffer[rdb->SymbolicLinkReparseBuffer.SubstituteNameOffset / sizeof(wchar16)];
            size_t len = rdb->SymbolicLinkReparseBuffer.SubstituteNameLength / sizeof(wchar16);
            return WideToUTF8(str, len);
        } else if (rdb->ReparseTag == IO_REPARSE_TAG_MOUNT_POINT) {
            wchar16* str = (wchar16*)&rdb->MountPointReparseBuffer.PathBuffer[rdb->MountPointReparseBuffer.SubstituteNameOffset / sizeof(wchar16)];
            size_t len = rdb->MountPointReparseBuffer.SubstituteNameLength / sizeof(wchar16);
            return WideToUTF8(str, len);
        }
        // this reparse point is unsupported in arcadia
        return TString();
    }

    ULONG WinReadReparseTag(HANDLE h) {
        TTempBuf buf;
        REPARSE_DATA_BUFFER* rdb = ReadReparsePoint(h, buf);
        return rdb ? rdb->ReparseTag : 0;
    }

    // we can't use this function to get an analog of unix inode due to a lot of NTFS folders do not have this GUID
    // (it will be 'create' case really)
    /*
bool GetObjectId(const char* path, GUID* id) {
    TFileHandle h = CreateFileWithUtf8Name(path, 0, FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
                                OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT|FILE_FLAG_BACKUP_SEMANTICS, true);
    if (h.IsOpen()) {
        FILE_OBJECTID_BUFFER fob;
        DWORD resSize = 0;
        if (DeviceIoControl(h, FSCTL_CREATE_OR_GET_OBJECT_ID, nullptr, 0, &fob, sizeof(fob), &resSize, nullptr)) {
            Y_ASSERT(resSize == sizeof(fob));
            memcpy(id, &fob.ObjectId, sizeof(GUID));
            return true;
        }
    }
    memset(id, 0, sizeof(GUID));
    return false;
}
*/

} // namespace NFsPrivate
