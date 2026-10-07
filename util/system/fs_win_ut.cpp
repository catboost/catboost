#include "fs.h"
#include "fs_win.h"

#include <winioctl.h>

#include <library/cpp/testing/unittest/registar.h>

#include "fileapi.h"

#include "error.h"
#include "file.h"
#include "fstat.h"
#include "win_undef.h"
#include <util/charset/wide.h>
#include <util/folder/path.h>
#include <util/generic/scope.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/stream/output.h>

#include <cstddef>
#include <cstring>

static void Touch(const TFsPath& path) {
    TFile file(path, CreateAlways | WrOnly);
    file.Write("1115", 4);
}

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

static void SetReadOnly(const TFsPath& path) {
    TUtf16String wstr;
    LPCWSTR wname = UTF8ToWCHAR(static_cast<const TString&>(path), wstr);
    UNIT_ASSERT(wname);
    UNIT_ASSERT(::SetFileAttributesW(wname, FILE_ATTRIBUTE_READONLY));
}

namespace {
    // The layout FSCTL_SET_REPARSE_POINT expects for a mount point, taken from
    // <Ntifs.h> the same way fs_win.cpp takes it: the SDK ships no such header.
    struct TMountPointReparseData {
        ULONG ReparseTag;
        USHORT ReparseDataLength;
        USHORT Reserved;
        USHORT SubstituteNameOffset;
        USHORT SubstituteNameLength;
        USHORT PrintNameOffset;
        USHORT PrintNameLength;
        wchar16 PathBuffer[1];
    };

    constexpr size_t REPARSE_HEADER_SIZE = offsetof(TMountPointReparseData, SubstituteNameOffset);
    constexpr size_t PATH_BUFFER_OFFSET = offsetof(TMountPointReparseData, PathBuffer);
    static_assert(REPARSE_HEADER_SIZE == 8);
    static_assert(PATH_BUFFER_OFFSET == 16);

    // A junction, unlike a symlink, asks for no privileges whatsoever, which
    // makes it the one reparse point a test may rely on creating anywhere.
    // Its target is always stored as an absolute NT path.
    bool CreateJunction(const TString& junction, const TString& absoluteTarget) {
        if (!NFsPrivate::WinMakeDirectory(junction)) {
            return false;
        }

        const TUtf16String substituteName = UTF8ToWide(R"(\??\)" + absoluteTarget);
        const TUtf16String printName = UTF8ToWide(absoluteTarget);
        const size_t substituteBytes = substituteName.size() * sizeof(wchar16);
        const size_t printBytes = printName.size() * sizeof(wchar16);

        // Both names are stored NUL-terminated, one after the other.
        TVector<char> buffer(
            PATH_BUFFER_OFFSET + substituteBytes + printBytes + 2 * sizeof(wchar16),
            0);
        auto& data = *reinterpret_cast<TMountPointReparseData*>(buffer.data());
        data.ReparseTag = IO_REPARSE_TAG_MOUNT_POINT;
        data.ReparseDataLength = static_cast<USHORT>(buffer.size() - REPARSE_HEADER_SIZE);
        data.SubstituteNameOffset = 0;
        data.SubstituteNameLength = static_cast<USHORT>(substituteBytes);
        data.PrintNameOffset = static_cast<USHORT>(substituteBytes + sizeof(wchar16));
        data.PrintNameLength = static_cast<USHORT>(printBytes);
        std::memcpy(data.PathBuffer, substituteName.data(), substituteBytes);
        std::memcpy(
            reinterpret_cast<char*>(data.PathBuffer) + data.PrintNameOffset,
            printName.data(),
            printBytes);

        TFileHandle h = NFsPrivate::CreateFileWithUtf8Name(
            junction,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
            true);
        if (h == INVALID_HANDLE_VALUE) {
            return false;
        }

        DWORD returned = 0;
        return ::DeviceIoControl(h, FSCTL_SET_REPARSE_POINT, buffer.data(),
                                 static_cast<DWORD>(buffer.size()), nullptr, 0,
                                 &returned, nullptr);
    }

    // Wine reports success from CreateSymbolicLinkW and from the reparse point
    // ioctl without creating anything, so nothing here can be asserted there.
    bool IsWine() {
        if (::RegOpenKeyExA(HKEY_CURRENT_USER, R"(Software\Wine)", 0, KEY_READ, nullptr) == ERROR_SUCCESS) {
            return true;
        }
        if (::RegOpenKeyExA(HKEY_LOCAL_MACHINE, R"(Software\Wine)", 0, KEY_READ, nullptr) == ERROR_SUCCESS) {
            return true;
        }

        HMODULE ntdll = ::GetModuleHandleA("ntdll.dll");
        return ntdll && ::GetProcAddress(ntdll, "wine_get_version");
    }

    bool IsProcessElevated() {
        HANDLE token = nullptr;
        if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &token)) {
            return false;
        }
        Y_DEFER {
            ::CloseHandle(token);
        };

        TOKEN_ELEVATION elevation = {};
        DWORD size = sizeof(elevation);
        if (!::GetTokenInformation(token, TokenElevation, &elevation, sizeof(elevation), &size)) {
            return false;
        }
        return elevation.TokenIsElevated != 0;
    }

    // The Developer Mode switch. Returns -1 when the value is not there to read.
    int DeveloperMode() {
        DWORD value = 0;
        DWORD size = sizeof(value);
        const LONG res = ::RegGetValueA(
            HKEY_LOCAL_MACHINE,
            R"(SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock)",
            "AllowDevelopmentWithoutDevLicense",
            RRF_RT_REG_DWORD,
            nullptr,
            &value,
            &size);
        return res == ERROR_SUCCESS ? static_cast<int>(value) : -1;
    }

    // Printed into the test log on purpose: it is the only way to tell from a CI
    // run whether the machine it landed on lets an ordinary process create
    // symlinks, and if not, which of the two reasons is to blame.
    void ReportSymlinkEnvironment(bool created, int error) {
        Cerr << "symlink environment: created=" << created
             << " error=" << error << " (" << LastSystemErrorText(error) << ")"
             << " elevated=" << IsProcessElevated()
             << " developerMode=" << DeveloperMode()
             << Endl;
    }
} // namespace

// Creating a symlink needs SeCreateSymbolicLinkPrivilege, which comes with
// elevation, with Developer Mode, or granted to the account outright. Report
// what the machine offers and leave the test when it offers none of it.
#define SYMLINK_OR_RETURN(target, link)                                \
    do {                                                               \
        const bool created = NFsPrivate::WinSymLink(target, link);     \
        const int error = created ? 0 : LastSystemError();             \
        ReportSymlinkEnvironment(created, error);                      \
        if (!created) {                                                \
            UNIT_ASSERT_VALUES_EQUAL(error, ERROR_PRIVILEGE_NOT_HELD); \
            return;                                                    \
        }                                                              \
        if (!NFsPrivate::WinExists(link) && IsWine()) {                \
            Cerr << "wine does not support symlinks" << Endl;          \
            return;                                                    \
        }                                                              \
    } while (false)

Y_UNIT_TEST_SUITE(TFsWinTest) {
    Y_UNIT_TEST(TestRemoveDirWithROFiles) {
        TFsPath dir1 = "dir1";
        NFs::RemoveRecursive(dir1);
        UNIT_ASSERT(!NFsPrivate::WinExists(dir1));
        UNIT_ASSERT(NFsPrivate::WinMakeDirectory(dir1));

        UNIT_ASSERT(TFileStat(dir1).IsDir());
        TFsPath file1 = dir1 / "file.txt";
        Touch(file1);
        UNIT_ASSERT(NFsPrivate::WinExists(file1));
        SetReadOnly(file1);

        // A read-only file may not be left in the way of a recursive removal:
        // on unix the attribute of the file says nothing about deleting it, and
        // windows is made to behave the same way.
        NFs::RemoveRecursive(dir1);
        UNIT_ASSERT(!NFsPrivate::WinExists(dir1));
    }

    Y_UNIT_TEST(TestRemoveReadOnlyDir) {
        TFsPath dir1 = "dir1";
        NFsPrivate::WinRemove(dir1);
        UNIT_ASSERT(!NFsPrivate::WinExists(dir1));
        UNIT_ASSERT(NFsPrivate::WinMakeDirectory(dir1));

        UNIT_ASSERT(TFileStat(dir1).IsDir());
        SetReadOnly(dir1);

        // Dropping the attribute must not cost the directory its own type:
        // what is removed here has to be removed as a directory.
        UNIT_ASSERT(NFsPrivate::WinRemove(dir1));
        UNIT_ASSERT(!NFsPrivate::WinExists(dir1));
    }

    Y_UNIT_TEST(TestSymLinkToFile) {
        TFsPath target = "symlink_target.txt";
        TFsPath link = "symlink.txt";
        NFsPrivate::WinRemove(link);
        NFsPrivate::WinRemove(target);
        Touch(target);

        SYMLINK_OR_RETURN(static_cast<const TString&>(target), static_cast<const TString&>(link));

        UNIT_ASSERT(TFileStat(link, true).IsSymlink());
        // Following the link lands on a plain file.
        UNIT_ASSERT(!TFileStat(link, false).IsSymlink());
        UNIT_ASSERT(TFileStat(link, false).IsFile());

        // A relative target is stored as given, no spelling of our own.
        UNIT_ASSERT_STRINGS_EQUAL(NFsPrivate::WinReadLink(link), target.GetPath());
        {
            TFile file(link, OpenExisting | RdOnly);
            UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 4);
        }

        // Removing the link leaves the target alone.
        UNIT_ASSERT(NFsPrivate::WinRemove(link));
        UNIT_ASSERT(NFsPrivate::WinExists(target));
        UNIT_ASSERT(NFsPrivate::WinRemove(target));
    }

    Y_UNIT_TEST(TestReadLinkOnAbsoluteSymLink) {
        TFsPath target = "abs_symlink_target.txt";
        TFsPath link = "abs_symlink.txt";
        NFsPrivate::WinRemove(link);
        NFsPrivate::WinRemove(target);
        Touch(target);
        const TString absoluteTarget = target.RealPath().GetPath();

        SYMLINK_OR_RETURN(absoluteTarget, static_cast<const TString&>(link));

        // The reparse point holds the NT spelling of the target - "\??\C:\dir\file" -
        // which names nothing outside the kernel, so a win32 path is handed out.
        UNIT_ASSERT_STRINGS_EQUAL(NFsPrivate::WinReadLink(link), absoluteTarget);

        UNIT_ASSERT(NFsPrivate::WinRemove(link));
        UNIT_ASSERT(NFsPrivate::WinRemove(target));
    }

    Y_UNIT_TEST(TestReadLinkOnJunction) {
        TFsPath target = "junction_target";
        TFsPath junction = "junction";
        NFsPrivate::WinRemove(junction);
        NFsPrivate::WinRemove(target);
        UNIT_ASSERT(NFsPrivate::WinMakeDirectory(target));
        const TString absoluteTarget = target.RealPath().GetPath();

        if (!CreateJunction(junction, absoluteTarget)) {
            Cerr << "can't create junction: "
                 << LastSystemErrorText(LastSystemError()) << Endl;
            UNIT_ASSERT(IsWine());
            return;
        }

        // Needs no privileges, so this is the one check of the NT prefix that
        // holds on any machine.
        UNIT_ASSERT_STRINGS_EQUAL(NFsPrivate::WinReadLink(junction), absoluteTarget);
        // TFileStat calls a mount point a symlink, and WinReadLink reads one.
        UNIT_ASSERT(TFileStat(junction, true).IsSymlink());

        // Removing the junction leaves the directory it points at alone.
        UNIT_ASSERT(NFsPrivate::WinRemove(junction));
        UNIT_ASSERT(NFsPrivate::WinExists(target));
        UNIT_ASSERT(NFsPrivate::WinRemove(target));
    }
} // Y_UNIT_TEST_SUITE(TFsWinTest)
