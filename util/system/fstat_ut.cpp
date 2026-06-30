#include "fstat.h"
#include "file.h"
#include "sysstat.h"
#include "fs.h"

#include <library/cpp/testing/unittest/registar.h>
#include <library/cpp/testing/unittest/tests_data.h>

#include <util/folder/path.h>

Y_UNIT_TEST_SUITE(TestFileStat) {
    Y_UNIT_TEST(FileTest) {
        TString fileName = "f1.txt";
        TFileStat oFs;
        {
            TFile file(fileName.data(), OpenAlways | WrOnly);
            file.Write("1234567", 7);

            {
                TFileStat fs(file);
                UNIT_ASSERT(fs.IsFile());
                UNIT_ASSERT(!fs.IsDir());
                UNIT_ASSERT(!fs.IsSymlink());
                UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), (i64)fs.Size);
                UNIT_ASSERT(fs.MTime >= fs.CTime);
                UNIT_ASSERT(fs.NLinks == 1);
                oFs = fs;
            }

            UNIT_ASSERT(file.IsOpen());
            UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 7);
            file.Close();
        }
        TFileStat cFs(fileName);
        UNIT_ASSERT(cFs.IsFile());
        UNIT_ASSERT(!cFs.IsDir());
        UNIT_ASSERT(!cFs.IsSymlink());
        UNIT_ASSERT_VALUES_EQUAL(cFs.Size, oFs.Size);
        UNIT_ASSERT(cFs.MTime >= oFs.MTime);
        UNIT_ASSERT_VALUES_EQUAL(cFs.CTime, oFs.CTime);
        UNIT_ASSERT_VALUES_EQUAL(cFs.NLinks, oFs.NLinks);
        UNIT_ASSERT_VALUES_EQUAL(cFs.Mode, oFs.Mode);
        UNIT_ASSERT_VALUES_EQUAL(cFs.Uid, oFs.Uid);
        UNIT_ASSERT_VALUES_EQUAL(cFs.Gid, oFs.Gid);
        UNIT_ASSERT_VALUES_EQUAL(cFs.INode, oFs.INode);
        UNIT_ASSERT(unlink(fileName.data()) == 0);
    }

    Y_UNIT_TEST(DirTest) {
        Mkdir("tmpd", MODE0777);
        TFileStat fs("tmpd");
        UNIT_ASSERT(!fs.IsFile());
        UNIT_ASSERT(fs.IsDir());
        UNIT_ASSERT(!fs.IsSymlink());
        // UNIT_ASSERT(fs.Size == 0); // it fails under unix
        UNIT_ASSERT(NFs::Remove("tmpd"));
        fs = TFileStat("tmpd");
        UNIT_ASSERT(!fs.IsFile());
        UNIT_ASSERT(!fs.IsDir());
        UNIT_ASSERT(!fs.IsSymlink());
        UNIT_ASSERT(fs.Size == 0);
        UNIT_ASSERT(fs.CTime == 0);
    }

#ifdef _win_
    // Symlinks require additional privileges on windows.
    // Skip test if we are not allowed to create one.
    // Wine returns true from NFs::SymLink, but actually does nothing
    #define SAFE_SYMLINK(target, link)                                                \
        do {                                                                          \
            auto res = NFs::SymLink(target, link);                                    \
            if (!res) {                                                               \
                auto err = LastSystemError();                                         \
                Cerr << "can't create symlink: " << LastSystemErrorText(err) << Endl; \
                UNIT_ASSERT(err == ERROR_PRIVILEGE_NOT_HELD);                         \
                return;                                                               \
            }                                                                         \
            if (!NFs::Exists(link) && IsWine()) {                                     \
                Cerr << "wine does not support symlinks" << Endl;                     \
                return;                                                               \
            }                                                                         \
        } while (false)

    bool IsWine() {
        HKEY subKey = nullptr;
        LONG result = RegOpenKeyEx(HKEY_CURRENT_USER, "Software\\Wine", 0, KEY_READ, &subKey);
        if (result == ERROR_SUCCESS) {
            return true;
        }
        result = RegOpenKeyEx(HKEY_LOCAL_MACHINE, "Software\\Wine", 0, KEY_READ, &subKey);
        if (result == ERROR_SUCCESS) {
            return true;
        }

        HMODULE hntdll = GetModuleHandle("ntdll.dll");
        if (!hntdll) {
            return false;
        }

        auto func = GetProcAddress(hntdll, "wine_get_version");
        return func != nullptr;
    }
#else
    #define SAFE_SYMLINK(target, link) UNIT_ASSERT(NFs::SymLink(target, link))
#endif

    Y_UNIT_TEST(SymlinkToExistingFileTest) {
        const auto path = GetOutputPath() / "file_1";
        const auto link = GetOutputPath() / "symlink_1";
        TFile(path, EOpenModeFlag::CreateNew | EOpenModeFlag::RdWr);
        SAFE_SYMLINK(path, link);

        const TFileStat statNoFollow(link, false);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(link, true);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

    Y_UNIT_TEST(SymlinkToNonExistingFileTest) {
        const auto path = GetOutputPath() / "file_2";
        const auto link = GetOutputPath() / "symlink_2";
        SAFE_SYMLINK(path, link);

        const TFileStat statNoFollow(link, false);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(link, true);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

    Y_UNIT_TEST(SymlinkToFileThatCantExistTest) {
        const auto path = TFsPath("/path") / "that" / "does" / "not" / "exists";
        const auto link = GetOutputPath() / "symlink_3";
        SAFE_SYMLINK(path, link);

        const TFileStat statNoFollow(link, false);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(link, true);
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

    Y_UNIT_TEST(FileDoesNotExistTest) {
        const auto path = TFsPath("/path") / "that" / "does" / "not" / "exists";

        const TFileStat statNoFollow(path, false);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statNoFollow.IsNull(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsFile(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsSymlink(), ToString(statNoFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statNoFollow.IsDir(), ToString(statNoFollow.Mode));

        const TFileStat statFollow(path, true);
        UNIT_ASSERT_VALUES_EQUAL_C(true, statFollow.IsNull(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsFile(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsSymlink(), ToString(statFollow.Mode));
        UNIT_ASSERT_VALUES_EQUAL_C(false, statFollow.IsDir(), ToString(statFollow.Mode));
    }

    Y_UNIT_TEST(ChmodTest) {
        const TString fileName = "m.txt";
        TFile file(fileName.c_str(), OpenAlways | WrOnly);
        file.Write("1", 1);
        file.Close();

        const TFileStat statDefault(fileName);
        UNIT_ASSERT(Chmod(fileName.c_str(), statDefault.Mode) == 0);
        const TFileStat statUnchanged(fileName);
        UNIT_ASSERT_VALUES_EQUAL(statDefault.Mode, statUnchanged.Mode);

        UNIT_ASSERT(Chmod(fileName.c_str(), S_IRUSR | S_IRGRP | S_IROTH) == 0);
        const TFileStat statReadOnly(fileName);
        UNIT_ASSERT_VALUES_UNEQUAL(statDefault.Mode, statReadOnly.Mode);
        UNIT_ASSERT(Chmod(fileName.c_str(), statReadOnly.Mode) == 0);
        UNIT_ASSERT_VALUES_EQUAL(statReadOnly.Mode, TFileStat(fileName).Mode);

        UNIT_ASSERT(Chmod(fileName.c_str(), statDefault.Mode) == 0);
        UNIT_ASSERT(unlink(fileName.c_str()) == 0);
    }

#ifdef _win_
    Y_UNIT_TEST(WinArchiveDirectoryTest) {
        TFsPath dir = "archive_dir";
        dir.MkDirs();

        SetFileAttributesA(dir.c_str(), FILE_ATTRIBUTE_ARCHIVE);

        const TFileStat stat(dir);
        UNIT_ASSERT(stat.IsDir());
        UNIT_ASSERT(!stat.IsSymlink());
        UNIT_ASSERT(!stat.IsFile());
        UNIT_ASSERT(!stat.IsNull());
    }

    Y_UNIT_TEST(WinArchiveFileTest) {
        TFsPath filename = "archive_file";
        TFile file(filename, OpenAlways | WrOnly);
        file.Write("1", 1);
        file.Close();

        SetFileAttributesA(filename.c_str(), FILE_ATTRIBUTE_ARCHIVE);

        const TFileStat stat(filename);
        UNIT_ASSERT(!stat.IsDir());
        UNIT_ASSERT(!stat.IsSymlink());
        UNIT_ASSERT(stat.IsFile());
        UNIT_ASSERT(!stat.IsNull());
    }
#endif
} // Y_UNIT_TEST_SUITE(TestFileStat)
