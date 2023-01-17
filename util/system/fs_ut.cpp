#include "fs.h"

#include <library/cpp/testing/unittest/registar.h>

#include "file.h"
#include "fstat.h"
#include <util/folder/path.h>

//WARNING: on windows the test must be run with administative rules

class TFsTest: public TTestBase {
    UNIT_TEST_SUITE(TFsTest);
    UNIT_TEST(TestCreateRemove);
    UNIT_TEST(TestRename);
    UNIT_TEST(TestSymlink);
    UNIT_TEST(TestHardlink);
    UNIT_TEST(TestCwdOpts);
    UNIT_TEST(TestEnsureExists);
    UNIT_TEST_SUITE_END();

public:
    void TestCreateRemove();
    void TestRename();
    void TestSymlink();
    void TestHardlink();
    void TestCwdOpts();
    void TestEnsureExists();
};

UNIT_TEST_SUITE_REGISTRATION(TFsTest);

static void Touch(const TFsPath& path) {
    TFile file(path, CreateAlways | WrOnly);
    file.Write("123", 3);
}

void TFsTest::TestCreateRemove() {
    TFsPath dir1 = "dir_aбвг";
    NFs::RemoveRecursive(dir1);
    UNIT_ASSERT(!NFs::Exists(dir1));
    UNIT_ASSERT(NFs::MakeDirectory(dir1));

    UNIT_ASSERT(TFileStat(dir1).IsDir());
    UNIT_ASSERT(!NFs::MakeDirectory(dir1));

    UNIT_ASSERT(NFs::Exists(dir1));
    TFsPath subdir1 = dir1 / "a" / "b";
    //TFsPath link = dir1 / "link";

    UNIT_ASSERT(NFs::MakeDirectoryRecursive(subdir1, NFs::FP_COMMON_FILE, true));
    UNIT_ASSERT(NFs::Exists(subdir1));
    UNIT_ASSERT(NFs::MakeDirectoryRecursive(subdir1, NFs::FP_COMMON_FILE, false));
    UNIT_ASSERT(NFs::MakeDirectoryRecursive(subdir1, NFs::FP_COMMON_FILE));
    UNIT_ASSERT_EXCEPTION(NFs::MakeDirectoryRecursive(subdir1, NFs::FP_COMMON_FILE, true), TIoException);

    TFsPath file1 = dir1 / "f1.txt";
    TFsPath file2 = subdir1 + TString("_f2.txt");
    TFsPath file3 = subdir1 / "f2.txt";
    Touch(file1);
    Touch(file2);
    Touch(file3);
    //UNIT_ASSERT(NFs::SymLink(file3.RealPath(), link));

    UNIT_ASSERT(NFs::MakeDirectoryRecursive(dir1 / "subdir1" / "subdir2" / "subdir3" / "subdir4", NFs::FP_COMMON_FILE, false));
    UNIT_ASSERT(NFs::MakeDirectoryRecursive(dir1 / "subdir1" / "subdir2", NFs::FP_COMMON_FILE, false));

    // the target path is a file or "subdirectory" of a file
    UNIT_ASSERT(!NFs::MakeDirectoryRecursive(file1 / "subdir1" / "subdir2", NFs::FP_COMMON_FILE, false));
    UNIT_ASSERT(!NFs::MakeDirectoryRecursive(file1, NFs::FP_COMMON_FILE, false));

    TString longUtf8Name = "";
    while (longUtf8Name.size() < 255) {
        longUtf8Name = longUtf8Name + "fф";
    }
    UNIT_ASSERT_EQUAL(longUtf8Name.size(), 255);
    TFsPath longfile = dir1 / longUtf8Name;
    Touch(longfile);

    UNIT_ASSERT(NFs::Exists(longfile));
    UNIT_ASSERT(NFs::Exists(file1));
    UNIT_ASSERT(NFs::Exists(file2));
    UNIT_ASSERT(NFs::Exists(file3));
    //UNIT_ASSERT(NFs::Exists(link));

    UNIT_ASSERT(!NFs::Remove(dir1));
    NFs::RemoveRecursive(dir1);

    UNIT_ASSERT(!NFs::Exists(file1));
    UNIT_ASSERT(!NFs::Exists(file2));
    UNIT_ASSERT(!NFs::Exists(file3));
    //UNIT_ASSERT(!NFs::Exists(link));
    UNIT_ASSERT(!NFs::Exists(subdir1));
    UNIT_ASSERT(!NFs::Exists(longfile));
    UNIT_ASSERT(!NFs::Exists(dir1));
}

void RunRenameTest(TFsPath src, TFsPath dst) {
    // if previous running was failed
    TFsPath dir1 = "dir";
    TFsPath dir2 = "dst_dir";

    NFs::Remove(src);
    NFs::Remove(dst);

    NFs::Remove(dir1 / src);
    NFs::Remove(dir1);
    NFs::Remove(dir2 / src);
    NFs::Remove(dir2);

    {
        TFile file(src, CreateNew | WrOnly);
        file.Write("123", 3);
    }

    UNIT_ASSERT(NFs::Rename(src, dst));
    UNIT_ASSERT(NFs::Exists(dst));
    UNIT_ASSERT(!NFs::Exists(src));

    {
        TFile file(dst, OpenExisting);
        UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 3);
    }

    NFs::MakeDirectory(dir1);
    {
        TFile file(dir1 / src, CreateNew | WrOnly);
        file.Write("123", 3);
    }
    UNIT_ASSERT(NFs::Rename(dir1, dir2));
    UNIT_ASSERT(NFs::Exists(dir2 / src));
    UNIT_ASSERT(!NFs::Exists(dir1));

    {
        TFile file(dir2 / src, OpenExisting);
        UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 3);
    }

    UNIT_ASSERT(!NFs::Remove(src));
    UNIT_ASSERT(NFs::Remove(dst));
    UNIT_ASSERT(!NFs::Remove(dir1));
    UNIT_ASSERT(NFs::Remove(dir2 / src));
    UNIT_ASSERT(NFs::Remove(dir2));
}

void TFsTest::TestRename() {
    RunRenameTest("src.txt", "dst.txt");
    RunRenameTest("src_абвг.txt", "dst_абвг.txt");
}

static void RunHardlinkTest(const TFsPath& src, const TFsPath& dst) {
    NFs::Remove(src);
    NFs::Remove(dst);

    {
        TFile file(src, CreateNew | WrOnly);
        file.Write("123", 3);
    }

    UNIT_ASSERT(NFs::HardLink(src, dst));

    {
        TFile file(dst, OpenExisting | RdOnly);
        UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 3);
    }
    {
        TFile file(src, OpenExisting | WrOnly);
        file.Write("1234", 4);
    }
    {
        TFile file(dst, OpenExisting | RdOnly);
        UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 4);
    }
    {
        TFile file(dst, OpenExisting | WrOnly);
        file.Write("12345", 5);
    }

    {
        TFile file(src, OpenExisting | RdOnly);
        UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 5);
    }

    UNIT_ASSERT(NFs::Remove(dst));
    UNIT_ASSERT(NFs::Remove(src));
}

void TFsTest::TestHardlink() {
    RunHardlinkTest("tempfile", "hardlinkfile");
    RunHardlinkTest("tempfile_абвг", "hardlinkfile_абвг"); //utf-8 names
}

static void RunSymLinkTest(TString fileLocalName, TString symLinkName) {
    // if previous running was failed
    TFsPath subDir = "tempsubdir";
    TFsPath srcFile = subDir / fileLocalName;

    TFsPath subsubDir1 = subDir / "dir1";
    TFsPath subsubDir2 = subDir / "dir2";

    TFsPath linkD1 = "symlinkdir";
    TFsPath linkD2 = subsubDir1 / "linkd2";
    TFsPath dangling = subsubDir1 / "dangling";

    NFs::Remove(srcFile);
    NFs::Remove(symLinkName);
    NFs::Remove(linkD2);
    NFs::Remove(dangling);
    NFs::Remove(subsubDir1);
    NFs::Remove(subsubDir2);
    NFs::Remove(subDir);
    NFs::Remove(linkD1);

    NFs::MakeDirectory(subDir);
    NFs::MakeDirectory(subsubDir1, NFs::FP_NONSECRET_FILE);
    NFs::MakeDirectory(subsubDir2, NFs::FP_SECRET_FILE);
    {
        TFile file(srcFile, CreateNew | WrOnly);
        file.Write("1234567", 7);
    }
    UNIT_ASSERT(NFs::SymLink(subDir, linkD1));
    UNIT_ASSERT(NFs::SymLink("../dir2", linkD2));
    UNIT_ASSERT(NFs::SymLink("../dir3", dangling));
    UNIT_ASSERT_STRINGS_EQUAL(NFs::ReadLink(linkD2), TString("..") + LOCSLASH_S "dir2");
    UNIT_ASSERT_STRINGS_EQUAL(NFs::ReadLink(dangling), TString("..") + LOCSLASH_S "dir3");
    {
        TFile file(linkD1 / fileLocalName, OpenExisting | RdOnly);
        UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 7);
    }
    UNIT_ASSERT(NFs::SymLink(srcFile, symLinkName));
    {
        TFile file(symLinkName, OpenExisting | RdOnly);
        UNIT_ASSERT_VALUES_EQUAL(file.GetLength(), 7);
    }
    {
        TFileStat fs(linkD1);
        UNIT_ASSERT(!fs.IsFile());
        UNIT_ASSERT(fs.IsDir());
        UNIT_ASSERT(!fs.IsSymlink());
    }
    {
        TFileStat fs(linkD1, true);
        UNIT_ASSERT(!fs.IsFile());
        //UNIT_ASSERT(fs.IsDir()); // failed on unix
        UNIT_ASSERT(fs.IsSymlink());
    }
    {
        TFileStat fs(symLinkName);
        UNIT_ASSERT(fs.IsFile());
        UNIT_ASSERT(!fs.IsDir());
        UNIT_ASSERT(!fs.IsSymlink());
        UNIT_ASSERT_VALUES_EQUAL(fs.Size, 7u);
    }

    {
        TFileStat fs(symLinkName, true);
        //UNIT_ASSERT(fs.IsFile()); // no evidence that symlink has to be a file as well
        UNIT_ASSERT(!fs.IsDir());
        UNIT_ASSERT(fs.IsSymlink());
    }

    UNIT_ASSERT(NFs::Remove(symLinkName));
    UNIT_ASSERT(NFs::Exists(srcFile));

    UNIT_ASSERT(NFs::Remove(linkD1));
    UNIT_ASSERT(NFs::Exists(srcFile));

    UNIT_ASSERT(!NFs::Remove(subDir));

    UNIT_ASSERT(NFs::Remove(srcFile));
    UNIT_ASSERT(NFs::Remove(linkD2));
    UNIT_ASSERT(NFs::Remove(dangling));
    UNIT_ASSERT(NFs::Remove(subsubDir1));
    UNIT_ASSERT(NFs::Remove(subsubDir2));
    UNIT_ASSERT(NFs::Remove(subDir));
}

void TFsTest::TestSymlink() {
    // if previous running was failed
    RunSymLinkTest("f.txt", "fl.txt");
    RunSymLinkTest("f_абвг.txt", "fl_абвг.txt"); //utf-8 names
}

void TFsTest::TestCwdOpts() {
    TFsPath initialCwd = NFs::CurrentWorkingDirectory();
    TFsPath subdir = "dir_forcwd_абвг";
    NFs::MakeDirectory(subdir, NFs::FP_SECRET_FILE | NFs::FP_ALL_READ);
    NFs::SetCurrentWorkingDirectory(subdir);
    TFsPath newCwd = NFs::CurrentWorkingDirectory();

    UNIT_ASSERT_EQUAL(newCwd.Fix(), (initialCwd / subdir).Fix());

    NFs::SetCurrentWorkingDirectory("..");
    TFsPath newCwd2 = NFs::CurrentWorkingDirectory();
    UNIT_ASSERT_EQUAL(newCwd2.Fix(), initialCwd.Fix());
    UNIT_ASSERT(NFs::Remove(subdir));
}

void TFsTest::TestEnsureExists() {
    TFsPath fileExists = "tmp_file_абвг.txt";
    TFsPath nonExists = "tmp2_file_абвг.txt";
    {
        NFs::Remove(fileExists);
        NFs::Remove(nonExists);
        TFile file(fileExists, CreateNew | WrOnly);
        file.Write("1234567", 7);
    }

    UNIT_ASSERT_NO_EXCEPTION(NFs::EnsureExists(fileExists));
    UNIT_ASSERT_EXCEPTION(NFs::EnsureExists(nonExists), TFileError);

    TStringBuilder expected;
    TString got;
    try {
        NFs::EnsureExists(nonExists);
        expected << __LOCATION__;
    } catch (const TFileError& err) {
        got = err.what();
    }
    UNIT_ASSERT(got.Contains(expected));
    UNIT_ASSERT(got.Contains(NFs::CurrentWorkingDirectory()));

    UNIT_ASSERT(NFs::Remove(fileExists));
}
