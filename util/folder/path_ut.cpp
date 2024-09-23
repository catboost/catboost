#include "path.h"
#include "pathsplit.h"
#include "dirut.h"
#include "tempdir.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/scope.h>
#include <util/system/platform.h>
#include <util/system/yassert.h>
#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/system/fs.h>

#include <algorithm>

#ifdef _win_
    #include <aclapi.h>
#endif

namespace {
    /// empty directory for test that needs filesystem
    /// recreates directory in constructor and removes directory in destructor
    class TTestDirectory {
    private:
        TFsPath Path_;

    public:
        TTestDirectory(const TString& name);
        ~TTestDirectory();

        TFsPath GetFsPath() const {
            return Path_;
        }

        TFsPath Child(const TString& name) const {
            return Path_.Child(name);
        }
    };

    TTestDirectory::TTestDirectory(const TString& name) {
        Y_ABORT_UNLESS(name.length() > 0, "have to specify name");
        Y_ABORT_UNLESS(name.find('.') == TString::npos, "must be simple name");
        Y_ABORT_UNLESS(name.find('/') == TString::npos, "must be simple name");
        Y_ABORT_UNLESS(name.find('\\') == TString::npos, "must be simple name");
        Path_ = TFsPath(name);

        Path_.ForceDelete();
        Path_.MkDir();
    }

    TTestDirectory::~TTestDirectory() {
        Path_.ForceDelete();
    }
} // namespace

Y_UNIT_TEST_SUITE(TFsPathTests) {
    Y_UNIT_TEST(TestMkDirs) {
        const TFsPath path = "a/b/c/d/e/f";
        path.ForceDelete();
        TFsPath current = path;
        ui32 checksCounter = 0;
        while (current != ".") {
            UNIT_ASSERT(!path.Exists());
            ++checksCounter;
            current = current.Parent();
        }
        UNIT_ASSERT_VALUES_EQUAL(checksCounter, 6);

        path.MkDirs();
        UNIT_ASSERT(path.Exists());

        current = path;
        while (current != ".") {
            UNIT_ASSERT(path.Exists());
            current = current.Parent();
        }
    }

    Y_UNIT_TEST(MkDirFreak) {
        TFsPath path;
        UNIT_ASSERT_EXCEPTION(path.MkDir(), TIoException);
        UNIT_ASSERT_EXCEPTION(path.MkDirs(), TIoException);
        path = ".";
        path.MkDir();
        path.MkDirs();
    }

    Y_UNIT_TEST(Parent) {
#ifdef _win_
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("\\etc/passwd").Parent(), TFsPath("\\etc"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("\\etc").Parent(), TFsPath("\\"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("\\").Parent(), TFsPath("\\"));

        UNIT_ASSERT_VALUES_EQUAL(TFsPath("etc\\passwd").Parent(), TFsPath("etc"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("etc").Parent(), TFsPath("."));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath(".\\etc").Parent(), TFsPath("."));

        UNIT_ASSERT_VALUES_EQUAL(TFsPath("C:\\etc/passwd").Parent(), TFsPath("C:\\etc"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("C:\\etc").Parent(), TFsPath("C:\\"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("C:\\").Parent(), TFsPath("C:\\"));
#else
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/etc/passwd").Parent(), TFsPath("/etc"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/etc").Parent(), TFsPath("/"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/").Parent(), TFsPath("/"));

        UNIT_ASSERT_VALUES_EQUAL(TFsPath("etc/passwd").Parent(), TFsPath("etc"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("etc").Parent(), TFsPath("."));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("./etc").Parent(), TFsPath("."));
#endif

#if 0
            UNIT_ASSERT_VALUES_EQUAL(TFsPath("./etc/passwd").Parent(), TFsPath("./etc"));
            UNIT_ASSERT_VALUES_EQUAL(TFsPath("./").Parent(), TFsPath(".."));
            UNIT_ASSERT_VALUES_EQUAL(TFsPath(".").Parent(), TFsPath(".."));
            UNIT_ASSERT_VALUES_EQUAL(TFsPath("..").Parent(), TFsPath("../.."));
#endif
    }

    Y_UNIT_TEST(GetName) {
        TTestDirectory d("GetName");
        UNIT_ASSERT_VALUES_EQUAL(TString("dfgh"), d.Child("dfgh").GetName());

        // check does not fail
        TFsPath(".").GetName();

#ifdef _unix_
        UNIT_ASSERT_VALUES_EQUAL(TString("/"), TFsPath("/").GetName());
#endif
    }

    Y_UNIT_TEST(GetExtension) {
        TTestDirectory d("GetExtension");
        UNIT_ASSERT_VALUES_EQUAL("", d.Child("a").GetExtension());
        UNIT_ASSERT_VALUES_EQUAL("", d.Child(".a").GetExtension());
        UNIT_ASSERT_VALUES_EQUAL("", d.Child("zlib").GetExtension());
        UNIT_ASSERT_VALUES_EQUAL("zlib", d.Child("file.zlib").GetExtension());
        UNIT_ASSERT_VALUES_EQUAL("zlib", d.Child("file.ylib.zlib").GetExtension());
    }

    Y_UNIT_TEST(TestRename) {
        TTestDirectory xx("TestRename");
        TFsPath f1 = xx.Child("f1");
        TFsPath f2 = xx.Child("f2");
        f1.Touch();
        f1.RenameTo(f2);
        UNIT_ASSERT(!f1.Exists());
        UNIT_ASSERT(f2.Exists());
    }

    Y_UNIT_TEST(TestForceRename) {
        TTestDirectory xx("TestForceRename");
        TFsPath fMain = xx.Child("main");

        TFsPath f1 = fMain.Child("f1");
        f1.MkDirs();
        TFsPath f1Child = f1.Child("f1child");
        f1Child.Touch();

        TFsPath f2 = fMain.Child("f2");
        f2.MkDirs();

        fMain.ForceRenameTo("TestForceRename/main1");

        UNIT_ASSERT(!xx.Child("main").Exists());
        UNIT_ASSERT(xx.Child("main1").Child("f1").Exists());
        UNIT_ASSERT(xx.Child("main1").Child("f2").Exists());
        UNIT_ASSERT(xx.Child("main1").Child("f1").Child("f1child").Exists());
    }

    Y_UNIT_TEST(TestRenameFail) {
        UNIT_ASSERT_EXCEPTION(TFsPath("sfsfsfsdfsfsdfdf").RenameTo("sdfsdf"), TIoException);
    }

#ifndef _win_
    Y_UNIT_TEST(TestRealPath) {
        UNIT_ASSERT(TFsPath(".").RealPath().IsDirectory());

        TTestDirectory td("TestRealPath");
        TFsPath link = td.Child("link");
        TFsPath target1 = td.Child("target1");
        target1.Touch();
        TFsPath target2 = td.Child("target2");
        target2.Touch();
        UNIT_ASSERT(NFs::SymLink(target1.RealPath(), link.GetPath()));
        UNIT_ASSERT_VALUES_EQUAL(link.RealPath(), target1.RealPath());
        UNIT_ASSERT(NFs::Remove(link.GetPath()));
        UNIT_ASSERT(NFs::SymLink(target2.RealPath(), link.GetPath()));
        UNIT_ASSERT_VALUES_EQUAL(link.RealPath(), target2.RealPath()); // must not cache old value
    }
#endif

    Y_UNIT_TEST(TestSlashesAndBasename) {
        TFsPath p("/db/BASE/primus121-025-1380131338//");
        UNIT_ASSERT_VALUES_EQUAL(p.Basename(), TString("primus121-025-1380131338"));
        TFsPath testP = p / "test";
#ifdef _win_
        UNIT_ASSERT_VALUES_EQUAL(testP.GetPath(), "\\db\\BASE\\primus121-025-1380131338\\test");
#else
        UNIT_ASSERT_VALUES_EQUAL(testP.GetPath(), "/db/BASE/primus121-025-1380131338/test");
#endif
    }

    Y_UNIT_TEST(TestSlashesAndBasenameWin) {
        TFsPath p("\\db\\BASE\\primus121-025-1380131338\\\\");
        TFsPath testP = p / "test";
#ifdef _win_
        UNIT_ASSERT_VALUES_EQUAL(p.Basename(), TString("primus121-025-1380131338"));
        UNIT_ASSERT_VALUES_EQUAL(testP.GetPath(), "\\db\\BASE\\primus121-025-1380131338\\test");
#else
        UNIT_ASSERT_VALUES_EQUAL(p.Basename(), TString("\\db\\BASE\\primus121-025-1380131338\\\\"));
        UNIT_ASSERT_VALUES_EQUAL(testP.GetPath(), "\\db\\BASE\\primus121-025-1380131338\\\\/test");
#endif
    }

    Y_UNIT_TEST(TestSlashesAndBasenameWinDrive) {
        TFsPath p("C:\\db\\BASE\\primus121-025-1380131338\\\\");
        TFsPath testP = p / "test";
#ifdef _win_
        UNIT_ASSERT_VALUES_EQUAL(p.Basename(), TString("primus121-025-1380131338"));
        UNIT_ASSERT_VALUES_EQUAL(testP.GetPath(), "C:\\db\\BASE\\primus121-025-1380131338\\test");
#else
        UNIT_ASSERT_VALUES_EQUAL(p.Basename(), TString("C:\\db\\BASE\\primus121-025-1380131338\\\\"));
        UNIT_ASSERT_VALUES_EQUAL(testP.GetPath(), "C:\\db\\BASE\\primus121-025-1380131338\\\\/test");
#endif
    }

    Y_UNIT_TEST(TestList) {
        TTestDirectory td("TestList-dir");

        TFsPath dir = td.GetFsPath();
        dir.Child("a").Touch();
        dir.Child("b").MkDir();
        dir.Child("b").Child("b-1").Touch();
        dir.Child("c").MkDir();
        dir.Child("d").Touch();

        TVector<TString> children;
        dir.ListNames(children);
        std::sort(children.begin(), children.end());

        TVector<TString> expected;
        expected.push_back("a");
        expected.push_back("b");
        expected.push_back("c");
        expected.push_back("d");

        UNIT_ASSERT_VALUES_EQUAL(expected, children);
    }

#ifdef _unix_
    Y_UNIT_TEST(MkDirMode) {
        TTestDirectory td("MkDirMode");
        TFsPath subDir = td.Child("subdir");
        const int mode = MODE0775;
        subDir.MkDir(mode);
        TFileStat stat;
        UNIT_ASSERT(subDir.Stat(stat));
        // mkdir(2) places umask(2) on mode argument.
        const int mask = Umask(0);
        Umask(mask);
        UNIT_ASSERT_VALUES_EQUAL(stat.Mode & MODE0777, mode & ~mask);
    }
#endif

    Y_UNIT_TEST(Cwd) {
        UNIT_ASSERT_VALUES_EQUAL(TFsPath::Cwd().RealPath(), TFsPath(".").RealPath());
    }

    Y_UNIT_TEST(TestSubpathOf) {
        UNIT_ASSERT(TFsPath("/a/b/c/d").IsSubpathOf("/a/b"));

        UNIT_ASSERT(TFsPath("/a").IsSubpathOf("/"));
        UNIT_ASSERT(!TFsPath("/").IsSubpathOf("/a"));
        UNIT_ASSERT(!TFsPath("/a").IsSubpathOf("/a"));

        UNIT_ASSERT(TFsPath("/a/b").IsSubpathOf("/a"));
        UNIT_ASSERT(TFsPath("a/b").IsSubpathOf("a"));
        UNIT_ASSERT(!TFsPath("/a/b").IsSubpathOf("/b"));
        UNIT_ASSERT(!TFsPath("a/b").IsSubpathOf("b"));

        // mixing absolute/relative
        UNIT_ASSERT(!TFsPath("a").IsSubpathOf("/"));
        UNIT_ASSERT(!TFsPath("a").IsSubpathOf("/a"));
        UNIT_ASSERT(!TFsPath("/a").IsSubpathOf("a"));
        UNIT_ASSERT(!TFsPath("a/b").IsSubpathOf("/a"));
        UNIT_ASSERT(!TFsPath("/a/b").IsSubpathOf("a"));

#ifdef _win_
        UNIT_ASSERT(TFsPath("x:/a/b").IsSubpathOf("x:/a"));
        UNIT_ASSERT(!TFsPath("x:/a/b").IsSubpathOf("y:/a"));
        UNIT_ASSERT(!TFsPath("x:/a/b").IsSubpathOf("a"));
#endif
    }

    Y_UNIT_TEST(TestNonStrictSubpathOf) {
        UNIT_ASSERT(TFsPath("/a/b/c/d").IsNonStrictSubpathOf("/a/b"));

        UNIT_ASSERT(TFsPath("/a").IsNonStrictSubpathOf("/"));
        UNIT_ASSERT(!TFsPath("/").IsNonStrictSubpathOf("/a"));

        UNIT_ASSERT(TFsPath("/a/b").IsNonStrictSubpathOf("/a"));
        UNIT_ASSERT(TFsPath("a/b").IsNonStrictSubpathOf("a"));
        UNIT_ASSERT(!TFsPath("/a/b").IsNonStrictSubpathOf("/b"));
        UNIT_ASSERT(!TFsPath("a/b").IsNonStrictSubpathOf("b"));

        // mixing absolute/relative
        UNIT_ASSERT(!TFsPath("a").IsNonStrictSubpathOf("/"));
        UNIT_ASSERT(!TFsPath("a").IsNonStrictSubpathOf("/a"));
        UNIT_ASSERT(!TFsPath("/a").IsNonStrictSubpathOf("a"));
        UNIT_ASSERT(!TFsPath("a/b").IsNonStrictSubpathOf("/a"));
        UNIT_ASSERT(!TFsPath("/a/b").IsNonStrictSubpathOf("a"));

        // equal paths
        UNIT_ASSERT(TFsPath("").IsNonStrictSubpathOf(""));
        UNIT_ASSERT(TFsPath("/").IsNonStrictSubpathOf("/"));
        UNIT_ASSERT(TFsPath("a").IsNonStrictSubpathOf("a"));
        UNIT_ASSERT(TFsPath("/a").IsNonStrictSubpathOf("/a"));
        UNIT_ASSERT(TFsPath("/a").IsNonStrictSubpathOf("/a/"));
        UNIT_ASSERT(TFsPath("/a/").IsNonStrictSubpathOf("/a"));
        UNIT_ASSERT(TFsPath("/a/").IsNonStrictSubpathOf("/a/"));

#ifdef _win_
        UNIT_ASSERT(TFsPath("x:/a/b").IsNonStrictSubpathOf("x:/a"));

        UNIT_ASSERT(TFsPath("x:/a").IsNonStrictSubpathOf("x:/a"));
        UNIT_ASSERT(TFsPath("x:/a/").IsNonStrictSubpathOf("x:/a"));
        UNIT_ASSERT(TFsPath("x:/a").IsNonStrictSubpathOf("x:/a/"));
        UNIT_ASSERT(TFsPath("x:/a/").IsNonStrictSubpathOf("x:/a/"));

        UNIT_ASSERT(!TFsPath("x:/").IsNonStrictSubpathOf("y:/"));
        UNIT_ASSERT(!TFsPath("x:/a/b").IsNonStrictSubpathOf("y:/a"));
        UNIT_ASSERT(!TFsPath("x:/a/b").IsNonStrictSubpathOf("a"));
#endif
    }

    Y_UNIT_TEST(TestRelativePath) {
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/a/b/c/d").RelativePath(TFsPath("/a/b")), TFsPath("c/d"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/a/b/c/d").RelativePath(TFsPath("/a/b/e/f")), TFsPath("../../c/d"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/").RelativePath(TFsPath("/")), TFsPath());
        UNIT_ASSERT_VALUES_EQUAL(TFsPath(".").RelativePath(TFsPath(".")), TFsPath());
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/a/c").RelativePath(TFsPath("/a/b/../c")), TFsPath());
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("a/.././b").RelativePath(TFsPath("b/c")), TFsPath(".."));

        UNIT_ASSERT_EXCEPTION(TFsPath("a/b/c").RelativePath(TFsPath("d/e")), TIoException);
    }

    Y_UNIT_TEST(TestUndefined) {
        UNIT_ASSERT_VALUES_EQUAL(TFsPath(), TFsPath(""));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath(), TFsPath().Fix());

        UNIT_ASSERT_VALUES_EQUAL(TFsPath() / TFsPath(), TFsPath());
#ifdef _win_
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("a\\b"), TFsPath() / TString("a\\b"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("a\\b"), "a\\b" / TFsPath());
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("\\a\\b"), TFsPath() / "\\a\\b");
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("\\a\\b"), "\\a\\b" / TFsPath());
#else
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("a/b"), TFsPath() / TString("a/b"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("a/b"), "a/b" / TFsPath());
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/a/b"), TFsPath() / "/a/b");
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("/a/b"), "/a/b" / TFsPath());
#endif
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("."), TFsPath() / ".");
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("."), "." / TFsPath());

        UNIT_ASSERT(TFsPath().PathSplit().empty());
        UNIT_ASSERT(!TFsPath().PathSplit().IsAbsolute);
        UNIT_ASSERT(TFsPath().IsRelative()); // undefined path is relative

        UNIT_ASSERT_VALUES_EQUAL(TFsPath().GetPath(), "");
        UNIT_ASSERT_VALUES_EQUAL(TFsPath().GetName(), "");
        UNIT_ASSERT_VALUES_EQUAL(TFsPath().GetExtension(), "");

        UNIT_ASSERT_VALUES_EQUAL(TFsPath().Parent(), TFsPath());
        UNIT_ASSERT_VALUES_EQUAL(TFsPath().Child("a"), TFsPath("a"));
        UNIT_ASSERT_VALUES_EQUAL(TFsPath().Basename(), "");
        UNIT_ASSERT_VALUES_EQUAL(TFsPath().Dirname(), "");

        UNIT_ASSERT(!TFsPath().IsSubpathOf("a/b"));
        UNIT_ASSERT(TFsPath().IsContainerOf("a/b"));
        UNIT_ASSERT(!TFsPath().IsContainerOf("/a/b"));
#ifdef _win_
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("a\\b").RelativeTo(TFsPath()), TFsPath("a\\b"));
#else
        UNIT_ASSERT_VALUES_EQUAL(TFsPath("a/b").RelativeTo(TFsPath()), TFsPath("a/b"));
#endif

        UNIT_ASSERT(!TFsPath().Exists());
        UNIT_ASSERT(!TFsPath().IsFile());
        UNIT_ASSERT(!TFsPath().IsDirectory());
        TFileStat stat;
        UNIT_ASSERT(!TFsPath().Stat(stat));
    }

    Y_UNIT_TEST(TestJoinFsPaths) {
#ifdef _win_
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a\\b", "c\\d"), "a\\b\\c\\d");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a\\b", "..\\c"), "a\\b\\..\\c");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a\\b\\..\\c", "d"), "a\\c\\d");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a", "b", "c", "d"), "a\\b\\c\\d");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a\\b\\..\\c"), "a\\b\\..\\c");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a\\b", ""), "a\\b");
#else
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a/b", "c/d"), "a/b/c/d");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a/b", "../c"), "a/b/../c");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a/b/../c", "d"), "a/c/d");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a", "b", "c", "d"), "a/b/c/d");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a/b/../c"), "a/b/../c");
        UNIT_ASSERT_VALUES_EQUAL(JoinFsPaths("a/b", ""), "a/b");
#endif
    }

    Y_UNIT_TEST(TestStringCast) {
        TFsPath pathOne;
        UNIT_ASSERT(TryFromString<TFsPath>("/a/b", pathOne));
        UNIT_ASSERT_VALUES_EQUAL(pathOne, TFsPath{"/a/b"});

        TFsPath pathTwo;
        UNIT_ASSERT_NO_EXCEPTION(TryFromString<TFsPath>("/a/b", pathTwo));

        UNIT_ASSERT_VALUES_EQUAL(FromString<TFsPath>("/a/b"), TFsPath{"/a/b"});

        TFsPath pathThree{"/a/b"};
        UNIT_ASSERT_VALUES_EQUAL(ToString(pathThree), "/a/b");
    }

#ifdef _unix_
    Y_UNIT_TEST(TestRemoveSymlinkToDir) {
        TTempDir tempDir;
        TFsPath tempDirPath(tempDir());

        const TString originDir = tempDirPath.Child("origin");
        MakePathIfNotExist(originDir.c_str());

        const TString originFile = TFsPath(originDir).Child("data");
        {
            TFixedBufferFileOutput out(originFile);
            out << "data111!!!";
        }

        const TString link = tempDirPath.Child("origin_symlink");
        NFs::SymLink(originDir, link);

        TFsPath(link).ForceDelete();

        UNIT_ASSERT(!NFs::Exists(link));
        UNIT_ASSERT(NFs::Exists(originFile));
        UNIT_ASSERT(NFs::Exists(originDir));
    }

    Y_UNIT_TEST(TestRemoveSymlinkToFile) {
        TTempDir tempDir;
        TFsPath tempDirPath(tempDir());

        const TString originDir = tempDirPath.Child("origin");
        MakePathIfNotExist(originDir.c_str());

        const TString originFile = TFsPath(originDir).Child("data");
        {
            TFixedBufferFileOutput out(originFile);
            out << "data111!!!";
        }

        const TString link = tempDirPath.Child("origin_symlink");
        NFs::SymLink(originFile, link);

        TFsPath(link).ForceDelete();

        UNIT_ASSERT(!NFs::Exists(link));
        UNIT_ASSERT(NFs::Exists(originFile));
        UNIT_ASSERT(NFs::Exists(originDir));
    }

    Y_UNIT_TEST(TestRemoveDirWithSymlinkToDir) {
        TTempDir tempDir;
        TFsPath tempDirPath(tempDir());

        const TString symlinkedDir = tempDirPath.Child("to_remove");
        MakePathIfNotExist(symlinkedDir.c_str());

        const TString originDir = tempDirPath.Child("origin");
        MakePathIfNotExist(originDir.c_str());

        const TString originFile = TFsPath(originDir).Child("data");
        {
            TFixedBufferFileOutput out(originFile);
            out << "data111!!!";
        }

        const TString symlinkedFile = TFsPath(symlinkedDir).Child("origin_symlink");
        NFs::SymLink(originDir, symlinkedFile);

        TFsPath(symlinkedDir).ForceDelete();

        UNIT_ASSERT(!NFs::Exists(symlinkedFile));
        UNIT_ASSERT(!NFs::Exists(symlinkedDir));
        UNIT_ASSERT(NFs::Exists(originFile));
        UNIT_ASSERT(NFs::Exists(originDir));
    }

    Y_UNIT_TEST(TestRemoveDirWithSymlinkToFile) {
        TTempDir tempDir;
        TFsPath tempDirPath(tempDir());

        const TString symlinkedDir = tempDirPath.Child("to_remove");
        MakePathIfNotExist(symlinkedDir.c_str());

        const TString originDir = tempDirPath.Child("origin");
        MakePathIfNotExist(originDir.c_str());

        const TString originFile = TFsPath(originDir).Child("data");
        {
            TFixedBufferFileOutput out(originFile);
            out << "data111!!!";
        }

        const TString symlinkedFile = TFsPath(symlinkedDir).Child("origin_symlink");
        NFs::SymLink(originFile, symlinkedFile);

        TFsPath(symlinkedDir).ForceDelete();

        UNIT_ASSERT(!NFs::Exists(symlinkedFile));
        UNIT_ASSERT(!NFs::Exists(symlinkedDir));
        UNIT_ASSERT(NFs::Exists(originFile));
        UNIT_ASSERT(NFs::Exists(originDir));
    }
#endif

    Y_UNIT_TEST(TestForceDeleteNonexisting) {
        TTempDir tempDir;
        TFsPath nonexisting = TFsPath(tempDir()).Child("nonexisting");
        nonexisting.ForceDelete();
    }

    // Here we want to test that all possible errors during TFsPath::ForceDelete
    // are properly handled. To do so we have to trigger fs operation errors in
    // three points:
    // 1. stat/GetFileInformationByHandle
    // 2. opendir
    // 3. unlink/rmdir
    //
    // On unix systems we can achieve this by simply setting access rights on
    // entry being deleted and its parent. But on windows it is more complicated.
    // Current Chmod implementation on windows is not enough as it sets only
    // FILE_ATTRIBUTE_READONLY throught SetFileAttributes call. But doing so does
    // not affect directory access rights on older versions of Windows and Wine
    // that we use to run autocheck tests.
    //
    // To get required access rights we use DACL in SetSecurityInfo. This is wrapped
    // in RAII class that drops requested permissions on file/dir and grantss them
    // back in destructor.
    //
    // Another obstacle is FILE_LIST_DIRECTORY permission when running on Wine.
    // Dropping this permission is necessary to provoke error
    // in GetFileInformationByHandle. Wine allows dropping this permission, but I
    // have not found a way to grant it back. So tests crash during cleanup sequence.
    // To make it possible to run this tests natively we detect Wine with special
    // registry key and skip these tests only there.

#ifdef _win_
    struct TLocalFree {
        static void Destroy(void* ptr) {
            LocalFree((HLOCAL)ptr);
        }
    };

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

    class TWinFileDenyAccessScope {
    public:
        TWinFileDenyAccessScope(const TFsPath& name, DWORD permissions)
            : Name_(name)
            , Perms_(permissions)
        {
            DWORD res = 0;
            PACL oldAcl = nullptr;
            PSECURITY_DESCRIPTOR sd = nullptr;

            res = GetNamedSecurityInfoA((LPSTR)name.c_str(),
                                        SE_FILE_OBJECT,
                                        DACL_SECURITY_INFORMATION,
                                        nullptr,
                                        nullptr,
                                        &oldAcl,
                                        nullptr,
                                        &sd);
            SdHolder_.Reset(sd);
            if (res != ERROR_SUCCESS) {
                ythrow TSystemError(res) << "error in GetNamedSecurityInfoA";
            }

            Acl_ = SetAcl(oldAcl, DENY_ACCESS);
        }

        ~TWinFileDenyAccessScope() {
            try {
                const TFsPath parent = Name_.Parent();
                Chmod(parent.c_str(), MODE0777);
                Chmod(Name_.c_str(), MODE0777);
                SetAcl((PACL)Acl_.Get(), GRANT_ACCESS);
            } catch (const yexception& ex) {
                Cerr << "~TWinFileDenyAccessScope failed: " << ex.AsStrBuf() << Endl;
            }
        }

        THolder<void, TLocalFree> SetAcl(PACL oldAcl, ACCESS_MODE accessMode) {
            DWORD res = 0;
            EXPLICIT_ACCESS ea;
            PACL newAcl = nullptr;
            THolder<void, TLocalFree> newAclHolder;

            memset(&ea, 0, sizeof(EXPLICIT_ACCESS));
            ea.grfAccessPermissions = Perms_;
            ea.grfAccessMode = accessMode;
            ea.grfInheritance = NO_INHERITANCE;
            ea.Trustee.TrusteeForm = TRUSTEE_IS_NAME;
            ea.Trustee.ptstrName = (LPSTR) "CURRENT_USER";

            res = SetEntriesInAcl(1, &ea, oldAcl, &newAcl);
            newAclHolder.Reset(newAcl);
            if (res != ERROR_SUCCESS) {
                ythrow TSystemError(res) << "error in SetEntriesInAcl";
            }

            res = SetNamedSecurityInfoA((LPSTR)Name_.c_str(),
                                        SE_FILE_OBJECT,
                                        DACL_SECURITY_INFORMATION,
                                        nullptr,
                                        nullptr,
                                        newAcl,
                                        nullptr);
            if (res != ERROR_SUCCESS) {
                ythrow TSystemError(res) << "error in SetNamedSecurityInfoA";
            }

            return std::move(newAclHolder);
        }

    private:
        const TFsPath Name_;
        const DWORD Perms_;
        THolder<void, TLocalFree> SdHolder_;
        THolder<void, TLocalFree> Acl_;
    };
#endif

    Y_UNIT_TEST(TestForceDeleteErrorRemove) {
        TTempDir tempDir;

        const TFsPath testDir = TFsPath(tempDir()).Child("dir");
        MakePathIfNotExist(testDir.c_str());

        const TFsPath testFile = testDir.Child("file");
        {
            TFixedBufferFileOutput out(testFile);
            out << "data111!!!";
        }

#ifdef _win_
        Chmod(testFile.c_str(), S_IRUSR);
        Y_DEFER {
            Chmod(testFile.c_str(), MODE0777);
        };
        // Checks that dir/file with readonly attribute will be deleted
        // on Windows
        UNIT_ASSERT_NO_EXCEPTION(testFile.ForceDelete());
#else
        Chmod(testDir.c_str(), S_IRUSR | S_IXUSR);
        Y_DEFER {
            Chmod(testDir.c_str(), MODE0777);
        };
        UNIT_ASSERT_EXCEPTION_CONTAINS(testFile.ForceDelete(), TIoException,
                                       "failed to delete");
#endif
    }

    Y_UNIT_TEST(TestForceDeleteErrorRmdir) {
        TTempDir tempDir;

        const TFsPath testDir = TFsPath(tempDir()).Child("dir");
        const TFsPath testSubdir = testDir.Child("file");
        MakePathIfNotExist(testSubdir.c_str());

#ifdef _win_
        Chmod(testSubdir.c_str(), 0);
        Y_DEFER {
            Chmod(testSubdir.c_str(), MODE0777);
        };
        TWinFileDenyAccessScope dirAcl(testDir, FILE_WRITE_DATA);
#else
        Chmod(testDir.c_str(), S_IRUSR | S_IXUSR);
        Y_DEFER {
            Chmod(testDir.c_str(), MODE0777);
        };
#endif

        UNIT_ASSERT_EXCEPTION_CONTAINS(testSubdir.ForceDelete(), TIoException, "failed to delete");
    }

    Y_UNIT_TEST(TestForceDeleteErrorStatDir) {
        TTempDir tempDir;

        const TFsPath testDir = TFsPath(tempDir()).Child("dir");
        const TFsPath testSubdir = testDir.Child("file");
        MakePathIfNotExist(testSubdir.c_str());

#ifdef _win_
        if (IsWine()) {
            // FILE_LIST_DIRECTORY seem to be irreversible on wine
            return;
        }
        TWinFileDenyAccessScope subdirAcl(testSubdir, FILE_READ_ATTRIBUTES);
        TWinFileDenyAccessScope dirAcl(testDir, FILE_LIST_DIRECTORY);
#else
        Chmod(testDir.c_str(), 0);
        Y_DEFER {
            Chmod(testDir.c_str(), MODE0777);
        };
#endif

        UNIT_ASSERT_EXCEPTION_CONTAINS(testSubdir.ForceDelete(), TIoException, "failed to stat");
    }

    Y_UNIT_TEST(TestForceDeleteErrorStatFile) {
        TTempDir tempDir;

        const TFsPath testDir = TFsPath(tempDir()).Child("dir");
        MakePathIfNotExist(testDir.c_str());

        const TFsPath testFile = testDir.Child("file");
        {
            TFixedBufferFileOutput out(testFile);
            out << "data111!!!";
        }

#ifdef _win_
        if (IsWine()) {
            // FILE_LIST_DIRECTORY seem to be irreversible on wine
            return;
        }
        TWinFileDenyAccessScope fileAcl(testFile, FILE_READ_ATTRIBUTES);
        TWinFileDenyAccessScope dirAcl(testDir, FILE_LIST_DIRECTORY);
#else
        Chmod(testDir.c_str(), 0);
        Y_DEFER {
            Chmod(testDir.c_str(), MODE0777);
        };
#endif

        UNIT_ASSERT_EXCEPTION_CONTAINS(testFile.ForceDelete(), TIoException, "failed to stat");
    }

    Y_UNIT_TEST(TestForceDeleteErrorListDir) {
        TTempDir tempDir;

        const TFsPath testDir = TFsPath(tempDir()).Child("dir");
        const TFsPath testSubdir = testDir.Child("file");
        MakePathIfNotExist(testSubdir.c_str());

#ifdef _win_
        if (IsWine()) {
            // FILE_LIST_DIRECTORY seem to be irreversible on wine
            return;
        }
        TWinFileDenyAccessScope subdirAcl(testSubdir, FILE_LIST_DIRECTORY);
#else
        Chmod(testSubdir.c_str(), 0);
        Y_DEFER {
            Chmod(testSubdir.c_str(), MODE0777);
        };
#endif

        UNIT_ASSERT_EXCEPTION_CONTAINS(testSubdir.ForceDelete(), TIoException, "failed to opendir");
    }

#ifdef _unix_
    Y_UNIT_TEST(TestForceDeleteErrorSymlink) {
        TTempDir tempDir;

        const TFsPath testDir = TFsPath(tempDir()).Child("dir");
        MakePathIfNotExist(testDir.c_str());

        const TFsPath testSymlink = testDir.Child("symlink");
        NFs::SymLink("something", testSymlink);

        Chmod(testSymlink.c_str(), S_IRUSR);
        Chmod(testDir.c_str(), S_IRUSR | S_IXUSR);
        Y_DEFER {
            Chmod(testDir.c_str(), MODE0777);
            Chmod(testSymlink.c_str(), MODE0777);
        };

        UNIT_ASSERT_EXCEPTION_CONTAINS(testSymlink.ForceDelete(), TIoException, "failed to delete");
    }
#endif

    Y_UNIT_TEST(TestCopyWithInitializedSplit) {
        const TFsPath path1 = TFsPath("some_folder_with_file") / TFsPath("file_in_folder");
        path1.PathSplit();

        const TFsPath path2 = path1;
        const TPathSplit& split2 = path2.PathSplit();

        for (const auto& it : split2) {
            UNIT_ASSERT(path2.GetPath().begin() <= it.begin());
            UNIT_ASSERT(it.end() <= path2.GetPath().end());
        }
    }

    Y_UNIT_TEST(TestAssignmentWithInitializedSplit) {
        TFsPath path1 = TFsPath("some_folder_with_file_1") / TFsPath("file_in_folder_1");
        TFsPath path2 = TFsPath("some_folder_with_file_2") / TFsPath("file_in_folder_2");
        path1.PathSplit();
        path1 = path2;
        UNIT_ASSERT_VALUES_EQUAL(path1.PathSplit().at(1), "file_in_folder_2");
    }

#ifdef TSTRING_IS_STD_STRING
    Y_UNIT_TEST(TestCopySplitSSO) {
        // Summary length of path must be less minimal SSO length 19 bytes
        constexpr TStringBuf A("a");
        constexpr TStringBuf B("b");
        constexpr TStringBuf C("c");
        for (auto constructorType = 0; constructorType < 2; ++constructorType) {
            TFsPath path1 = TFsPath(A) / TFsPath(B);
            const auto& split1 = path1.PathSplit();
            // Check split of path1
            UNIT_ASSERT_VALUES_EQUAL(split1.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(split1.at(0), A);
            UNIT_ASSERT_VALUES_EQUAL(split1.at(1), B);
            TFsPath path2;
            if (constructorType == 0) {            // copy
                path2 = TFsPath(path1);            // copy constructor
            } else if (constructorType == 1) {     // move
                path2 = TFsPath(std::move(path1)); // move constructor
            }
            const auto& split2 = path2.PathSplit();
            path1 = TFsPath(C); // invalidate previous Path_ in path1
            const auto& newsplit1 = path1.PathSplit();
            // Check that split of path1 was overwrited (invalidate previous TStringBuf)
            UNIT_ASSERT_VALUES_EQUAL(newsplit1.size(), 1);
            UNIT_ASSERT_VALUES_EQUAL(newsplit1.at(0), C);
            // Check split of path2 without segfault
            UNIT_ASSERT_VALUES_EQUAL(split2.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(split2.at(0), A);
            UNIT_ASSERT_VALUES_EQUAL(split2.at(1), B);
        }
    }
#endif

    Y_UNIT_TEST(TestCopySplitNoneSSO) {
        // Lenght of directory name must overhead SSO length 19-23 bytes
        const TString DIR_A = TString("Dir") + TString(32, 'A');
        const TString DIR_B = TString("Dir") + TString(64, 'B');
        const TString DIR_C = TString("Dir") + TString(128, 'C');
        for (auto constructorType = 0; constructorType < 2; ++constructorType) {
            TFsPath path1 = TFsPath(DIR_A) / TFsPath(DIR_B);
            auto& split1 = path1.PathSplit();
            // Check split of path1
            UNIT_ASSERT_VALUES_EQUAL(split1.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(split1.at(0), DIR_A);
            UNIT_ASSERT_VALUES_EQUAL(split1.at(1), DIR_B);
            TFsPath path2;
            if (constructorType == 0) {            // copy
                path2 = TFsPath(path1);            // copy constructor
            } else if (constructorType == 1) {     // move
                path2 = TFsPath(std::move(path1)); // move constructor
            }
            const auto& split2 = path2.PathSplit();
            path1 = TFsPath(DIR_C); // invalidate previous Path_ in path1
            const auto& newsplit1 = path1.PathSplit();
            // Check that split of path1 was overwrited (invalidate previous TStringBuf)
            UNIT_ASSERT_VALUES_EQUAL(newsplit1.size(), 1);
            UNIT_ASSERT_VALUES_EQUAL(newsplit1.at(0), DIR_C);
            // Check split of path2 without segfault
            UNIT_ASSERT_VALUES_EQUAL(split2.size(), 2);
            UNIT_ASSERT_VALUES_EQUAL(split2.at(0), DIR_A);
            UNIT_ASSERT_VALUES_EQUAL(split2.at(1), DIR_B);
        }
    }
} // Y_UNIT_TEST_SUITE(TFsPathTests)
