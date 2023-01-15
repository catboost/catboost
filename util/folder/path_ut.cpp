#include "path.h"
#include "pathsplit.h"
#include "dirut.h"
#include "tempdir.h"

#include <library/cpp/unittest/registar.h>

#include <util/system/platform.h>
#include <util/system/yassert.h>
#include <util/stream/output.h>
#include <util/stream/file.h>
#include <util/system/fs.h>

#include <algorithm>

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
        Y_VERIFY(name.length() > 0, "have to specify name");
        Y_VERIFY(name.find('.') == TString::npos, "must be simple name");
        Y_VERIFY(name.find('/') == TString::npos, "must be simple name");
        Y_VERIFY(name.find('\\') == TString::npos, "must be simple name");
        Path_ = TFsPath(name);

        Path_.ForceDelete();
        Path_.MkDir();
    }

    TTestDirectory::~TTestDirectory() {
        Path_.ForceDelete();
    }
}

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
        try {
            path.MkDir();
            UNIT_FAIL("freak case not excepted");
        } catch (...) {
        }
        try {
            path.MkDirs();
            UNIT_FAIL("freak case not excepted");
        } catch (...) {
        }
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
        try {
            TFsPath("sfsfsfsdfsfsdfdf").RenameTo("sdfsdf");
            UNIT_FAIL("excepting error");
        } catch (const TIoException&) {
            // expecting
        }
    }

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
        try {
            TFsPath("a/b/c").RelativePath(TFsPath("d/e"));
            UNIT_FAIL("excepting error");
        } catch (const TIoException&) {
            // expecting
        }
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
}
